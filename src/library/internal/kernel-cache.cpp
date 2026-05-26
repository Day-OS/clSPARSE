/* ************************************************************************
 * Copyright 2015 Vratis, Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ************************************************************************ */
/* PATCHED: disk-based kernel binary cache (C++11-compatible, POSIX I/O). */

#include "kernel-cache.hpp"

#include <iostream>
#include <iterator>
#include <fstream>
#include <string>
#include <vector>
#include <functional>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>
#include <sys/types.h>
#include "source-provider.hpp"

KernelCache KernelCache::singleton;

KernelCache::KernelCache()
{
}

cl::Kernel KernelCache::get(cl::CommandQueue& queue,
                            const std::string& program_name,
                            const std::string& kernel_name,
                            const std::string& params)
{
    return getInstance().getKernel(queue, program_name, kernel_name, params);
}

cl::Kernel KernelCache::getKernel(cl::CommandQueue& queue,
                                  const std::string& program_name,
                                  const std::string& kernel_name,
                                  const std::string& params)
{
#if (BUILD_CLVERSION >= 120)
    std::string _params = " -cl-kernel-arg-info -cl-std=CL1.2";
#else
    std::string _params = " -cl-std=CL1.1";
#endif
    if (params.length() > 0)
    {
        if (params.at(0) != ' ')
            _params.append(" ");
        _params.append(params);
    }
    std::string key;
    key.append("[" + program_name + "/" + kernel_name + "]");
    key.append(_params);

    auto hash = rsHash(key);

    auto kernel_iterator = kernel_map.find(hash);
    if (kernel_iterator != kernel_map.end())
        return kernel_iterator->second;

    const cl::Program* program = getProgram(queue, program_name, _params);
    if (program == nullptr)
    {
        std::cout << "Problem with getting program [" << program_name << "] " << std::endl;
        return cl::Kernel();
    }

    cl_int status;
    cl::Kernel kernel(*program, kernel_name.c_str(), &status);

    if (status != CL_SUCCESS)
    {
        std::cout << "Problem with creating kernel [" << kernel_name << "]" << std::endl;
        delete program;
        return cl::Kernel();
    }

    kernel_map[hash] = kernel;
    delete program;
    return kernel;
}

// ── disk cache helpers (C++11 / POSIX) ──────────────────────────────────────

namespace {

std::string disk_cache_dir()
{
    const char* xdg  = std::getenv("XDG_CACHE_HOME");
    const char* home = std::getenv("HOME");
    std::string base = xdg ? std::string(xdg)
                           : (home ? std::string(home) + "/.cache" : std::string("/tmp"));
    return base + "/clsparse_kernel_cache";
}

void create_dir_recursive(const std::string& path)
{
    struct stat st;
    if (stat(path.c_str(), &st) == 0)
        return;
    auto slash = path.rfind('/');
    if (slash != std::string::npos && slash > 0)
        create_dir_recursive(path.substr(0, slash));
    mkdir(path.c_str(), 0755);
}

bool file_exists(const std::string& path)
{
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

std::string binary_path_for(const std::string& cache_key)
{
    auto h   = std::hash<std::string>{}(cache_key);
    auto dir = disk_cache_dir();
    create_dir_recursive(dir);
    return dir + "/" + std::to_string(h) + ".clbin";
}

bool save_program_binary(cl_program prog, const std::string& path)
{
    cl_uint num_devices = 0;
    clGetProgramInfo(prog, CL_PROGRAM_NUM_DEVICES,
                     sizeof(num_devices), &num_devices, nullptr);
    if (num_devices == 0) return false;

    std::vector<size_t> sizes(num_devices, 0);
    clGetProgramInfo(prog, CL_PROGRAM_BINARY_SIZES,
                     num_devices * sizeof(size_t), sizes.data(), nullptr);
    if (sizes[0] == 0) return false;

    std::vector<unsigned char> binary(sizes[0]);
    unsigned char* ptr = binary.data();
    clGetProgramInfo(prog, CL_PROGRAM_BINARIES,
                     sizeof(unsigned char*), &ptr, nullptr);

    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) return false;
    ofs.write(reinterpret_cast<const char*>(binary.data()),
              static_cast<std::streamsize>(binary.size()));
    return ofs.good();
}

cl::Program* load_program_binary(cl::Context& context,
                                 std::vector<cl::Device>& devices,
                                 const std::string& path,
                                 const std::string& params)
{
    if (!file_exists(path)) return nullptr;

    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return nullptr;

    ifs.seekg(0, std::ios::end);
    auto file_size = static_cast<std::size_t>(ifs.tellg());
    ifs.seekg(0, std::ios::beg);
    if (file_size == 0) return nullptr;

    std::vector<unsigned char> data(file_size);
    ifs.read(reinterpret_cast<char*>(data.data()),
             static_cast<std::streamsize>(file_size));
    if (!ifs) return nullptr;

    cl_int    binary_status = CL_SUCCESS;
    cl_int    status        = CL_SUCCESS;
    cl_device_id  dev  = devices[0]();
    size_t        sz   = data.size();
    const unsigned char* ptr = data.data();

    cl_program raw = clCreateProgramWithBinary(
        context(), 1, &dev, &sz, &ptr, &binary_status, &status);

    if (status != CL_SUCCESS || binary_status != CL_SUCCESS) {
        if (raw) clReleaseProgram(raw);
        return nullptr;
    }

    status = clBuildProgram(raw, 1, &dev, params.c_str(), nullptr, nullptr);
    if (status != CL_SUCCESS) {
        clReleaseProgram(raw);
        return nullptr;
    }

    return new cl::Program(raw);
}

} // namespace

// ── getProgram: loads from disk cache or compiles from source ─────────────────

const cl::Program* KernelCache::getProgram(cl::CommandQueue& queue,
                                           const std::string& program_name,
                                           const std::string& params)
{
    cl_int status;
    cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();

    const char* source = SourceProvider::GetSource(program_name);
    if (source == nullptr) {
        std::cout << "Source not found [" << program_name << "]" << std::endl;
        return nullptr;
    }
    size_t size = std::char_traits<char>::length(source);

    std::vector<cl::Device> devices;
    auto d = queue.getInfo<CL_QUEUE_DEVICE>(&status);
    if (status != CL_SUCCESS) {
        std::cout << "Problem with getting device from queue: " << status << std::endl;
        return nullptr;
    }
    devices.push_back(d);

    // Cache key: uniquely identifies the compiled output.
    // Keying on driver version ensures recompilation after driver updates.
    std::string device_name    = d.getInfo<CL_DEVICE_NAME>();
    std::string driver_version = d.getInfo<CL_DRIVER_VERSION>();
    std::string cache_key = program_name + "|" + params
                          + "|" + device_name + "|" + driver_version
                          + "|" + std::string(source, size);
    std::string bpath = binary_path_for(cache_key);

    // Fast path: pre-compiled binary on disk
    cl::Program* cached = load_program_binary(context, devices, bpath, params);
    if (cached) return cached;

    // Slow path: compile from source (first run)
    std::cout << "[clSPARSE] Compilando kernel '" << program_name
              << "' (primeira vez — pode levar minutos)..." << std::endl;

    cl::Program::Sources sources;
    sources.push_back(std::make_pair(source, size));

    cl::Program* program = new cl::Program(context, sources);
    status = program->build(devices, params.c_str());

    if (status == CL_INVALID_BUILD_OPTIONS) {
        std::cout << "Error during program compilation err = " << status
                  << "(CL_INVALID_BUILD_OPTIONS)\n"
                     "\tCheck the definition of the program parameters"
                  << std::endl;
        delete program;
        return nullptr;
    }
    else if (status != CL_SUCCESS) {
        std::cout << "#######################################" << std::endl;
        std::cout << "sources: ";
        for (auto& s : sources)
            std::cout << s.first << std::endl;
        std::cout << std::endl;
        std::cout << "---------------------------------------" << std::endl;
        std::cout << "parameters: " << params << std::endl;
        std::cout << "---------------------------------------" << std::endl;
        cl_int getBuildInfoStatus;
        auto log = program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(
            devices[0], &getBuildInfoStatus);
        if (getBuildInfoStatus == CL_SUCCESS)
            std::cout << log << std::endl;
        else
            std::cout << "Problem with obtaining build log info: "
                      << getBuildInfoStatus << std::endl;
        std::cout << "#######################################" << std::endl;
        delete program;
        return nullptr;
    }

    // Save compiled binary to disk for future runs
    if (save_program_binary((*program)(), bpath))
        std::cout << "[clSPARSE] Kernel '" << program_name
                  << "' salvo em cache." << std::endl;

    return program;
}

KernelCache& KernelCache::getInstance()
{
    return singleton;
}

unsigned int KernelCache::rsHash(const std::string& key)
{
    unsigned int b    = 378551;
    unsigned int a    = 63689;
    unsigned int hash = 0;

    for (auto c : key)
    {
        hash = hash * a + static_cast<unsigned char>(c);
        a    = a * b;
    }
    return hash;
}
