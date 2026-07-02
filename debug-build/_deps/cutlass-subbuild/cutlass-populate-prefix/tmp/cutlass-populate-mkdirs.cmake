# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/hx/cutlass-main")
  file(MAKE_DIRECTORY "/home/hx/cutlass-main")
endif()
file(MAKE_DIRECTORY
  "/home/hx/vLLM/vllm-main/debug-build/_deps/cutlass-build"
  "/home/hx/vLLM/vllm-main/debug-build/_deps/cutlass-subbuild/cutlass-populate-prefix"
  "/home/hx/vLLM/vllm-main/debug-build/_deps/cutlass-subbuild/cutlass-populate-prefix/tmp"
  "/home/hx/vLLM/vllm-main/debug-build/_deps/cutlass-subbuild/cutlass-populate-prefix/src/cutlass-populate-stamp"
  "/home/hx/vLLM/vllm-main/debug-build/_deps/cutlass-subbuild/cutlass-populate-prefix/src"
  "/home/hx/vLLM/vllm-main/debug-build/_deps/cutlass-subbuild/cutlass-populate-prefix/src/cutlass-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/hx/vLLM/vllm-main/debug-build/_deps/cutlass-subbuild/cutlass-populate-prefix/src/cutlass-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/hx/vLLM/vllm-main/debug-build/_deps/cutlass-subbuild/cutlass-populate-prefix/src/cutlass-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
