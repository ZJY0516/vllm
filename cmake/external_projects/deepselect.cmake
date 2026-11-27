include(FetchContent)

# If DEEPSELECT_SRC_DIR is set, DeepSelect is built from that directory
# instead of downloading.
# It can be set as an environment variable or passed as a cmake argument.
# The environment variable takes precedence.
if (DEFINED ENV{DEEPSELECT_SRC_DIR})
  set(DEEPSELECT_SRC_DIR $ENV{DEEPSELECT_SRC_DIR})
endif()

if(DEEPSELECT_SRC_DIR)
  FetchContent_Declare(
        deepselect
        SOURCE_DIR ${DEEPSELECT_SRC_DIR}
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
  )
else()
  FetchContent_Declare(
        deepselect
        GIT_REPOSITORY https://github.com/vllm-project/DeepSelect.git
        GIT_TAG d96d33afe1fab0d6066da49cdc91e64c2bee65ea
        GIT_PROGRESS TRUE
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
  )
endif()

FetchContent_MakeAvailable(deepselect)
message(STATUS "DeepSelect is available at ${deepselect_SOURCE_DIR}")

# Local changes on top of the pinned commit, applied to the fetched tree only
# (a DEEPSELECT_SRC_DIR tree is used as is). A tree that already has a patch,
# such as a reused _deps checkout, is left alone.
#   deepselect_bf16_topk2048_tier.patch: a max_topk 2048 tier for BF16 top-k
#   with 1024 < topk <= 2048 (the DSA sparse indexer's prefill top-k), the
#   same tuple as the max_topk 4096 tier otherwise; output indices and their
#   order are identical.
if(NOT DEEPSELECT_SRC_DIR)
  find_package(Git REQUIRED)
  foreach(_ds_patch
      "${CMAKE_SOURCE_DIR}/cmake/patches/deepselect_bf16_topk2048_tier.patch")
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" apply --reverse --check "${_ds_patch}"
      WORKING_DIRECTORY "${deepselect_SOURCE_DIR}"
      RESULT_VARIABLE _ds_patch_present
      OUTPUT_QUIET ERROR_QUIET)
    if(NOT _ds_patch_present EQUAL 0)
      message(STATUS "DeepSelect: applying ${_ds_patch}")
      execute_process(
        COMMAND "${GIT_EXECUTABLE}" apply "${_ds_patch}"
        WORKING_DIRECTORY "${deepselect_SOURCE_DIR}"
        COMMAND_ERROR_IS_FATAL ANY)
    endif()
  endforeach()
endif()

# DeepSelect kernels require CUDA 12.9+; 10.0f (family) covers SM100 and SM103.
set(DEEPSELECT_SUPPORT_ARCHS)
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.9)
    list(APPEND DEEPSELECT_SUPPORT_ARCHS "10.0f")
endif()

cuda_archs_loose_intersection(DEEPSELECT_ARCHS "${DEEPSELECT_SUPPORT_ARCHS}" "${CUDA_ARCHS}")
# When the SM100 family is targeted, also build SM103-specific code (B300 /
# GB300), which the driver picks over the family code on SM103; the kernels
# measured faster than the 10.0f build there.
if("10.0f" IN_LIST DEEPSELECT_ARCHS)
    list(APPEND DEEPSELECT_ARCHS "10.3a")
endif()
if(DEEPSELECT_ARCHS)
    message(STATUS "DeepSelect CUDA architectures: ${DEEPSELECT_ARCHS}")

    # All explicit template instantiations must be compiled: api.cpp
    # references every dispatch combination.
    file(GLOB DeepSelect_SOURCES CONFIGURE_DEPENDS
        ${deepselect_SOURCE_DIR}/csrc/cuda_kernels/v3/instantiations/*.cu
        ${deepselect_SOURCE_DIR}/csrc/cuda_kernels/v3_fp32/instantiations/*.cu
        ${deepselect_SOURCE_DIR}/csrc/cuda_kernels/v3_cluster/instantiations/*.cu
    )
    list(PREPEND DeepSelect_SOURCES ${deepselect_SOURCE_DIR}/csrc/api.cpp)

    set(DeepSelect_INCLUDES
        ${deepselect_SOURCE_DIR}/csrc
        ${deepselect_SOURCE_DIR}/csrc/3rdparty/cutlass/include
        ${deepselect_SOURCE_DIR}/csrc/3rdparty/kerutils/include
    )

    set_gencode_flags_for_srcs(
        SRCS "${DeepSelect_SOURCES}"
        CUDA_ARCHS "${DEEPSELECT_ARCHS}")

    set(VLLM_DEEPSELECT_GPU_FLAGS ${VLLM_GPU_FLAGS})
    list(APPEND VLLM_DEEPSELECT_GPU_FLAGS
        "-O3" "--expt-relaxed-constexpr" "--expt-extended-lambda"
        "--use_fast_math" "--ftz=false"
        "-U__CUDA_NO_HALF_OPERATORS__" "-U__CUDA_NO_HALF_CONVERSIONS__"
        "-U__CUDA_NO_HALF2_OPERATORS__" "-U__CUDA_NO_BFLOAT16_CONVERSIONS__")

    define_extension_target(
        _deepselect_C
        DESTINATION vllm
        LANGUAGE ${VLLM_GPU_LANG}
        SOURCES ${DeepSelect_SOURCES}
        COMPILE_FLAGS ${VLLM_DEEPSELECT_GPU_FLAGS}
        ARCHITECTURES ${VLLM_GPU_ARCHES}
        INCLUDE_DIRECTORIES ${DeepSelect_INCLUDES}
        USE_SABI 3
        WITH_SOABI)

    # Only use C-shim APIs available in PyTorch 2.10.
    # _deepselect_C is abi compatible with PyTorch >= TORCH_TARGET_VERSION.
    target_compile_definitions(_deepselect_C PRIVATE
        TORCH_TARGET_VERSION=0x020A000000000000ULL)

    # Needed to use cuda APIs from C-shim
    if(VLLM_GPU_LANG STREQUAL "CUDA")
        target_compile_definitions(_deepselect_C PRIVATE USE_CUDA)
    endif()

    # DeepSelect requires C++20
    target_compile_options(_deepselect_C PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-std=c++20>
        $<$<COMPILE_LANGUAGE:CUDA>:-std=c++20>)
else()
    message(STATUS "DeepSelect will not compile: unsupported CUDA architecture ${CUDA_ARCHS}")
    # Create an empty target for setup.py on unsupported systems
    add_custom_target(_deepselect_C)
endif()
