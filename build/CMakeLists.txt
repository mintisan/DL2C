cmake_minimum_required(VERSION 3.18)

project(DL2C_Unified_ONNX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置C标准
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# 平台检测和配置
if(ANDROID)
    # Android 特定设置
    set(CMAKE_SYSTEM_NAME Android)
    set(CMAKE_SYSTEM_VERSION 21)
    set(CMAKE_ANDROID_ARCH_ABI arm64-v8a)
    
    # ONNX Runtime Android 路径配置
    set(ONNXRUNTIME_BUILD_DIR "/Users/mintisan/Workplaces/onnxruntime/build/Android/Release")
    set(ONNXRUNTIME_ROOT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-android-${ANDROID_ABI}")
    set(ONNXRUNTIME_INCLUDE_DIR "/Users/mintisan/Workplaces/onnxruntime/include")
    set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT_PATH}/lib")
    
    message(STATUS "配置Android构建")
    message(STATUS "ONNX Runtime构建目录: ${ONNXRUNTIME_BUILD_DIR}")
    message(STATUS "ONNX Runtime包含目录: ${ONNXRUNTIME_INCLUDE_DIR}")
else()
    # macOS/Linux 设置
    find_package(PkgConfig REQUIRED)
    
    # 尝试多种方式查找ONNX Runtime
    set(ONNXRUNTIME_FOUND FALSE)
    
    # 方法1: 查找系统安装的ONNX Runtime (参考工作版本)
    find_path(ONNXRUNTIME_INCLUDE_DIR
        NAMES onnxruntime_c_api.h onnxruntime_cxx_api.h
        PATHS 
            /usr/local/include/onnxruntime
            /opt/homebrew/include/onnxruntime
            /usr/include/onnxruntime
            ${CMAKE_CURRENT_SOURCE_DIR}/../onnxruntime-osx-x86_64-1.16.0/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../onnxruntime-osx-arm64-1.16.0/include
            /usr/local/include
            /opt/homebrew/include
            /usr/include
        PATH_SUFFIXES onnxruntime
    )
    
    find_library(ONNXRUNTIME_LIB
        NAMES onnxruntime
        PATHS 
            /usr/local/lib
            /opt/homebrew/lib
            /usr/lib
            ${CMAKE_CURRENT_SOURCE_DIR}/../onnxruntime-osx-x86_64-1.16.0/lib
            ${CMAKE_CURRENT_SOURCE_DIR}/../onnxruntime-osx-arm64-1.16.0/lib
    )
    
    if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIB)
        set(ONNXRUNTIME_FOUND TRUE)
        message(STATUS "找到系统安装的ONNX Runtime")
    endif()
    
    # 方法2: 如果未找到，尝试从环境变量获取
    if(NOT ONNXRUNTIME_FOUND AND DEFINED ENV{ONNXRUNTIME_ROOT})
        set(ONNXRUNTIME_ROOT $ENV{ONNXRUNTIME_ROOT})
        find_path(ONNXRUNTIME_INCLUDE_DIR 
            NAMES onnxruntime_c_api.h onnxruntime_cxx_api.h
            PATHS ${ONNXRUNTIME_ROOT}/include
            PATH_SUFFIXES onnxruntime
        )
        find_library(ONNXRUNTIME_LIB
            NAMES onnxruntime
            PATHS ${ONNXRUNTIME_ROOT}/lib
        )
        if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIB)
            set(ONNXRUNTIME_FOUND TRUE)
            message(STATUS "找到环境变量指定的ONNX Runtime")
        endif()
    endif()
    
    # 方法3: 如果仍未找到，提供安装指导 (参考工作版本)
    if(NOT ONNXRUNTIME_FOUND)
        message(FATAL_ERROR 
            "未找到ONNX Runtime库！\n"
            "请安装ONNX Runtime:\n"
            "1. 使用Homebrew: brew install onnxruntime\n"
            "2. 或下载预编译版本: https://github.com/microsoft/onnxruntime/releases\n"
            "3. 并将其放在build目录下\n"
            "\n"
            "搜索的路径包括:\n"
            "  - /usr/local/include/onnxruntime, /usr/local/lib\n"
            "  - /opt/homebrew/include/onnxruntime, /opt/homebrew/lib\n"
            "  - build目录下的预编译版本"
        )
    endif()
    
    message(STATUS "配置macOS构建")
    message(STATUS "ONNX Runtime包含目录: ${ONNXRUNTIME_INCLUDE_DIR}")
    message(STATUS "ONNX Runtime库: ${ONNXRUNTIME_LIB}")
endif()

# 包含目录
include_directories(${ONNXRUNTIME_INCLUDE_DIR})
if(ANDROID)
    include_directories(${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime)
    include_directories(${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime/core/session)
endif()

# 源文件
set(CPP_UNIFIED_SOURCES 
    ../../inference/cpp_inference.cpp
)

set(C_UNIFIED_SOURCES 
    ../../inference/c_inference.c
)

# ========================================================
# C++ 统一版本
# ========================================================

add_executable(cpp_inference ${CPP_UNIFIED_SOURCES})

if(ANDROID)
    # Android特定的库链接
    
    # ONNX Runtime 核心库（按依赖顺序）
    set(ONNXRUNTIME_LIBS
        ${ONNXRUNTIME_BUILD_DIR}/libonnxruntime_session.a
        ${ONNXRUNTIME_BUILD_DIR}/libonnxruntime_providers.a  
        ${ONNXRUNTIME_BUILD_DIR}/libonnxruntime_framework.a
        ${ONNXRUNTIME_BUILD_DIR}/libonnxruntime_graph.a
        ${ONNXRUNTIME_BUILD_DIR}/libonnxruntime_optimizer.a
        ${ONNXRUNTIME_BUILD_DIR}/libonnxruntime_util.a
        ${ONNXRUNTIME_BUILD_DIR}/libonnxruntime_mlas.a
        ${ONNXRUNTIME_BUILD_DIR}/libonnxruntime_common.a
        ${ONNXRUNTIME_BUILD_DIR}/libonnxruntime_flatbuffers.a
        ${ONNXRUNTIME_BUILD_DIR}/libonnxruntime_lora.a
    )

    # ONNX Protocol Buffer 库
    set(ONNX_LIBS
        ${ONNXRUNTIME_BUILD_DIR}/_deps/onnx-build/libonnx.a
        ${ONNXRUNTIME_BUILD_DIR}/_deps/onnx-build/libonnx_proto.a
    )

    # 基础第三方库
    set(BASIC_THIRD_PARTY_LIBS
        ${ONNXRUNTIME_BUILD_DIR}/_deps/protobuf-build/libprotobuf-lite.a
        ${ONNXRUNTIME_BUILD_DIR}/_deps/pytorch_cpuinfo-build/libcpuinfo.a
        ${ONNXRUNTIME_BUILD_DIR}/_deps/re2-build/libre2.a
    )

    # Abseil C++ 核心库
    set(ABSEIL_CORE_LIBS
        ${ONNXRUNTIME_BUILD_DIR}/_deps/abseil_cpp-build/absl/container/libabsl_raw_hash_set.a
        ${ONNXRUNTIME_BUILD_DIR}/_deps/abseil_cpp-build/absl/strings/libabsl_strings.a
        ${ONNXRUNTIME_BUILD_DIR}/_deps/abseil_cpp-build/absl/strings/libabsl_strings_internal.a
        ${ONNXRUNTIME_BUILD_DIR}/_deps/abseil_cpp-build/absl/strings/libabsl_string_view.a
        ${ONNXRUNTIME_BUILD_DIR}/_deps/abseil_cpp-build/absl/strings/libabsl_str_format_internal.a
        ${ONNXRUNTIME_BUILD_DIR}/_deps/abseil_cpp-build/absl/base/libabsl_base.a
        ${ONNXRUNTIME_BUILD_DIR}/_deps/abseil_cpp-build/absl/base/libabsl_log_severity.a
        ${ONNXRUNTIME_BUILD_DIR}/_deps/abseil_cpp-build/absl/base/libabsl_raw_logging_internal.a
        ${ONNXRUNTIME_BUILD_DIR}/_deps/abseil_cpp-build/absl/base/libabsl_throw_delegate.a
        ${ONNXRUNTIME_BUILD_DIR}/_deps/abseil_cpp-build/absl/base/libabsl_spinlock_wait.a
        ${ONNXRUNTIME_BUILD_DIR}/_deps/abseil_cpp-build/absl/synchronization/libabsl_synchronization.a
        ${ONNXRUNTIME_BUILD_DIR}/_deps/abseil_cpp-build/absl/synchronization/libabsl_graphcycles_internal.a
        ${ONNXRUNTIME_BUILD_DIR}/_deps/abseil_cpp-build/absl/synchronization/libabsl_kernel_timeout_internal.a
        ${ONNXRUNTIME_BUILD_DIR}/_deps/abseil_cpp-build/absl/time/libabsl_time.a
        ${ONNXRUNTIME_BUILD_DIR}/_deps/abseil_cpp-build/absl/time/libabsl_civil_time.a
        ${ONNXRUNTIME_BUILD_DIR}/_deps/abseil_cpp-build/absl/time/libabsl_time_zone.a
        ${ONNXRUNTIME_BUILD_DIR}/_deps/abseil_cpp-build/absl/hash/libabsl_hash.a
        ${ONNXRUNTIME_BUILD_DIR}/_deps/abseil_cpp-build/absl/hash/libabsl_city.a
        ${ONNXRUNTIME_BUILD_DIR}/_deps/abseil_cpp-build/absl/hash/libabsl_low_level_hash.a
        ${ONNXRUNTIME_BUILD_DIR}/_deps/abseil_cpp-build/absl/numeric/libabsl_int128.a
    )

    # Abseil C++ 日志库
    set(ABSEIL_LOG_LIBS
        ${ONNXRUNTIME_BUILD_DIR}/_deps/abseil_cpp-build/absl/log/libabsl_log_internal_message.a
        ${ONNXRUNTIME_BUILD_DIR}/_deps/abseil_cpp-build/absl/log/libabsl_log_internal_format.a
        ${ONNXRUNTIME_BUILD_DIR}/_deps/abseil_cpp-build/absl/log/libabsl_log_internal_globals.a
        ${ONNXRUNTIME_BUILD_DIR}/_deps/abseil_cpp-build/absl/log/libabsl_log_internal_log_sink_set.a
        ${ONNXRUNTIME_BUILD_DIR}/_deps/abseil_cpp-build/absl/log/libabsl_log_sink.a
        ${ONNXRUNTIME_BUILD_DIR}/_deps/abseil_cpp-build/absl/log/libabsl_log_globals.a
        ${ONNXRUNTIME_BUILD_DIR}/_deps/abseil_cpp-build/absl/log/libabsl_log_internal_proto.a
        ${ONNXRUNTIME_BUILD_DIR}/_deps/abseil_cpp-build/absl/log/libabsl_log_internal_structured_proto.a
        ${ONNXRUNTIME_BUILD_DIR}/_deps/abseil_cpp-build/absl/log/libabsl_log_internal_nullguard.a
        ${ONNXRUNTIME_BUILD_DIR}/_deps/abseil_cpp-build/absl/log/libabsl_log_internal_check_op.a
    )

    # Abseil C++ 字符串和容器库
    set(ABSEIL_CONTAINER_LIBS
        ${ONNXRUNTIME_BUILD_DIR}/_deps/abseil_cpp-build/absl/container/libabsl_hashtablez_sampler.a
        ${ONNXRUNTIME_BUILD_DIR}/_deps/abseil_cpp-build/absl/strings/libabsl_cord.a
        ${ONNXRUNTIME_BUILD_DIR}/_deps/abseil_cpp-build/absl/strings/libabsl_cord_internal.a
        ${ONNXRUNTIME_BUILD_DIR}/_deps/abseil_cpp-build/absl/strings/libabsl_cordz_functions.a
        ${ONNXRUNTIME_BUILD_DIR}/_deps/abseil_cpp-build/absl/strings/libabsl_cordz_handle.a
        ${ONNXRUNTIME_BUILD_DIR}/_deps/abseil_cpp-build/absl/strings/libabsl_cordz_info.a
        ${ONNXRUNTIME_BUILD_DIR}/_deps/abseil_cpp-build/absl/profiling/libabsl_exponential_biased.a
    )

    # Abseil C++ 调试和其他库
    set(ABSEIL_OTHER_LIBS
        ${ONNXRUNTIME_BUILD_DIR}/_deps/abseil_cpp-build/absl/debugging/libabsl_stacktrace.a
        ${ONNXRUNTIME_BUILD_DIR}/_deps/abseil_cpp-build/absl/debugging/libabsl_symbolize.a
        ${ONNXRUNTIME_BUILD_DIR}/_deps/abseil_cpp-build/absl/debugging/libabsl_debugging_internal.a
        ${ONNXRUNTIME_BUILD_DIR}/_deps/abseil_cpp-build/absl/debugging/libabsl_demangle_internal.a
        ${ONNXRUNTIME_BUILD_DIR}/_deps/abseil_cpp-build/absl/debugging/libabsl_demangle_rust.a
        ${ONNXRUNTIME_BUILD_DIR}/_deps/abseil_cpp-build/absl/debugging/libabsl_examine_stack.a
        ${ONNXRUNTIME_BUILD_DIR}/_deps/abseil_cpp-build/absl/debugging/libabsl_leak_check.a
        ${ONNXRUNTIME_BUILD_DIR}/_deps/abseil_cpp-build/absl/debugging/libabsl_utf8_for_code_point.a
        ${ONNXRUNTIME_BUILD_DIR}/_deps/abseil_cpp-build/absl/base/libabsl_malloc_internal.a
        ${ONNXRUNTIME_BUILD_DIR}/_deps/abseil_cpp-build/absl/base/libabsl_strerror.a
        ${ONNXRUNTIME_BUILD_DIR}/_deps/abseil_cpp-build/absl/base/libabsl_tracing_internal.a
    )

    # 链接所有库（Android）
    target_link_libraries(cpp_inference 
        ${ONNXRUNTIME_LIBS}
        ${ONNX_LIBS}
        ${BASIC_THIRD_PARTY_LIBS}
        ${ABSEIL_CORE_LIBS}
        ${ABSEIL_LOG_LIBS}
        ${ABSEIL_CONTAINER_LIBS}
        ${ABSEIL_OTHER_LIBS}
        log
        m
        dl
        -static-libgcc
        -static-libstdc++
    )
    
    # Android编译选项
    target_compile_options(cpp_inference PRIVATE
        -ffunction-sections
        -fdata-sections
        -DONNX_MINIMAL_BUILD
        -DORT_MINIMAL_BUILD
        -D__ANDROID__
        -Os  # 优化大小
    )

    # Android链接选项
    target_link_options(cpp_inference PRIVATE
        -Wl,--gc-sections
        -Wl,--strip-all
        -static-libstdc++
    )
    
    # Android输出目录
    set_target_properties(cpp_inference
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "../../android_executables/arm64-v8a"
    )
    
else()
    # macOS链接
    target_link_libraries(cpp_inference ${ONNXRUNTIME_LIB})
    
    # macOS输出目录
    set_target_properties(cpp_inference
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "../../inference"
    )
endif()

# ========================================================
# C 统一版本
# ========================================================

add_executable(c_inference ${C_UNIFIED_SOURCES})

if(ANDROID)
    # C版本需要用C++链接器链接ONNX Runtime
    set_target_properties(c_inference PROPERTIES
        LINKER_LANGUAGE CXX
    )

    # C版本链接所有必需的ONNX Runtime库（与C++版本相同）
    target_link_libraries(c_inference
        ${ONNXRUNTIME_LIBS}
        ${ONNX_LIBS}
        ${BASIC_THIRD_PARTY_LIBS}
        ${ABSEIL_CORE_LIBS}
        ${ABSEIL_LOG_LIBS}
        ${ABSEIL_CONTAINER_LIBS}
        ${ABSEIL_OTHER_LIBS}
        log
        m
        dl
        -static-libgcc
        -static-libstdc++
    )

    # Android C版本编译选项
    target_compile_options(c_inference PRIVATE
        -ffunction-sections
        -fdata-sections
        -DONNX_MINIMAL_BUILD
        -DORT_MINIMAL_BUILD
        -D__ANDROID__
        -Os  # 优化大小
        -std=c99
    )

    # Android C版本链接选项
    target_link_options(c_inference PRIVATE
        -Wl,--gc-sections
        -Wl,--strip-all
        -static-libstdc++
        -static-libgcc
    )
    
    # Android C版本输出目录
    set_target_properties(c_inference
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "../../android_executables/arm64-v8a"
    )
    
else()
    # macOS C版本链接
    target_link_libraries(c_inference ${ONNXRUNTIME_LIB})
    
    # macOS C版本输出目录
    set_target_properties(c_inference
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "../../inference"
    )
endif()

# 显示配置信息
if(ANDROID)
    message(STATUS "构建Android统一版本:")
    message(STATUS "  - cpp_inference -> ../../android_executables/arm64-v8a/")
    message(STATUS "  - c_inference -> ../../android_executables/arm64-v8a/")
else()
    message(STATUS "构建macOS统一版本:")
    message(STATUS "  - cpp_inference -> ../../inference/")
    message(STATUS "  - c_inference -> ../../inference/")
endif() 