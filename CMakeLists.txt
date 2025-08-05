cmake_minimum_required(VERSION 3.6)
project(MoPiRAG C CXX)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif()

# control Build Device: only 1 can set to be ON
option(BUILD_LINUX "build MoPiRAG for LINUX" ON)
option(BUILD_ANDROID "build MoPiRAG for Android" OFF)
option(BUILD_IOS "build MoPiRAG for iOS" OFF)

# if both ON, build android
if (BUILD_ANDROID)
    set(BUILD_LINUX OFF)
    set(BUILD_IOS OFF)
endif()
if (BUILD_IOS)
    set(BUILD_LINUX OFF)
    set(BUILD_ANDROID OFF)
endif()
if (BUILD_LINUX)
    set(BUILD_ANDROID OFF)
    set(BUILD_IOS OFF)
endif()

# set flags
if (BUILD_ANDROID)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -llog") # additional link to android/log
    set(DYNAMIC_LOAD_SYMBOLS ON) # dynamically load jni symbols in Java end
else()
    if (BUILD_IOS)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
        set(DYNAMIC_LOAD_SYMBOLS OFF) # build ordinary shared library logic
    endif()
endif()

# include dir
include_directories(
                    # sqlite include
                    ${CMAKE_CURRENT_LIST_DIR}/sqlite/sqlite-vec/
                    ${CMAKE_CURRENT_LIST_DIR}/sqlite/sqlite-vec/vendor/
                    ${CMAKE_CURRENT_LIST_DIR}/include/
                    # wrapper include
                    ${CMAKE_CURRENT_LIST_DIR}/dataset/include
                    # ${CMAKE_CURRENT_LIST_DIR}/wrapper/include
                    # MNN include
                    ${CMAKE_CURRENT_LIST_DIR}/MNN/tools/audio/include/
                    ${CMAKE_CURRENT_LIST_DIR}/MNN/transformers/llm/engine/include/
                    ${CMAKE_CURRENT_LIST_DIR}/MNN/include/
                    ${CMAKE_CURRENT_LIST_DIR}/MNN/tools/cv/include/
                    ${CMAKE_CURRENT_LIST_DIR}/MNN/3rd_party/
                    ${CMAKE_CURRENT_LIST_DIR}/sqlite/sqlite-vec/
                    )

# MNN flags
# compile MNN first
set(MNN_BUILD_LLM ON CACHE BOOL "Open MNN_BUILD_LLM" FORCE)
set(MNN_LOW_MEMORY ON CACHE BOOL "Open MNN_LOW_MEMORY" FORCE)
set(MNN_SUPPORT_TRANSFORMER_FUSE ON CACHE BOOL "Open MNN_SUPPORT_TRANSFORMER_FUSE" FORCE)
set(MNN_CPU_WEIGHT_DEQUANT_GEMM ON CACHE BOOL "Open MNN_CPU_WEIGHT_DEQUANT_GEMM" FORCE)
set(MNN_ARM82 ON CACHE BOOL "Open MNN_ARM82" FORCE)
set(LLM_SUPPORT_AUDIO ON CACHE BOOL "Open LLM_SUPPORT_AUDIO" FORCE)
set(MNN_BUILD_AUDIO ON CACHE BOOL "Open MNN_BUILD_AUDIO" FORCE)
set(LLM_SUPPORT_VISION ON CACHE BOOL "Open LLM_SUPPORT_VISION" FORCE)
add_definitions(-DLLM_SUPPORT_VISION)
set(MNN_BUILD_OPENCV ON CACHE BOOL "Open MNN_BUILD_OPENCV" FORCE)
set(MNN_IMGCODECS ON CACHE BOOL "Open MNN_IMGCODECS" FORCE)

if (BUILD_ANDROID)
set(MNN_USE_LOGCAT ON CACHE BOOL "Open MNN_USE_LOGCAT" FORCE)
set(MNN_OPENCL ON CACHE BOOL "Open MNN_OPENCL" FORCE)
endif()

if (BUILD_IOS)
set(MNN_OPENCL OFF CACHE BOOL "Close MNN_OPENCL" FORCE)
set(MNN_METAL ON CACHE BOOL "Open MNN_METAL" FORCE)
set(MNN_USE_THREAD_POOL OFF CACHE BOOL "Close MNN_USE_THREAD_POOL, iOS dosen't support CPU affinity" FORCE)
set(MNN_BUILD_SHARED_LIBS OFF CACHE BOOL "Close MNN_BUILD_SHARED_LIBS" FORCE)
set(MNN_SEP_BUILD OFF CACHE BOOL "Close MNN_SEP_BUILD" FORCE)
# Do not use MNN.framework, but mnn_wrapper
endif()

if (BUILD_ANDROID)
# jni include
include_directories(
    ${CMAKE_CURRENT_LIST_DIR}/android2/app/src/main/jni/include/
    )
endif()

# add dataset wrapper
# add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/dataset)
# add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/wrapper)

# add engine source directory
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/MNN)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/dataset)

# jni file
# if (BUILD_ANDROID)
# set(SRCS "${CMAKE_CURRENT_LIST_DIR}/android2/app/src/main/jni/iot_audio_jni.cpp")

# # compile dynamic so, support Linux/Mac
# add_library(mnn_jni SHARED ${SRCS})
# # add_library(dataset_jni SHARED "${CMAKE_CURRENT_LIST_DIR}/android2/app/src/main/jni/dataset_jni.cpp")
# # target_link_libraries(dataset_jni dataset)
# # target_link_libraries(mnn_jni dataset mnn_wrapper)
# target_link_libraries(mnn_jni mnn_wrapper)


# set(MNN_LLM_LIBS "")
# # list(APPEND MNN_LLM_LIBS $<TARGET_FILE:dataset>)
# # MNN
# list(APPEND MNN_LLM_LIBS $<TARGET_FILE:MNN>)
# list(APPEND MNN_LLM_LIBS $<TARGET_FILE:MNN_Express>)
# list(APPEND MNN_LLM_LIBS $<TARGET_FILE:MNNOpenCV>)
# list(APPEND MNN_LLM_LIBS $<TARGET_FILE:MNN_CL>)
# list(APPEND MNN_LLM_LIBS $<TARGET_FILE:MNNAudio>)
# list(APPEND MNN_LLM_LIBS $<TARGET_FILE:mnn_wrapper>)

# # after the last one finished
# add_custom_command(TARGET mnn_jni POST_BUILD
#     COMMAND ${CMAKE_COMMAND} -E copy
#             ${MNN_LLM_LIBS}
#             ${CMAKE_CURRENT_LIST_DIR}/android2/app/src/main/jni/libs/arm64-v8a)
# endif()

add_library(sqlite-vec SHARED
            ${CMAKE_CURRENT_LIST_DIR}/sqlite/sqlite-vec/sqlite-vec.c 
            ${CMAKE_CURRENT_LIST_DIR}/sqlite/sqlite-vec/vendor/sqlite3.c)
add_definitions(-DSQLITE_CORE)

FILE(GLOB RAG_SRCS "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp")
add_library(rag SHARED ${RAG_SRCS})
target_link_libraries(rag sqlite-vec dataset MNN MNN_Express llm)

add_executable(rag_demo ${CMAKE_CURRENT_LIST_DIR}/demo/rag_demo.cpp)
target_link_libraries(rag_demo rag)

