cmake_minimum_required(VERSION 3.22.1)
project(YOLOv8Seg)

set(ONNXRUNTIME_DIR ${CMAKE_SOURCE_DIR}/onnxruntime) # onnxruntime root

find_package(OpenCV REQUIRED)

# --- Configure your project files ---
include_directories(include) # Include your header files directory

# Recursively collect all source files under 'src' directory
file(GLOB_RECURSE CURR_SOURCES src/*.cpp)

# Create the executable
add_executable(${PROJECT_NAME} ${CURR_SOURCES})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

target_include_directories(${PROJECT_NAME} PRIVATE "${ONNXRUNTIME_DIR}/include")

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

target_link_libraries(${PROJECT_NAME}
                      "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
