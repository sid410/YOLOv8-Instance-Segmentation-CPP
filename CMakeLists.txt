cmake_minimum_required(VERSION 3.0.0)
project(YOLOv8CPP)

set(ONNXRUNTIME_DIR ${CMAKE_SOURCE_DIR}/onnxruntime) # onnxruntime root

message("The root onnxruntime directory is: ${ONNXRUNTIME_DIR}")

find_package(OpenCV REQUIRED)

# --- Configure your project files ---
include_directories(include) # Include your header files directory

# Recursively collect all source files under 'src' directory
file(GLOB_RECURSE CURR_SOURCES src/*.cpp)

# Create the executable
add_executable(YOLOv8CPP ${CURR_SOURCES})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

target_include_directories(YOLOv8CPP PRIVATE "${ONNXRUNTIME_DIR}/include")

target_compile_features(YOLOv8CPP PRIVATE cxx_std_17)

target_link_libraries(YOLOv8CPP ${OpenCV_LIBS})

target_link_libraries(YOLOv8CPP "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
