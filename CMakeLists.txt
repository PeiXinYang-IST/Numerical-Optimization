cmake_minimum_required(VERSION 3.10)
project(MyProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB SRC_FILES src/*.cpp)
set(CMAKE_BUILD_TYPE Debug)

foreach(SRC_FILE ${SRC_FILES})
    get_filename_component(EXE_NAME ${SRC_FILE} NAME_WE)
    add_executable(${EXE_NAME} ${SRC_FILE})
    target_include_directories(${EXE_NAME} PRIVATE include)
    target_link_libraries(${EXE_NAME} PRIVATE pthread)
endforeach()