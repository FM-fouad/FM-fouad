cmake_minimum_required(VERSION 3.0.0)
project(detect_vid VERSION 0.1.0)

include(CTest)
enable_testing()


find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(VISP REQUIRED)

add_executable(DetectWSound DetectionWSound.cpp)


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX__STANDARD_REQUIRED ON)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${VISP_INCLUDE_DIRS}
)

target_link_libraries(DetectWSound ${CMAKE_THREAD_LIBS_INIT} ${VISP_LIBRARIES} -lespeak-ng)
