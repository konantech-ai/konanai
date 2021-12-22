project ("kai")
################################################################################
# Set target arch type if empty. Visual studio solution generator provides it.
################################################################################
if(NOT CMAKE_VS_PLATFORM_NAME)
    set(CMAKE_VS_PLATFORM_NAME "x64")
endif()
message("${CMAKE_VS_PLATFORM_NAME} architecture in use")

if(NOT ("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64"))
    message(FATAL_ERROR "${CMAKE_VS_PLATFORM_NAME} arch is not supported!")
endif()


################################################################################
# CUDA enable
################################################################################
enable_language(CXX CUDA) 
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()
#if(NOT DEFINED ${CMAKE_CUDA_ARCHITECTURES})
set(CMAKE_CUDA_ARCHITECTURES 61 75 86)
#endif()
message(STATUS "CUDA architectures set to ${CMAKE_CUDA_ARCHITECTURES}")

################################################################################
# C++ 
################################################################################
if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 14)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

cmake_policy(SET CMP0069 NEW)

################################################################################
# MSVC /ZI option enabled
################################################################################
if(MSVC)
  string(REGEX REPLACE "/Z[iI7]" ""
         CMAKE_CXX_FLAGS_DEBUG
         "${CMAKE_CXX_FLAGS_DEBUG}")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /ZI")
endif()


################################################################################
# Binary directory should be ${CMAKE_BINARY_DIR}/.. for debugging 
################################################################################
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/..)


################################################################################
# copy dll to ${CMAKE_BINARY_DIR}/.. for debugging
################################################################################
if (MSVC) 
    file(COPY "${CMAKE_SOURCE_DIR}/install/opencv/build/${CMAKE_VS_PLATFORM_NAME}/vc14/bin/opencv_world452.dll"
    DESTINATION  ${CMAKE_BINARY_DIR}/..)
    file(COPY "${CMAKE_SOURCE_DIR}/install/opencv/build/${CMAKE_VS_PLATFORM_NAME}/vc14/bin/opencv_world452d.dll"
    DESTINATION  ${CMAKE_BINARY_DIR}/..)

    file(COPY  "${CMAKE_SOURCE_DIR}/install/MySQL/Connector C++ 8.0/lib64/libcrypto-1_1-x64.dll"
    DESTINATION  ${CMAKE_BINARY_DIR}/..)
    file(COPY "${CMAKE_SOURCE_DIR}/install/MySQL/Connector C++ 8.0/lib64/libssl-1_1-x64.dll"
    DESTINATION  ${CMAKE_BINARY_DIR}/..)
    file(COPY "${CMAKE_SOURCE_DIR}/install/MySQL/Connector C++ 8.0/lib64/mysqlcppconn8-2-vs14.dll" 
    DESTINATION  ${CMAKE_BINARY_DIR}/..) 
endif()

################################################################################
# Sub-projects
################################################################################
add_subdirectory(kai_engine)
add_subdirectory(kai_shell)
#add_subdirectory(kai_test)

