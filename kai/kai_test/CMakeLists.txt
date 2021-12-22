set(PROJECT_NAME kai_test)

################################################################################
# Source groups
################################################################################
set(___src1
    src/test1/test_kai_array.cpp
    src/test1/test_kai_value.cpp
    
)
source_group("src1" FILES ${___src1})

file(GLOB_RECURSE SRC_FILES  ${CMAKE_SOURCE_DIR}/kai/kai_engine/src/*.cpp)
set(ALL_FILES
    ${___src1} ${SRC_FILES}
)
#MESSAGE(STATUS ${CMAKE_SOURCE_DIR}/kai/kai_engine/src)

################################################################################
# Target
################################################################################
add_executable(${PROJECT_NAME} ${ALL_FILES})
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

include(GoogleTest)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})


set(ROOT_NAMESPACE kai_test)
################################################################################
# Compile definitions
################################################################################
if(MSVC)
   #target_compile_definitions(${PROJECT_NAME} PRIVATE
   list(APPEND COMPILE_DEF 
    "$<$<CONFIG:Debug>:"
        "_DEBUG" 
    ">"
   "KAI_LIBRARY" "KAI2021_WINDOWS;" "_CONSOLE;" "_WINDLL" "_MBCS" "_CRT_SECURE_NO_WARNINGS"
    )
endif()

if(NO_RANDOM)
    list(APPEND COMPILE_DEF NORANDOM)
endif()

target_compile_definitions(${PROJECT_NAME} PRIVATE ${COMPILE_DEF})
################################################################################
# Include directories
################################################################################
if(MSVC)
    list(APPEND INCLUDE_DIRECTORY
        "${CMAKE_CURRENT_SOURCE_DIR}/../../install/opencv/build/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../install/MySQL/Connector C++ 8.0/include;"
        "${CMAKE_CURRENT_SOURCE_DIR}/../kai_engine/src"
    )
else(UNIX)
    list(APPEND INCLUDE_DIRECTORY
        ${OpenCV_INCLUDE_DIRS}    
    )
endif()
list(APPEND INCLUDE_DIRECTORY  "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}" ${gtest_SOURCE_DIR}/include )
target_include_directories(${PROJECT_NAME} PUBLIC ${INCLUDE_DIRECTORY})
################################################################################
# Compile definitions
################################################################################
if(NO_RANDOM)
    list(APPEND COMPILE_DEF NORANDOM)
endif()
target_compile_definitions(${PROJECT_NAME} PRIVATE ${COMPILE_DEF})

################################################################################
# Library depency
################################################################################
if(MSVC)
    list(APPEND ADDITIONAL_LIBRARY_DEPENDENCIES
        "$<$<CONFIG:Debug>:"
            "opencv_world452d"
        ">"
        "$<$<CONFIG:Release>:"
            "opencv_world452"
        ">"
        cudart_static  curand  "mysqlcppconn-static" "mysqlcppconn8"  "mysqlcppconn8-static" 
    )
else(UNIX)
    list(APPEND ADDITIONAL_LIBRARY_DEPENDENCIES "curand" "stdc++fs" ${OpenCV_LIBS} )
endif()
 list(APPEND ADDITIONAL_LIBRARY_DEPENDENCIES gtest gtest_main)
target_link_libraries(${PROJECT_NAME} PUBLIC "${ADDITIONAL_LIBRARY_DEPENDENCIES}")

################################################################################
# Library path
################################################################################
 if (MSVC)
     list(APPEND LIBRARY_PATH
        "${CMAKE_SOURCE_DIR}/install/MySQL/Connector C++ 8.0/lib64/vs14;"
        "${CMAKE_SOURCE_DIR}/install/opencv/build/${CMAKE_VS_PLATFORM_NAME}/vc14/lib"   
    )
endif()
target_link_directories(${PROJECT_NAME} PUBLIC ${LIBRARY_PATH} ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES} )
gtest_discover_tests(${PROJECT_NAME})