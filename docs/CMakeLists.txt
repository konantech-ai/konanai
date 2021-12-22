find_package(Doxygen)

if (DOXYGEN_FOUND)
    # Find all the public headers
    #set(doxyfile ${CMAKE_CURRENT_BINARY_DIR}/doxyfile)
    file(GLOB_RECURSE KAI_PUBLIC_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/../kai/*.h)

    #This will be the main output of our command
    set(DOXYGEN_INDEX_FILE ${CMAKE_CURRENT_SOURCE_DIR}/html/index.html)


    add_custom_command(OUTPUT ${DOXYGEN_INDEX_FILE}
                       DEPENDS ${KAI_PUBLIC_HEADERS}
                       COMMAND ${DOXYGEN_EXECUTABLE} Doxyfile
                       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                       MAIN_DEPENDENCY Doxyfile
                       COMMENT "Generating docs")


    add_custom_target(Doxygen ALL DEPENDS ${DOXYGEN_INDEX_FILE})
endif()