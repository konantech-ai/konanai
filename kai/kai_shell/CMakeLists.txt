set(PROJECT_NAME kai_shell)

################################################################################
# Source groups
################################################################################
set(___src
    "src/common.h"
    "src/kai_shell.cpp"
    "src/kai_shell.h"
    "src/mission.cpp"
    "src/mission.h"
    "src/data_feeder.cpp"
    "src/data_feeder.h"
    "src/reporter.h"
    "src/reporter.cpp"
)
source_group("src" FILES ${___src})

set(___src__missions
    "src/missions/abalone.cpp"
    "src/missions/abalone.h"
    "src/missions/flower.cpp"
    "src/missions/flower.h"
    "src/missions/mnist_reader.cpp" 
    "src/missions/mnist_reader.h"
    "src/missions/office31.cpp"
    "src/missions/office31.h"
    "src/missions/pulsar.cpp"
    "src/missions/pulsar.h"
    "src/missions/steel.cpp"
    "src/missions/steel.h"
    "src/missions/urban.cpp"
    "src/missions/urban.h"
    "src/missions/water.cpp"
    "src/missions/water.h"
    "src/missions/bert.cpp"
    "src/missions/bert.h"
    "src/missions/mini_bert.cpp"
    "src/missions/mini_bert.h"
    "src/missions/yolo3.cpp"
    "src/missions/yolo3.h"
)
source_group("src\\missions" FILES ${___src__missions})

set(___src__data_feeders
    "src/data_feeders/csv_feeder.cpp"    
    "src/data_feeders/csv_feeder.h"
    "src/data_feeders/folder_class_feeder.cpp"    
    "src/data_feeders/folder_class_feeder.h"
    "src/data_feeders/mnist_reader_feeder.cpp"
    "src/data_feeders/mnist_reader_feeder.h"
    "src/data_feeders/urban_feeder.cpp"
    "src/data_feeders/urban_feeder.h"
    "src/data_feeders/bert_feeder.h"
    "src/data_feeders/bert_feeder.cpp"
    "src/data_feeders/yolo3_feeder.cpp"
    "src/data_feeders/yolo3_feeder.h"
)
source_group("src\\data_feeders" FILES ${___src__data_feeders})

set(___src__reporters
    "src/reporters/bert_reporter.cpp"
    "src/reporters/bert_reporter.h"
    "src/reporters/yolo3_reporter.h"
    "src/reporters/yolo3_reporter.cpp"
    "src/reporters/abalone_reporter.h"
    "src/reporters/abalone_reporter.cpp"
    "src/reporters/binary_reporter.h"
    "src/reporters/binary_reporter.cpp"
    "src/reporters/select_reporter.h"
    "src/reporters/select_reporter.cpp"
    "src/reporters/office31_reporter.h"
    "src/reporters/office31_reporter.cpp"
    "src/reporters/mnist_reader_reporter.h"
    "src/reporters/mnist_reader_reporter.cpp"
)
source_group("src\\reporters" FILES ${___src__reporters})

set(___src__utils
    "src/utils/json_parser.cpp"
    "src/utils/json_parser.h"
    "src/utils/kvalue_ext.cpp"
    "src/utils/kvalue_ext.h"
    "src/utils/utils.cpp"
    "src/utils/utils.h"
   )
source_group("src\\utils" FILES ${___src__utils})

set(ALL_FILES
    ${___src}
    ${___src__missions}
    ${___src__data_feeders}
    ${___src__reporters}
    ${___src__utils}
)

################################################################################
# Target
################################################################################
add_executable(${PROJECT_NAME} ${ALL_FILES})

set(ROOT_NAMESPACE kaishell)

set_target_properties(${PROJECT_NAME} PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION_RELEASE "TRUE"
)



 
################################################################################
# Compile definitions
################################################################################
if(MSVC)
   #target_compile_definitions(${PROJECT_NAME} PRIVATE
   list(APPEND COMPILE_DEF 
    "$<$<CONFIG:Debug>:"
        "_DEBUG" 
    ">"
    "WIN64;"  "KAI2021_WINDOWS;" "_CONSOLE;"  "_MBCS" "_CRT_SECURE_NO_WARNINGS"
    )
endif()

target_compile_definitions(${PROJECT_NAME} PRIVATE ${COMPILE_DEF})

################################################################################
# Compile options
################################################################################
if(MSVC)
    list(APPEND COMPILE_OPT
        $<$<COMPILE_LANGUAGE:CXX>:
            /permissive-  /sdl /GS /Zc:wchar_t /Zc:forScope /Zc:inline /Zc:strictStrings- /fp:precise
        > 

   )
elseif(UNIX)
    list(APPEND COMPILE_OPT
    $<$<COMPILE_LANGUAGE:CXX>:
        -ffp-contract=off
        -m64
    >

    )
endif()
target_compile_options(${PROJECT_NAME} PRIVATE
        ${COMPILE_OPT}
        ${DEFAULT_CXX_DEBUG_INFORMATION_FORMAT};
        ${DEFAULT_CXX_EXCEPTION_HANDLING};
)

################################################################################
# LInker options
################################################################################
if(MSVC)
    list(APPEND ${LINKER_OPT}
        $<$<CONFIG:Debug>:
        >
        $<$<CONFIG:Release>:
            /OPT:REF; /OPT:ICF
        >
        /DYNAMICBASE /NXCOMPAT
    )
endif()
 target_link_options(${PROJECT_NAME} PRIVATE ${LINKER_OPT})


################################################################################
# Link Dependencies
################################################################################
add_dependencies(${PROJECT_NAME}
    kai_engine
)

set(ADDITIONAL_LIBRARY_DEPENDENCIES
    "kai_engine"
)
target_link_libraries(${PROJECT_NAME} PRIVATE "${ADDITIONAL_LIBRARY_DEPENDENCIES}")

target_link_directories(${PROJECT_NAME} PRIVATE
    "${OUTPUT_DIRECTORY}"
)

