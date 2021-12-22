set(PROJECT_NAME kai_engine)

################################################################################
# Source groups
################################################################################
set(src__api
    "src/api/kai_api_array.cpp"
    "src/api/kai_api_common.h"
    "src/api/kai_api_dataset.cpp"
    "src/api/kai_api_expression.cpp"
    "src/api/kai_api_layer.cpp"
    "src/api/kai_api_library.cpp"
    "src/api/kai_api_model.cpp"
    "src/api/kai_api_network.cpp"
    "src/api/kai_api_optimizer.cpp"
    "src/api/kai_api_shell.cpp"
    "src/api/kai_api_model_instance.cpp"
    "src/api/kai_api_component.cpp"
)
source_group("src\\api" FILES ${src__api})

set(src__components
    "src/components/component.cpp"
    "src/components/component.h"
    "src/components/component_info.cpp"
    "src/components/component_info.h"
    "src/components/kdataset.cpp"
    "src/components/kdataset.h"
    "src/components/kexpression.cpp"
    "src/components/kexpression.h"
    "src/components/klayer.cpp"
    "src/components/klayer.h"
    "src/components/kmodel.cpp"
    "src/components/kmodel.h"
    "src/components/knetwork.cpp"
    "src/components/knetwork.h"
    "src/components/koptimizer.cpp"
    "src/components/koptimizer.h"
    "src/components/kmodel_instance.h"
    "src/components/kmodel_instance.cpp"
)
source_group("src\\components" FILES ${src__components})

set(src__exec
    "src/exec/callback.cpp"
    "src/exec/callback.h"
    "src/exec/exec_context.cpp"
    "src/exec/exec_context.h"
    "src/exec/expr_graph.h"
    "src/exec/expr_graph.cpp"
)
source_group("src\\exec" FILES ${src__exec})

set(src__gpu_cuda
    "src/gpu_cuda/cuda_manager.cpp"
    "src/gpu_cuda/cuda_manager.h"
    "src/gpu_cuda/device_manager.h"
    "src/gpu_cuda/device_manager.cpp"
    "src/gpu_cuda/kai_kernels.cuh"
    "src/gpu_cuda/kai_kernels.cu"
)
source_group("src\\gpu_cuda" FILES ${src__gpu_cuda})

set(src__include
    "src/include/kai_api.h"
    "src/include/kai_api_shell.h"
    "src/include/kai_errors.h"
    "src/include/kai_types.h"
    "src/include/kai_value.hpp"
    "src/include/kai_callback.h"
)
source_group("src\\include" FILES ${src__include})



set(src__library
    "src/library/library.cpp"
    "src/library/library.h"
    "src/library/local_folder.cpp"
    "src/library/local_folder.h"
    "src/library/local_library.cpp"
    "src/library/local_library.h"
    "src/library/object.cpp"
    "src/library/object.h"
    "src/library/public_library.cpp"
    "src/library/public_library.h"
)
source_group("src\\library" FILES ${src__library})

set(src__math
    "src/math/karr_math.cpp"
    "src/math/karr_math.h"
    "src/math/karray.cpp"
    "src/math/karray.h"
    "src/math/khostmath.cpp"
    "src/math/khostmath.h"
    "src/math/kmath.cpp"
    "src/math/kmath.h"
    "src/math/kcudamath.cu"
    "src/math/kcudamath.h"
    "src/math/kmath.h"
    "src/math/kshape.cpp"
    "src/math/kshape.h"
)
source_group("src\\math" FILES ${src__math})

set(src__nightly
    #"src/nightly/path_generator.cpp"
    #"src/nightly/path_generator.h"
     "src/nightly/findfirst.c" 
     "src/nightly/findfirst.h"
     "src/nightly/nightly_utils.cpp" 
     "src/nightly/nightly_utils.h"
)
source_group("src\\nightly" FILES ${src__nightly})

set(src__session
    "src/session/kargs.cpp"
    "src/session/kargs.h"
    "src/session/kcommon.h"
    "src/session/kvalue.cpp"
    "src/session/session.cpp"
    "src/session/session.h"
    "src/session/callback_manager.h"
    "src/session/callback_manager.cpp"
)
source_group("src\\session" FILES ${src__session})

set(src__shell_temp
    "src/shell_temp/session_shell.cpp"
)
source_group("src\\shell_temp" FILES ${src__shell_temp})

set(src__utils
    "src/utils/kexception.cpp"
    "src/utils/kexception.h"
    "src/utils/klogger.cpp"
    "src/utils/klogger.h"
    "src/utils/kutil.cpp"
    "src/utils/kutil.h"
    "src/utils/kv_util.cpp"
    "src/utils/kv_util.h"
)
source_group("src\\utils" FILES ${src__utils})

set(src2020__apps
    "src2020/apps/autoencoder.cpp"
    "src2020/apps/autoencoder.h"
    "src2020/apps/encdec.cpp"
    "src2020/apps/encdec.h"
    "src2020/apps/gan.cpp"
    "src2020/apps/gan.h"
)
source_group("src2020\\apps" FILES ${src2020__apps})

set(src2020__core
    "src2020/core/array.cpp"
    "src2020/core/array.h"
    "src2020/core/common.h"
    "src2020/core/corpus.cpp"
    "src2020/core/corpus.h"
    "src2020/core/data_channel.cpp"
    "src2020/core/data_channel.h"
    "src2020/core/dataset.cpp"
    "src2020/core/dataset.h"
    "src2020/core/dim.cpp"
    "src2020/core/dim.h"
    "src2020/core/engine.cpp"
    "src2020/core/engine.h"
    "src2020/core/func_timer.cpp"
    "src2020/core/func_timer.h"
    "src2020/core/host_math.cpp"
    "src2020/core/host_math.h"
    "src2020/core/idx.cpp"
    "src2020/core/idx.h"
    "src2020/core/log.cpp"
    "src2020/core/log.h"
    "src2020/core/macro_pack.cpp"
    "src2020/core/macro_pack.h"
    "src2020/core/random.cpp"
    "src2020/core/random.h"
    "src2020/core/samples.cpp"
    "src2020/core/samples.h"
    "src2020/core/shape.cpp"
    "src2020/core/shape.h"
    "src2020/core/shell.cpp"
    "src2020/core/shell.h"
    "src2020/core/util.cpp"
    "src2020/core/util.h"
    "src2020/core/value.cpp"
    "src2020/core/value.h"
)
source_group("src2020\\core" FILES ${src2020__core})

set(src2020__cuda
    "src2020/cuda/cuda_conn.cuh"
    "src2020/cuda/cuda_kernels.h"
    "src2020/cuda/cuda_math.h"
    "src2020/cuda/cuda_note.h"
    "src2020/cuda/cuda_util.cuh"
    "src2020/cuda/cuda_conn.cu"
    "src2020/cuda/cuda_kernels.cu"
    "src2020/cuda/cuda_math.cu"
    "src2020/cuda/cuda_note.cu"
    "src2020/cuda/cuda_util.cu"
)
source_group("src2020\\cuda" FILES ${src2020__cuda})

set(src2020__cuda__cuda_datasets 
    "src2020/cuda/cuda_datasets/cuda_dataset.cu"
    "src2020/cuda/cuda_datasets/cuda_yolo.cu"
)
source_group("src2020\\cuda\\cuda_datasets" FILES ${src2020__cuda__cuda_datasets})
    

set(src2020__datasets
    "src2020/datasets/abalone.cpp"
    "src2020/datasets/abalone.h"
    "src2020/datasets/automata.cpp"
    "src2020/datasets/automata.h"
    "src2020/datasets/bert.cpp"
    "src2020/datasets/bert.h"
    "src2020/datasets/cifar10_auto.cpp"
    "src2020/datasets/cifar10_auto.h"
    "src2020/datasets/coco.cpp"
    "src2020/datasets/coco.h"
    "src2020/datasets/dummy.cpp"
    "src2020/datasets/dummy.h"
    "src2020/datasets/flower.cpp"
    "src2020/datasets/flower.h"
    "src2020/datasets/gan_cifar10.cpp"
    "src2020/datasets/gan_cifar10.h"
    "src2020/datasets/mnist_auto.cpp"
    "src2020/datasets/mnist_auto.h"
    "src2020/datasets/mnist_encdec.cpp"
    "src2020/datasets/mnist_encdec.h"
    "src2020/datasets/mnist_gan.cpp"
    "src2020/datasets/mnist_gan.h"
    "src2020/datasets/office31.cpp"
    "src2020/datasets/office31.h"
    "src2020/datasets/pulsar.cpp"
    "src2020/datasets/pulsar.h"
    "src2020/datasets/steel.cpp"
    "src2020/datasets/steel.h"
    "src2020/datasets/test_dataset.cpp"
    "src2020/datasets/test_dataset.h"
    "src2020/datasets/urbansound.cpp"
    "src2020/datasets/urbansound.h"
    "src2020/datasets/videoshot.cpp"
    "src2020/datasets/videoshot.h"
    "src2020/datasets/word2vec.cpp"
    "src2020/datasets/word2vec.h"
    "src2020/datasets/word_seq.cpp"
    "src2020/datasets/word_seq.h"
    "src2020/datasets/yolo.cpp"
    "src2020/datasets/yolo.h"
)
source_group("src2020\\datasets" FILES ${src2020__datasets})



set(src2020__datasets__corpus
    "src2020/datasets/corpus/korean_news.cpp"
    "src2020/datasets/corpus/korean_news.h"
    "src2020/datasets/corpus/ptb_corpus.cpp"
    "src2020/datasets/corpus/ptb_corpus.h"
)
source_group("src2020\\datasets\\corpus" FILES ${src2020__datasets__corpus})

set(src2020__int_plugin
    "src2020/int_plugin/internal_plugin.cpp"
    "src2020/int_plugin/internal_plugin.h"
    "src2020/int_plugin/layer.cuh"
    "src2020/int_plugin/optimizer.cuh"
    "src2020/int_plugin/layer.cu"
    "src2020/int_plugin/optimizer.cu"
)
source_group("src2020\\int_plugin" FILES ${src2020__int_plugin})

set(src2020__int_plugin__layers
    "src2020/int_plugin/layers/add_layer.cuh"
    "src2020/int_plugin/layers/attention_layer.cuh"
    "src2020/int_plugin/layers/batch_norm_layer.cuh"
    "src2020/int_plugin/layers/conv_layer.cuh"
    "src2020/int_plugin/layers/custom_layer.cuh"
    "src2020/int_plugin/layers/deconv_layer.cuh"
    "src2020/int_plugin/layers/dropout_layer.cuh"
    "src2020/int_plugin/layers/embed_layer.cuh"
    "src2020/int_plugin/layers/embedding_layer.cuh"
    "src2020/int_plugin/layers/expand_layer.cuh"
    "src2020/int_plugin/layers/extract_layer.cuh"
    "src2020/int_plugin/layers/full_layer.cuh"
    "src2020/int_plugin/layers/merge_layer.cuh"
    "src2020/int_plugin/layers/noise_layer.cuh"
    "src2020/int_plugin/layers/parallel_layer.cuh"
    "src2020/int_plugin/layers/pass_layer.cuh"
    "src2020/int_plugin/layers/pool_layer.cuh"
    "src2020/int_plugin/layers/reshape_layer.cuh"
    "src2020/int_plugin/layers/rnn_layer.cuh"
    "src2020/int_plugin/layers/serial_layer.cuh"
    "src2020/int_plugin/layers/add_layer.cu"
    "src2020/int_plugin/layers/attention_layer.cu"
    "src2020/int_plugin/layers/batch_norm_layer.cu"
    "src2020/int_plugin/layers/conv_layer.cu"
    "src2020/int_plugin/layers/custom_layer.cu"
    "src2020/int_plugin/layers/deconv_layer.cu"
    "src2020/int_plugin/layers/dropout_layer.cu"
    "src2020/int_plugin/layers/embed_layer.cu"
    "src2020/int_plugin/layers/embedding_layer.cu"
    "src2020/int_plugin/layers/expand_layer.cu"
    "src2020/int_plugin/layers/extract_layer.cu"
    "src2020/int_plugin/layers/full_layer.cu"
    "src2020/int_plugin/layers/merge_layer.cu"
    "src2020/int_plugin/layers/noise_layer.cu"
    "src2020/int_plugin/layers/parallel_layer.cu"
    "src2020/int_plugin/layers/pass_layer.cu"
    "src2020/int_plugin/layers/pool_layer.cu"
    "src2020/int_plugin/layers/reshape_layer.cu"
    "src2020/int_plugin/layers/rnn_layer.cu"
    "src2020/int_plugin/layers/serial_layer.cu"
)
source_group("src2020\\int_plugin\\layers" FILES ${src2020__int_plugin__layers})

set(src2020__int_plugin__optimizers
    "src2020/int_plugin/optimizers/adam_optimizer.cuh"
    "src2020/int_plugin/optimizers/new_optimizer.cuh"
    "src2020/int_plugin/optimizers/sgd_optimizer.cuh"
    "src2020/int_plugin/optimizers/adam_optimizer.cu"
    "src2020/int_plugin/optimizers/new_optimizer.cu"
    "src2020/int_plugin/optimizers/sgd_optimizer.cu"
)

source_group("src2020\\int_plugin\\optimizers" FILES ${src2020__int_plugin__optimizers})

set(src2020__server
    "src2020/server/client_service.cpp"
    "src2020/server/client_service.h"
    "src2020/server/connect.cpp"
    "src2020/server/connect.h"
)
source_group("src2020\\server" FILES ${src2020__server})

set(src2020__utils
    "src2020/utils/news_reformer.cpp"
    "src2020/utils/news_reformer.h"
    "src2020/utils/test.cpp"
    "src2020/utils/test.h"
)
source_group("src2020\\utils" FILES ${src2020__utils})


set(ALL_FILES
    ${src__api}
    ${src__components}
    ${src__exec}
    ${src__gpu_cuda}
    ${src__include}
    ${src__library}
    ${src__math}
    ${src__nightly}
    ${src__session}
    ${src__shell_temp}
    ${src__utils}
    ${src2020__apps}
    ${src2020__core}
    ${src2020__cuda}
    ${src2020__cuda__cuda_datasets}
    ${src2020__datasets}
    ${src2020__datasets__corpus}
    ${src2020__int_plugin}
    ${src2020__int_plugin__layers}
    ${src2020__int_plugin__optimizers}
    ${src2020__server}
    ${src2020__utils}
)

################################################################################
# Target
################################################################################
add_library(${PROJECT_NAME} SHARED ${ALL_FILES})

set(ROOT_NAMESPACE kai_cuda)

################################################################################
# required package
################################################################################
if(MSVC)
    #opencv should be in install folder!
    find_package(CUDAToolkit) # for library directory information 
else(UNIX)
    #find_package(CUDAToolkit) #CMAKE >= 18
    find_package(OpenCV REQUIRED)
endif()

################################################################################
# Include directories
################################################################################
if(MSVC)
    list(APPEND INCLUDE_DIRECTORY
        "${CMAKE_CURRENT_SOURCE_DIR}/../../install/opencv/build/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../install/MySQL/Connector C++ 8.0/include;"
    )
else(UNIX)
    list(APPEND INCLUDE_DIRECTORY
        ${OpenCV_INCLUDE_DIRS}    
    )
endif()
list(APPEND INCLUDE_DIRECTORY  "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}" )
target_include_directories(${PROJECT_NAME} PUBLIC ${INCLUDE_DIRECTORY})
################################################################################
# Compile definitions
################################################################################
if(MSVC)
   #target_compile_definitions(${PROJECT_NAME} PRIVATE
   list(APPEND COMPILE_DEF 
    "$<$<CONFIG:Debug>:"
        "_DEBUG" 
    ">"
    "WIN64;" "KAI_EXPORTS;" "KAI2021_WINDOWS;" "_CONSOLE;" "_WINDLL" "_MBCS" "_CRT_SECURE_NO_WARNINGS"
    )
endif()

if(NO_RANDOM)
    list(APPEND COMPILE_DEF NORANDOM)
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
        $<$<COMPILE_LANGUAGE:CUDA>:
            $<$<CONFIG:Debug>:
                -g 
                
                -ftz=false
                -prec-div=true
                -prec-sqrt=true
            >
            -fmad=false
            -cudart static            
       >
   )
elseif(UNIX)
    list(APPEND COMPILE_OPT
    $<$<COMPILE_LANGUAGE:CXX>:
        -ffp-contract=off
     >
    $<$<COMPILE_LANGUAGE:CUDA>:
        $<$<CONFIG:Debug>:
        -g 
       
        -ftz=false
        -prec-div=true
        -prec-sqrt=true
        >
        -fmad=false
        -Xcompiler "-ffp-contract=off"
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