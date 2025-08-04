set(TFLM_ROOT ${CMAKE_CURRENT_LIST_DIR}/../../../tflite-micro)

message(STATUS "TFLM_ROOT = ${TFLM_ROOT}")

idf_component_register(
    SRCS
        ${TFLM_ROOT}/tensorflow/lite/micro/micro_interpreter.cc
        ${TFLM_ROOT}/tensorflow/lite/micro/micro_allocator.cc
#        ${TFLM_ROOT}/tensorflow/lite/micro/simple_memory_allocator.cc
#        ${TFLM_ROOT}/tensorflow/lite/micro/micro_mutable_op_resolver.cc
        ${TFLM_ROOT}/tensorflow/lite/micro/system_setup.cc
        ${TFLM_ROOT}/tensorflow/lite/kernels/internal/common.cc
        ${TFLM_ROOT}/tensorflow/lite/kernels/kernel_util.cc
#        ${TFLM_ROOT}/tensorflow/lite/core/api/error_reporter.cc
#        ${TFLM_ROOT}/tensorflow/lite/core/api/op_resolver.cc
#        ${TFLM_ROOT}/tensorflow/lite/schema/schema_utils.cc

    INCLUDE_DIRS 
        ${TFLM_ROOT}
        ${TFLM_ROOT}/tensorflow
        ${TFLM_ROOT}/tensorflow/lite
        ${TFLM_ROOT}/tensorflow/lite/micro
        ${TFLM_ROOT}/tensorflow/lite/kernels
        ${TFLM_ROOT}/tensorflow/lite/kernels/internal
        ${TFLM_ROOT}/third_party/flatbuffers/include
        ${TFLM_ROOT}/third_party/gemmlowp
        ${TFLM_ROOT}/tensorflow/lite/core/api
        ${TFLM_ROOT}/third_party

    REQUIRES 
        esp_timer
        esp_common
        newlib

    PRIV_REQUIRES
        heap
)

# Only define what’s needed
target_compile_definitions(${COMPONENT_LIB} PRIVATE
    TF_LITE_STATIC_MEMORY
)

target_compile_options(${COMPONENT_LIB} PRIVATE
    -fno-rtti
    -fno-exceptions
    -fno-threadsafe-statics
    -fmessage-length=0
    -O2
)
