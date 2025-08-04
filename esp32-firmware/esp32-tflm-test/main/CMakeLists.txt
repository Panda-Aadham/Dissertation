idf_component_register(
    SRCS 
        "main.cpp"
        "model_data.cpp"
    INCLUDE_DIRS "."
    REQUIRES 
        tflite-lib
        esp_timer
        esp_common
)

# Enable C++17
set_property(TARGET ${COMPONENT_LIB} PROPERTY CXX_STANDARD 17)
target_compile_options(${COMPONENT_LIB} PRIVATE -std=c++17)
