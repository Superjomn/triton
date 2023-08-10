add_mlir_conversion_library(TritonLangToTriton
    TritonLangToTritonPass.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/triton/Conversion/TritonLangToTriton
    ${PROJECT_BINARY_DIR}/include/triton/Conversion/TritonLangToTriton

    DEPENDS
    TritonLangConversionPassIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransforms
    TritonLangIR
    TritonIR
)
