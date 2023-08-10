add_mlir_dialect_library(TritonLangIR
  Dialect.cpp
  Ops.cpp

  DEPENDS
  TritonLangTableGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRMathDialect
  MLIRSCFDialect
)
