set(LLVM_TARGET_DEFINITIONS TritonLangDialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=tl)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=tl)
#mlir_tablegen(Ops.h.inc -gen-op-decls)
#mlir_tablegen(Ops.cpp.inc -gen-op-defs)


set(LLVM_TARGET_DEFINITIONS TritonLangOps.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
#mlir_tablegen(OpsEnums.h.inc -gen-enum-decls)
#mlir_tablegen(OpsEnums.cpp.inc -gen-enum-defs)

add_public_tablegen_target(TritonLangTableGen)
