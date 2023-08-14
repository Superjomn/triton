#ifndef TRITON_LANG_DIALECT
#define TRITON_LANG_DIALECT

include "mlir/IR/OpBase.td"

def TritonLang_Dialect : Dialect {
  let name = "tl";

  let cppNamespace = "::mlir::triton_lang";

  let summary = "The Triton Lang in MLIR";

  let description = [{
    Triton Lang Dialect.
  }];

  let dependentDialects = [
    "arith::ArithDialect",
    "math::MathDialect",
    "scf::SCFDialect",
    "cf::ControlFlowDialect",
    "triton::TritonDialect",
    "tensor::TensorDialect"
  ];

  let extraClassDeclaration = [{
  }];
}


#endif // TRITON_LANG_DIALECT
