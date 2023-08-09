#ifndef TRITON_LANG_DIALECT_TRITONGPU_IR_DIALECT_H_
#define TRITON_LANG_DIALECT_TRITONGPU_IR_DIALECT_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"

// TritonLang depends on Triton
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonLang/IR/Dialect.h.inc"

#define GET_OP_CLASSES
#include "triton/Dialect/TritonLang/IR/Ops.h.inc"

#endif
