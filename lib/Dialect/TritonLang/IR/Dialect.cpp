#include "triton/Dialect/TritonLang/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton_lang;

void TritonLangDialect::initialize() {

  addOperations<
#define GET_OP_LIST
#include "triton/Dialect/TritonLang/IR/Ops.cpp.inc"
      >();
}

#include "triton/Dialect/TritonLang/IR/Dialect.cpp.inc"
