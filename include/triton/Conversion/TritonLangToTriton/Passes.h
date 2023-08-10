#ifndef TRITON_LANG_CONVERSION_PASSES_H
#define TRITON_LANG_CONVERSION_PASSES_H

#include "triton/Conversion/TritonLangToTriton/TritonLangToTritonPass.h"

namespace mlir {
namespace triton_lang {

#define GEN_PASS_REGISTRATION
#include "triton/Conversion/TritonLangToTriton/Passes.h.inc"

} // namespace triton_lang
} // namespace mlir

#endif
