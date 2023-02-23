#include "Utility.h"

namespace mlir {
namespace LLVM {

Value operator+(const Value &a, const Value &b) {
  auto &loc = *RewriteEnvGuard::kLoc;
  auto &rewriter = *RewriteEnvGuard::kPatternRewriter;
  return add(a, b);
}

Value operator+(const Value &a, int32_t x) {
  auto &loc = *RewriteEnvGuard::kLoc;
  auto &rewriter = *RewriteEnvGuard::kPatternRewriter;
  auto xval = i32_val(x);
  return a + xval;
}

Value operator-(const Value &a, const Value &b) {
  auto &loc = *RewriteEnvGuard::kLoc;
  auto &rewriter = *RewriteEnvGuard::kPatternRewriter;
  return sub(a, b);
}
Value operator-(const Value &a, int32_t x) {
  auto &loc = *RewriteEnvGuard::kLoc;
  auto &rewriter = *RewriteEnvGuard::kPatternRewriter;
  auto xval = i32_val(x);
  return a - xval;
}
Value operator*(const Value &a, const Value &b) {
  auto &loc = *RewriteEnvGuard::kLoc;
  auto &rewriter = *RewriteEnvGuard::kPatternRewriter;
  return mul(a, b);
}
Value operator/(const Value &a, const Value &b) {
  auto &loc = *RewriteEnvGuard::kLoc;
  auto &rewriter = *RewriteEnvGuard::kPatternRewriter;
  if (a.getType().isIntOrIndex()) {
    assert(b.getType().isIntOrIndex());
    return udiv(a, b);
  }
  return Value{};
}
Value operator%(const Value &a, const Value &b) {
  auto &loc = *RewriteEnvGuard::kLoc;
  auto &rewriter = *RewriteEnvGuard::kPatternRewriter;
  return urem(a, b);
}

} // namespace LLVM
} // namespace mlir
