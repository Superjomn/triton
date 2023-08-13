#ifndef TRITON_CONVERSION_TRITONLANGTOTRITON_TRITONLANGTOTRITONPASS_H
#define TRITON_CONVERSION_TRITONLANGTOTRITON_TRITONLANGTOTRITONPASS_H

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton_lang {

// Create the pass with numWarps passed from cl::opt.

std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonLangToTritonPass();

} // namespace triton_lang
} // namespace mlir

#endif
