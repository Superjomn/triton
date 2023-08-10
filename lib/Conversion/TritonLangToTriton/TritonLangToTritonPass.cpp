#include "triton/Conversion/TritonLangToTriton/TritonLangToTritonPass.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonLang/IR/Dialect.h"
#include "llvm/ADT/APSInt.h"
#include <functional>
#include <numeric>

using namespace mlir;
using namespace mlir::triton_lang;

#define GEN_PASS_CLASSES
#include "triton/Conversion/TritonLangToTriton/Passes.h.inc"

struct ConstexprSpec {
  std::map<int, int> specs; // arg-offset to value
};

class TritonLangTypeConverter : public LLVMTypeConverter {
  using TypeConverter::convertType;

public:
  TritonLangTypeConverter(MLIRContext *ctx,
                          const DataLayoutAnalysis *analysis = nullptr)
      : LLVMTypeConverter(ctx, analysis) {}
};

class TritonConversionTarget : public ConversionTarget {

public:
  explicit TritonConversionTarget(MLIRContext &ctx,
                                  TritonLangTypeConverter &typeConverter)
      : ConversionTarget(ctx) {
    addLegalDialect<arith::ArithDialect, scf::SCFDialect,
                    triton::TritonDialect>();
  }
};

// Replace the constexpr arguments with integer constants
class TritonFuncOpPattern : public OpConversionPattern<triton::FuncOp> {
  std::map<std::string, ConstexprSpec> specs;

public:
  TritonFuncOpPattern(MLIRContext *ctx,
                      const std::map<std::string, ConstexprSpec> &specs)
      : OpConversionPattern<triton::FuncOp>(ctx), specs(specs) {}

  LogicalResult
  matchAndRewrite(triton::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto it = specs.find(op.getName().str());
    if (it == specs.end())
      return failure();

    auto const_values = it->second.specs;

    BitVector argOffsets;
    for (auto &spec : const_values) {
      auto arg = op.getArgument(spec.first);
      auto cst = rewriter.create<arith::ConstantIntOp>(op.getLoc(), spec.second,
                                                       arg.getType());
      rewriter.replaceUsesOfBlockArgument(arg, cst);
      argOffsets.push_back(spec.first);
    }
    op.eraseArguments(argOffsets);

    return success();
  }
};

struct TritonMakeRangePattern
    : public OpConversionPattern<triton_lang::MakeRangeOp> {
  using OpConversionPattern<triton_lang::MakeRangeOp>::OpConversionPattern;

  TritonMakeRangePattern(MLIRContext *ctx,
                         const std::map<std::string, ConstexprSpec> &specs)
      : OpConversionPattern<triton_lang::MakeRangeOp>(ctx) {}

  LogicalResult
  matchAndRewrite(triton_lang::MakeRangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type retType = op.getType();

    auto tensorTy = retType.cast<RankedTensorType>();
    auto shape = tensorTy.getShape();

    // check if it is dynamic shape

    bool isDynamic = tensorTy.isDynamicDim(0);
    if (!isDynamic)
      return failure();
    int64_t start = op.getStart().getDefiningOp<arith::ConstantIntOp>().value();
    int64_t end = op.getEnd().getDefiningOp<arith::ConstantIntOp>().value();

    auto newTensorTy =
        RankedTensorType::get({end - start}, tensorTy.getElementType());
    auto newOp = rewriter.create<triton::MakeRangeOp>(op.getLoc(), newTensorTy,
                                                      start, end);
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, retType, newOp);

    return success();
  }
};

void populateTritonPatterns(RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  TritonLangTypeConverter typeConverter(context);
  std::map<std::string, ConstexprSpec> specs;
  patterns.insert<TritonFuncOpPattern, TritonMakeRangePattern>(context, specs);
}

class ConvertTritonLangToTriton
    : public ConvertTritonLangToTritonBase<ConvertTritonLangToTriton> {
public:
  ConvertTritonLangToTriton() {}

  void runOnOperation() override {

    auto &context = getContext();
    RewritePatternSet patterns(&context);
    populateTritonPatterns(patterns);

    auto mod = getOperation();

    TritonLangTypeConverter typeConverter(&context);
    TritonConversionTarget target(context, typeConverter);
    if (failed(applyPartialConversion(mod, target, std::move(patterns))))
      return signalPassFailure();
  }
};

namespace mlir {
namespace triton_lang {
std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonLangToTritonPass() {

  return std::make_unique<ConvertTritonLangToTriton>();
}
} // namespace triton_lang
} // namespace mlir
