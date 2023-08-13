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
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
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
  mutable std::map<std::string, ConstexprSpec> specs;

public:
  TritonFuncOpPattern(MLIRContext *ctx,
                      const std::map<std::string, ConstexprSpec> &specs)
      : OpConversionPattern<triton::FuncOp>(ctx), specs(specs) {}

  LogicalResult
  matchAndRewrite(triton::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    llvm::outs() << "TritonFuncOpPattern\n";
    llvm::outs() << "func.name: " << op.getName() << "\n";
    auto it = specs.find(op.getName().str());
    if (it == specs.end())
      return failure();
    llvm::outs() << "continue to replace constant: " << op.getName() << "\n";

    auto const_values = it->second.specs;

    BitVector argOffsets;
    for (auto &spec : const_values) {
      auto arg = op.getArgument(spec.first);
      auto cst = rewriter.create<arith::ConstantIntOp>(op.getLoc(), spec.second,
                                                       arg.getType());
      rewriter.replaceAllUsesWith(arg, cst);
      llvm::outs() << "replace arg: " << spec.first
                   << " with constant: " << spec.second << "\n";
      argOffsets.push_back(spec.first);
    }
    op.eraseArguments(argOffsets);

    specs.erase(op.getName().str());

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

    llvm::outs() << "TritonMakeRangePattern\n";
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

void populateTritonPatterns(RewritePatternSet &patterns,
                            const std::map<std::string, ConstexprSpec> &specs) {
  MLIRContext *context = patterns.getContext();
  TritonLangTypeConverter typeConverter(context);

  patterns.insert<TritonMakeRangePattern>(context, specs);
}

void populateFuncOpPattern(RewritePatternSet &patterns,
                           const std::map<std::string, ConstexprSpec> &specs) {
  MLIRContext *context = patterns.getContext();
  TritonLangTypeConverter typeConverter(context);

  patterns.insert<TritonFuncOpPattern>(context, specs);
}

class ConvertTritonLangToTriton
    : public ConvertTritonLangToTritonBase<ConvertTritonLangToTriton> {
public:
  ConvertTritonLangToTriton() {
    // for debug
    kernel_name = "kernel0";
    specs = {9, 16, 10, 16, 11, 16};
  }

  void runOnOperation() override {
    auto &context = getContext();
    auto mod = getOperation();

    TritonLangTypeConverter typeConverter(&context);
    TritonConversionTarget target(context, typeConverter);

    std::map<std::string, ConstexprSpec> specs_map;
    for (int i = 0; i < this->specs.size(); i += 2)
      specs_map[kernel_name].specs[this->specs[i]] = this->specs[i + 1];

    materializeConstExprs();

    /*
        RewritePatternSet func_patterns(&context);
        populateFuncOpPattern(func_patterns, specs_map);
        if (failed(applyPartialConversion(mod, target,
       std::move(func_patterns)))) return signalPassFailure();

        llvm::outs() << "mod:\n" << mod << "\n";


        RewritePatternSet patterns(&context);
        populateTritonPatterns(patterns, specs_map);


        //if (failed(applyPatternsAndFoldGreedily(mod, std::move(patterns))))
        if (failed(applyPartialConversion(mod, target, std::move(patterns))))
          return signalPassFailure();
        */
  }

  void materializeConstExprs() {
    auto &context = getContext();
    auto mod = getOperation();

    mod.walk([&](triton::FuncOp op) {
      if (op.getName().str() != this->kernel_name)
        return;
      OpBuilder builder(op);
      Block &entryBlock = op.getBody().front();
      builder.setInsertionPointToStart(&entryBlock);

      std::map<int, int> specs;
      SmallVector<int, 8> arg_offsets;
      for (int i = 0; i < this->specs.size(); i += 2) {
        specs[this->specs[i]] = this->specs[i + 1];
        arg_offsets.push_back(this->specs[i]);
      }

      BitVector argOffsets;
      for (auto &item : specs) {
        auto arg = op.getArgument(item.first);
        auto cst = builder.create<arith::ConstantIntOp>(
            op.getLoc(), item.second, arg.getType());

        arg.replaceAllUsesWith(cst);
        argOffsets.push_back(item.first);
      }

      // update function type
      SmallVector<Type, 8> newInputTypes;
      for (unsigned i = 0, e = op.getNumArguments(); i != e; ++i) {
        if (!specs.count(i)) {
          newInputTypes.push_back(op.getArgument(i).getType());
        }
      }
      auto newType = FunctionType::get(op.getContext(), newInputTypes,
                                       op.getResultTypes());
      llvm::outs() << "newType: " << newType << "\n";
      op.setType(newType);

      // update entry block's type
      std::reverse(arg_offsets.begin(), arg_offsets.end());
      for (int offset : arg_offsets) {
        op.getBlocks().front().eraseArgument(offset);
      }
    });

    mlir::OpPrintingFlags printFlags;
    // printFlags.enableDebugInfo().assumeVerified();
    mod.print(llvm::outs(), printFlags);
  }
};

namespace mlir {
namespace triton_lang {
std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonLangToTritonPass() {
  return std::make_unique<ConvertTritonLangToTriton>();
}
} // namespace triton_lang
} // namespace mlir
