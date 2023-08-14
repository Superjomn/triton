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
#include <queue>

using namespace mlir;
using namespace mlir::triton_lang;

#define GEN_PASS_CLASSES
#include "triton/Conversion/TritonLangToTriton/Passes.h.inc"

template <typename T> void printVec(llvm::ArrayRef<T> vec) {
  llvm::outs() << "[";
  for (auto item : vec) {
    llvm::outs() << item << ", ";
  }
  llvm::outs() << "]\n";
}

class TritonLangTypeConverter : public TypeConverter {
  using TypeConverter::convertType;

public:
  TritonLangTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    // If the origValue still has live user(s), use this to
    // convert origValue to newValue
    addSourceMaterialization([&](OpBuilder &builder,
                                 RankedTensorType tensorType, ValueRange inputs,
                                 Location loc) -> std::optional<Value> {
      llvm::outs() << "TritonLangTypeConverter to convert back to "
                   << tensorType << "\n";
      return builder.create<tensor::CastOp>(loc, tensorType, inputs[0]);
    });

    addTargetMaterialization([&](OpBuilder &builder, Type type,
                                 ValueRange inputs,
                                 Location loc) -> std::optional<Value> {
      llvm::outs() << "TritonLangTypeConverter to convert to " << type << "\n";
      return builder.create<tensor::CastOp>(loc, type, inputs[0]);
    });
  }
};

class TritonConversionTarget : public ConversionTarget {

public:
  explicit TritonConversionTarget(MLIRContext &ctx,
                                  TritonLangTypeConverter &typeConverter)
      : ConversionTarget(ctx) {
    addLegalDialect<arith::ArithDialect, scf::SCFDialect,
                    triton::TritonDialect>();
    // addIllegalDialect<triton_lang::TritonLangDialect>();
  }
};

struct TritonMakeRangePattern
    : public OpConversionPattern<triton_lang::MakeRangeOp> {
  using OpConversionPattern<triton_lang::MakeRangeOp>::OpConversionPattern;

  TritonMakeRangePattern(MLIRContext *ctx)
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
    rewriter.replaceOpWithNewOp<triton_lang::CvtShapeOp>(op, retType,
                                                         newOp.getResult());
    // rewriter.replaceOpWithNewOp<triton::MakeRangeOp>(op, newTensorTy, start,
    // end);
    /*
    auto newOp = rewriter.create<triton::MakeRangeOp>(op.getLoc(), newTensorTy,
                                                      start, end);
    auto cast = rewriter.create<tensor::CastOp>(op.getLoc(), retType,
                                                newOp.getResult());

    llvm::outs() << "replace " << op << " with " << cast << "\n";

    rewriter.replaceOp(op, {newOp.getResult()});
    */

    return success();
  }
};

bool isDynamicShape(RankedTensorType tensorTy) {
  auto shape = tensorTy.getShape();
  for (auto dim : shape) {
    if (tensorTy.isDynamicDim(dim))
      return true;
  }
  return false;
}

class ExpandDimsPattern : public OpConversionPattern<triton::ExpandDimsOp> {
  using OpConversionPattern<triton::ExpandDimsOp>::OpConversionPattern;

public:
  ExpandDimsPattern(MLIRContext *ctx)
      : OpConversionPattern<triton::ExpandDimsOp>(ctx) {}

  LogicalResult
  matchAndRewrite(triton::ExpandDimsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tensorTy = op.getOperand().getType().cast<RankedTensorType>();
    auto resTy = op.getResult().getType().cast<RankedTensorType>();
    llvm::outs() << "ExpandDimsPattern\n";

    // TODO[Superjomn]: Consider the case where axis is dyanmic
    if (isDynamicShape(tensorTy)) // cannot deduce
      return failure();
    if (!isDynamicShape(resTy)) // no need to deduce
      return failure();

    llvm::outs() << "materialize ExpandDimsPattern\n";

    auto operandShape = tensorTy.getShape();
    auto retShape = resTy.getShape();
    llvm::SmallVector<int64_t> resShape(retShape.begin(), retShape.end());
    for (int dim = 0; dim < tensorTy.getRank(); ++dim) {
      if (tensorTy.isDynamicDim(dim))
        resShape[dim] = operandShape[dim];
    }

    auto newTensorTy =
        RankedTensorType::get(resShape, tensorTy.getElementType());
    auto newOp = rewriter.create<triton::ExpandDimsOp>(op.getLoc(), newTensorTy,
                                                       op.getOperand());
    rewriter.replaceOp(op, {newOp.getResult()});

    return success();
  }
};

// Deducing the shape and data type
class TypeInference {
  std::deque<triton_lang::CvtShapeOp> queue;

  using CastOp = triton_lang::CvtShapeOp;
  using ShapeT = llvm::ArrayRef<int64_t>;

public:
  void run(ModuleOp op) {
    // materialize all the tl.make_range
    op.walk([&](triton::FuncOp func) { processAllMakeRange(func); });

    // graph traversal
    while (!queue.empty()) {
      auto op = queue.front();
      queue.pop_front();
      if (isForward(op)) {
        propagateForward(op);
      } else if (isBackward(op)) {
        propagateBackward(op);
      } else {
        llvm_unreachable("unknown op");
      }
    }
  }

private:
  void pushQueue(CastOp op) {
    assert(isForward(op) || isBackward(op));
    queue.push_back(op);
  }

  bool propagate(CastOp cast) {
    return isBackward(cast) ? propagateBackward(cast) : propagateForward(cast);
  }

  unsigned getNumUsers(Value value) {
    return std::distance(value.getUsers().begin(), value.getUsers().end());
  }

  bool propagateForward(CastOp cast) {
    Value input = cast.getOperand();
    Value output = cast.getResult();
    unsigned numUsers = getNumUsers(output);

    if (numUsers == 0) {
      cast.erase();
      return true;
    }

    auto shape = input.getType().cast<RankedTensorType>().getShape();

    if (numUsers == 1) {
      Operation *op = *output.getUsers().begin();
      if (auto nextCast = dyn_cast<CastOp>(op)) {
        foldAdjacentCasts(cast, nextCast);
      } else if (auto expandDims = dyn_cast<triton::ExpandDimsOp>(op)) {
        processExpandDimsOp(expandDims, shape);
      } else {
        llvm_unreachable("unknown op");
      }
    }

    return true;
  }

  bool propagateBackward(CastOp cast) {
    Value input = cast.getSrc();
    Value output = cast.getResult();
    unsigned numUsers = getNumUsers(input);

    if (numUsers == 0) {
      cast.erase();
      return true;
    }

    if (numUsers == 1) {
      Type outTy = output.getType();
      Operation *op = input.getDefiningOp();
      if (op == nullptr) {
        assert(outTy.isa<RankedTensorType>());
        processBlockArgBackward(input.cast<BlockArgument>(), cast);
      } else if (auto prevCast = dyn_cast<CastOp>(op)) {
        foldAdjacentCasts(prevCast, cast);
      } else {
        llvm_unreachable("unknown op");
      }
    }

    return false;
  }

  void processBlockArgBackward(BlockArgument arg, CastOp cast) {}

  void processExpandDimsOp(triton::ExpandDimsOp op, ArrayRef<int64_t> shape) {
    auto type = op.getResult().getType().cast<RankedTensorType>();
    auto inShape =
        op.getOperand().getType().cast<RankedTensorType>().getShape();
    llvm::SmallVector<int64_t> newResShape(shape.begin(), shape.end());
    int axis = op.getAxis();
    newResShape.insert(newResShape.begin() + axis, 1);

    llvm::outs() << "inputShape: ";
    printVec<int64_t>(shape);

    llvm::outs() << "newResShape: ";
    printVec<int64_t>(newResShape);

    amendTypeWithCasts(op, {shape}, {newResShape});
  }

  void amendTypeWithCasts(Operation *op, llvm::ArrayRef<ShapeT> inShapes,
                          llvm::ArrayRef<ShapeT> outShapes) {
    assert(op->getNumOperands() == inShapes.size());
    assert(op->getNumResults() == outShapes.size());

    Location loc = op->getLoc();
    OpBuilder builder(op);

    builder.setInsertionPoint(op);
    for (unsigned i = 0; i < op->getNumOperands(); ++i) {
      auto operand = op->getOperand(i);
      if (triton::isTensorOrTensorPointerType(operand.getType())) {
        auto newType = replaceShape(operand.getType().cast<RankedTensorType>(),
                                    inShapes[i]);
        auto newCast = builder.create<CastOp>(loc, newType, operand);
        newCast = markBackward(newCast);
        op->setOperand(i, newCast.getResult());
        pushQueue(newCast);
      }
    }

    builder.setInsertionPointAfter(op);
    for (unsigned i = 0; i < op->getNumResults(); ++i) {
      auto result = op->getResult(i);
      if (triton::isTensorOrTensorPointerType(result.getType())) {
        auto newResTy = replaceShape(result.getType().cast<RankedTensorType>(),
                                     outShapes[i]);
        auto newCast = builder.create<CastOp>(loc, newResTy, result);
        newCast = markForward(newCast);

        result.setType(newResTy);
        result.replaceAllUsesExcept(newCast.getResult(),
                                    newCast.getOperation());

        pushQueue(newCast);
      }
    }
  }

  RankedTensorType replaceShape(RankedTensorType type, ShapeT shape) {
    return RankedTensorType::get(shape, type.getElementType());
  }

  // cast0 -> cast1 => cast1
  void foldAdjacentCasts(CastOp cast0, CastOp cast1) {
    assert(cast0.getResult() == cast1.getOperand());
    Value input = cast0.getSrc();
    Value output = cast1.getResult();

    if (input.getType() == output.getType()) {
      output.replaceAllUsesWith(input);
    } else {
      OpBuilder builder(cast1);
      auto newCast =
          builder.create<CastOp>(cast1.getLoc(), output.getType(), input);
      output.replaceAllUsesWith(newCast.getResult());
    }

    eraseCastOpFromQueue({cast0, cast1});
    cast0.erase();
    cast1.erase();
  }

  void eraseCastOpFromQueue(llvm::ArrayRef<CastOp> ops) {
    llvm::DenseSet<CastOp> toErase(ops.begin(), ops.end());
    for (int i = 0, size = queue.size(); i < size; ++i) {
      auto op = queue.front();
      queue.pop_front();
      if (!toErase.count(op))
        queue.push_back(op);
    }
  }

  triton_lang::CvtShapeOp createCvtShape(Type retType, Value input,
                                         Location loc) {
    OpBuilder builder(loc->getContext());
    builder.setInsertionPointAfterValue(input);
    return builder.create<triton_lang::CvtShapeOp>(loc, retType, input);
  }

  triton_lang::CvtShapeOp markForward(triton_lang::CvtShapeOp op) {
    OpBuilder builder(op);
    op->setAttr("forward", builder.getBoolAttr(true));
    return op;
  }

  triton_lang::CvtShapeOp markBackward(triton_lang::CvtShapeOp op) {
    OpBuilder builder(op);
    op->setAttr("backward", builder.getBoolAttr(true));
    return op;
  }

  bool isForward(triton_lang::CvtShapeOp op) {
    if (!op->hasAttr("forward"))
      return false;
    return op->getAttr("forward").cast<BoolAttr>().getValue();
  }

  bool isBackward(triton_lang::CvtShapeOp op) {
    if (!op->hasAttr("backward"))
      return false;
    return op->getAttr("backward").cast<BoolAttr>().getValue();
  }

  void processAllMakeRange(triton::FuncOp func) {
    func.walk([&](triton_lang::MakeRangeOp op) {
      assert(succeeded(materializeMarkRange(op)));
    });
  }

  LogicalResult materializeMarkRange(triton_lang::MakeRangeOp op) {
    auto tensorTy = op.getType().cast<RankedTensorType>();
    auto shape = tensorTy.getShape();

    OpBuilder builder(op);

    // check if it is dynamic shape
    bool isDynamic = tensorTy.isDynamicDim(0);
    if (!isDynamic)
      return failure();

    int64_t start = op.getStart().getDefiningOp<arith::ConstantIntOp>().value();
    int64_t end = op.getEnd().getDefiningOp<arith::ConstantIntOp>().value();

    auto newTensorTy =
        RankedTensorType::get({end - start}, tensorTy.getElementType());
    auto newOp = builder.create<triton::MakeRangeOp>(op.getLoc(), newTensorTy,
                                                     start, end);

    auto cvt = createCvtShape(tensorTy, newOp.getResult(), op.getLoc());
    cvt = markForward(cvt);
    pushQueue(cvt);

    op.replaceAllUsesWith(cvt.getResult());
    op.erase();
    return success();
  }
};

void populateTritonPatterns(RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  TritonLangTypeConverter typeConverter(context);

  patterns.insert<TritonMakeRangePattern, ExpandDimsPattern>(context);
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

    // TritonLangTypeConverter typeConverter(&context);
    // TritonConversionTarget target(context, typeConverter);
    // target.addDynamicallyLegalDialect<arith::ArithDialect>(
    //     [&](mlir::Operation *op) {
    //       return true;
    //       return typeConverter.isLegal(op);
    //     });
    materializeConstExprs();

    // RewritePatternSet patterns(&context);
    // populateTritonPatterns(patterns);

    // // if (failed(applyPatternsAndFoldGreedily(mod, std::move(patterns))))
    // if (failed(applyPartialConversion(mod, target, std::move(patterns))))
    //   return signalPassFailure();

    TypeInference typeInference;
    typeInference.run(mod);

    llvm::outs() << "after conversion:\n" << mod << "\n";
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
  }
};

namespace mlir {
namespace triton_lang {
std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonLangToTritonPass() {
  return std::make_unique<ConvertTritonLangToTriton>();
}
} // namespace triton_lang
} // namespace mlir
