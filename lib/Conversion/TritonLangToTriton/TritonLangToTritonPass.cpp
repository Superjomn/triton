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

using ShapeT = llvm::ArrayRef<int64_t>;

template <typename T> void printVec(llvm::ArrayRef<T> vec) {
  llvm::outs() << "[";
  for (auto item : vec) {
    llvm::outs() << item << ", ";
  }
  llvm::outs() << "]\n";
}

bool isDynamicShape(ShapeT shape) {
  for (auto dim : shape) {
    if (dim == ShapedType::kDynamic)
      return true;
  }
  return false;
}

bool isDynamicShape(RankedTensorType tensorTy) {
  auto shape = tensorTy.getShape();
  return isDynamicShape(shape);
}

// Deducing the shape and data type
class TypeInference {
  std::deque<triton_lang::CvtShapeOp> queue;

  using CastOp = triton_lang::CvtShapeOp;

public:
  void run(ModuleOp op) {
    // materialize all the tl.make_range
    op.walk([&](triton::FuncOp func) {
      processAllMakeRange(func);
      processAllSplat(func);
      processAllBroadcast(func);
    });

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

    op.walk([&](triton::FuncOp func) { cleanUpCastOp(func); });
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
        // assert(false && "expand_dims should be backward");
      } else if (auto dot = dyn_cast<triton::DotOp>(op)) {
        if (output == dot.getOperand(0))
          processDot(dot, 0, shape);
        else if (output == dot.getOperand(1))
          processDot(dot, 1, shape);
        else
          ; // C is not needed
      } else if (isElementwiseOp(op)) {
        processElementwise(op, shape);
      } else if (dyn_cast<triton::LoadOp>(op) ||
                 dyn_cast<triton::StoreOp>(op)) {
        processLoadStore(op, shape);
      } else {
        // llvm_unreachable("unknown op");
        llvm::errs() << "forward unknown op: " << *op << "\n";
      }
    } else {
      processMultiUsersForward(output, cast);
    }

    return true;
  }

  bool propagateBackward(CastOp cast) {
    Value input = cast.getSrc();
    Value output = cast.getResult();
    unsigned numUsers = getNumUsers(input);

    if (numUsers == 0) {
      assert(false && "unreachable");
    } else if (numUsers == 1) {
      Type outTy = output.getType();
      Operation *op = input.getDefiningOp();
      auto shape = outTy.cast<RankedTensorType>().getShape();

      if (op == nullptr) {
        assert(outTy.isa<RankedTensorType>());
        processBlockArgBackward(input.cast<BlockArgument>(), cast);
      } else if (auto prevCast = dyn_cast<CastOp>(op)) {
        foldAdjacentCasts(prevCast, cast);
      } else if (auto expandDims = dyn_cast<triton::ExpandDimsOp>(op)) {
        processExpandDimsBackward(expandDims, shape);
      } else if (isElementwiseOp(op)) {
        processElementwise(op, shape);
      } else {
        llvm::errs() << "backward unknown op: " << *op << "\n";
        // llvm_unreachable("unknown op");
      }
    }

    return false;
  }

  void processMultiUsersForward(Value castRes, CastOp cast) {
    Value input = cast.getOperand();
    Location loc = cast.getLoc();
    OpBuilder builder(cast);

    while (!castRes.use_empty()) {
      auto newCast =
          markForward(builder.create<CastOp>(loc, castRes.getType(), input));
      castRes.use_begin()->set(newCast.getResult());
      pushQueue(newCast);
    }

    cast.erase();
  }

  // TODO[Superjomn]: support block arg
  void processBlockArgBackward(BlockArgument arg, CastOp cast) {}

  // A trick
  // TODO[Superjomn]: find nicer way
  void cleanUpCastOp(triton::FuncOp op) {
    op.walk([&](CastOp cast) {
      // T -> T
      auto inTy = cast.getOperand().getType();
      auto outTy = cast.getResult().getType();
      if (inTy == outTy) {
        cast.getResult().replaceAllUsesWith(cast.getOperand());
        cast.erase();
        return;
      }

      if (cast.getResult().getUsers().empty()) {
        cast.erase();
      }
    });
  }

  // propagate from the output to the input
  void processExpandDimsBackward(triton::ExpandDimsOp op,
                                 ArrayRef<int64_t> shape,
                                 llvm::StringRef annotation = "") {
    auto type = op.getResult().getType().cast<RankedTensorType>();
    auto inShape =
        op.getOperand().getType().cast<RankedTensorType>().getShape();
    assert(!isDynamicShape(shape) && "Unvalid propagate shape");

    llvm::SmallVector<int64_t> newInputShape(shape.begin(), shape.end());
    int axis = op.getAxis();
    newInputShape.erase(newInputShape.begin() + axis);

    amendTypeWithCasts(op, {newInputShape}, {shape}, "expand_dims");
  }

  void processLoadStore(Operation *op, ShapeT shape) {
    llvm::SmallVector<ShapeT> inShapes(op->getNumOperands(), shape);
    llvm::SmallVector<ShapeT> outShapes(op->getNumResults(), shape);
    amendTypeWithCasts(op, inShapes, outShapes, "load/store");
  }

  void amendOperandTypeWithCast(Operation *op, int index, ShapeT shape,
                                llvm::StringRef annotation = "") {
    Location loc = op->getLoc();
    OpBuilder builder(op->getContext());
    builder.setInsertionPoint(op);

    auto operand = op->getOperand(index);
    Type operandTy = operand.getType();
    if (triton::isTensorOrTensorPointerType(operandTy)) {
      auto newType = replaceShape(operandTy.cast<RankedTensorType>(), shape);

      auto newCast = builder.create<CastOp>(loc, newType, operand);
      if (!annotation.empty())
        newCast->setAttr("annotation", builder.getStringAttr(annotation));
      newCast = markBackward(newCast);
      op->setOperand(index, newCast.getResult());
      pushQueue(newCast);
    }
  }

  void amendResultTypeWithCast(Operation *op, int index, ShapeT shape,
                               llvm::StringRef annotation = "") {
    Location loc = op->getLoc();
    OpBuilder builder(op->getContext());
    builder.setInsertionPointAfter(op);

    auto result = op->getResult(index);
    Type resultTy = result.getType();
    if (triton::isTensorOrTensorPointerType(resultTy)) {
      auto newType = replaceShape(resultTy.cast<RankedTensorType>(), shape);

      auto newCast = builder.create<CastOp>(loc, newType, result);
      if (!annotation.empty())
        newCast->setAttr("annotation", builder.getStringAttr(annotation));
      newCast = markForward(newCast);
      result.setType(newType);
      result.replaceAllUsesExcept(newCast.getResult(), newCast.getOperation());
      pushQueue(newCast);
    }
  }

  void amendTypeWithCasts(Operation *op, llvm::ArrayRef<ShapeT> inShapes,
                          llvm::ArrayRef<ShapeT> outShapes,
                          llvm::StringRef annotation = "") {
    Location loc = op->getLoc();
    OpBuilder builder(op->getContext());

    if (!inShapes.empty()) {
      assert(op->getNumOperands() == inShapes.size());
      builder.setInsertionPoint(op);
      for (unsigned i = 0; i < op->getNumOperands(); ++i) {
        amendOperandTypeWithCast(op, i, inShapes[i], annotation);
      }
    }

    if (!outShapes.empty()) {
      assert(op->getNumResults() == outShapes.size());
      builder.setInsertionPointAfter(op);
      for (unsigned i = 0; i < op->getNumResults(); ++i) {
        amendResultTypeWithCast(op, i, outShapes[i], annotation);
      }
    }
  }

  static RankedTensorType replaceShape(RankedTensorType type, ShapeT shape) {
    return RankedTensorType::get(shape, type.getElementType());
  }

  // cast0 -> cast1 => cast1
  void foldAdjacentCasts(CastOp cast0, CastOp cast1) {
    assert(cast0.getResult() == cast1.getOperand());
    assert(isForward(cast0) && isBackward(cast1));

    // input => cast0 => cast1 => output
    Value input = cast0.getSrc();
    Value output = cast1.getResult();

    if (input.getType() == output.getType()) {
      output.replaceAllUsesWith(input);
    } else {
      OpBuilder builder(cast1);
      auto newCast =
          builder.create<CastOp>(cast1.getLoc(), output.getType(), input);
      newCast->setAttr("annotation", builder.getStringAttr("fold"));
      output.replaceAllUsesWith(newCast.getResult());
    }

    // erase cast1 first since cast0 still consumes it
    if (cast1.getResult().getUsers().empty()) {
      eraseCastOpFromQueue({cast1});
      cast1.erase();
    }
    if (cast0.getResult().getUsers().empty()) {
      eraseCastOpFromQueue({cast0});
      cast0.erase();
    }
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

  static triton_lang::CvtShapeOp markForward(triton_lang::CvtShapeOp op) {
    OpBuilder builder(op);
    op->setAttr("forward", builder.getBoolAttr(true));
    return op;
  }

  static triton_lang::CvtShapeOp markBackward(triton_lang::CvtShapeOp op) {
    OpBuilder builder(op);
    op->setAttr("backward", builder.getBoolAttr(true));
    return op;
  }

  static bool isForward(triton_lang::CvtShapeOp op) {
    if (!op->hasAttr("forward"))
      return false;
    return op->getAttr("forward").cast<BoolAttr>().getValue();
  }

  static bool isBackward(triton_lang::CvtShapeOp op) {
    if (!op->hasAttr("backward"))
      return false;
    return op->getAttr("backward").cast<BoolAttr>().getValue();
  }

  void processAllSplat(triton::FuncOp func) {
    func.walk(
        [&](triton_lang::SplatOp op) { assert(succeeded(processSplat(op))); });
  }

  void processAllMakeRange(triton::FuncOp func) {
    func.walk([&](triton_lang::MakeRangeOp op) {
      assert(succeeded(processMarkRange(op)));
    });
  }

  void processAllBroadcast(triton::FuncOp func) {
    func.walk([&](triton_lang::BroadcastOp op) {
      assert(succeeded(processBroadcast(op)));
    });
  }

  void processElementwise(Operation *op, ShapeT shape) {
    llvm::SmallVector<ShapeT> inShapes(op->getNumOperands(), shape);
    llvm::SmallVector<ShapeT> outShapes(op->getNumResults(), shape);
    amendTypeWithCasts(op, inShapes, outShapes, "elementwise");
  }

  void processDot(triton::DotOp op, int operandOffset, ShapeT shape) {
    assert((operandOffset == 0 || operandOffset == 1) &&
           "Only operand A and B are needed in shape inference");
    assert(!isDynamicShape(shape));

    amendOperandTypeWithCast(op, operandOffset, shape);
    auto ATy = op.getA().getType().cast<RankedTensorType>();
    auto BTy = op.getB().getType().cast<RankedTensorType>();
    auto CTy = op.getC().getType().cast<RankedTensorType>();
    auto DTy = op.getResult().getType().cast<RankedTensorType>();

    if (!isDynamicShape(ATy) && !isDynamicShape(BTy)) {
      auto AShape = ATy.getShape();
      auto BShape = BTy.getShape();
      SmallVector<int64_t> retShape({AShape[0], BShape[1]});
      if (isDynamicShape(DTy)) {
        amendResultTypeWithCast(op, 0, retShape, "dot");
      }

      if (op.getC() && isDynamicShape(CTy)) {
        amendOperandTypeWithCast(op, 2, retShape, "dot C");
      }
    }
  }

  LogicalResult processMarkRange(triton_lang::MakeRangeOp op) {
    auto tensorTy = op.getType().cast<RankedTensorType>();
    auto shape = tensorTy.getShape();

    OpBuilder builder(op);

    // check if it is dynamic shape
    bool isDynamic = tensorTy.isDynamicDim(0);
    if (!isDynamic)
      return failure();

    int64_t start = op.getStart().getDefiningOp<arith::ConstantIntOp>().value();
    int64_t end = op.getEnd().getDefiningOp<arith::ConstantIntOp>().value();

    amendTypeWithCasts(op, {}, {shape}, "make_range");

    auto newTensorTy =
        RankedTensorType::get({end - start}, tensorTy.getElementType());

    auto newOp = builder.create<triton::MakeRangeOp>(op.getLoc(), newTensorTy,
                                                     start, end);

    op.replaceAllUsesWith(newOp.getResult());
    op.erase();
    return success();
  }

  LogicalResult processSplat(triton_lang::SplatOp op) {
    auto retShape = op.getType().cast<RankedTensorType>().getShape();
    SmallVector<int64_t> resShape(retShape.begin(), retShape.end());

    // symbolDims should all be constant
    auto symbolDims = op.getSymbolDims();
    int symbolOffest{0};

    for (int64_t &dim : resShape) {
      if (dim == ShapedType::kDynamic) {
        dim = symbolDims[symbolOffest++]
                  .getDefiningOp<arith::ConstantIntOp>()
                  .value();
      }
    }

    amendTypeWithCasts(op, {}, {resShape}, "splat");

    OpBuilder builder(op);
    auto newSplat = builder.create<triton::SplatOp>(op.getLoc(), op.getType(),
                                                    op.getValue());
    op.getResult().replaceAllUsesWith(newSplat.getResult());
    op.erase();

    return success();
  }

  LogicalResult processBroadcast(triton_lang::BroadcastOp op) {
    auto retShape = op.getType().cast<RankedTensorType>().getShape();
    auto inShape = op.getSrc().getType().cast<RankedTensorType>().getShape();
    SmallVector<int64_t> resShape(retShape.begin(), retShape.end());
    SmallVector<int64_t> newInShape(inShape.begin(), inShape.end());

    // symbolDims should all be constant
    auto symbolDims = op.getSymbolDims();
    int symbolOffest{0};

    for (int i = 0; i < resShape.size(); ++i) {
      if (resShape[i] == ShapedType::kDynamic) {
        resShape[i] = symbolDims[symbolOffest++]
                          .getDefiningOp<arith::ConstantIntOp>()
                          .value();

        if (inShape[i] == ShapedType::kDynamic) {
          newInShape[i] = resShape[i];
        }
      }
    }

    SmallVector<ShapeT> inShapes{newInShape};
    inShapes.resize(1 + op.getSymbolDims().size(), {});

    amendTypeWithCasts(op, inShapes, {resShape}, "broadcast");

    OpBuilder builder(op);
    auto newBroadcast = builder.create<triton::BroadcastOp>(
        op.getLoc(), op.getType(), op.getSrc());
    op.getResult().replaceAllUsesWith(newBroadcast.getResult());
    op.erase();

    return success();
  }

  // Borrowed from PlanCTA.cpp
  // TODO[Superjomn]: Make it a common utility function
  static bool isElementwiseOp(Operation *op) {
    if (llvm::isa<arith::AddFOp, arith::AddIOp, arith::AndIOp,
                  arith::CeilDivSIOp, arith::CeilDivUIOp, arith::DivFOp,
                  arith::DivSIOp, arith::DivUIOp, arith::ExtFOp, arith::ExtSIOp,
                  arith::ExtUIOp, arith::FloorDivSIOp, arith::FPToSIOp,
                  arith::FPToUIOp, arith::MaxFOp, arith::MaxSIOp,
                  arith::MaxUIOp, arith::MinFOp, arith::MinSIOp, arith::MinUIOp,
                  arith::MulFOp, arith::MulIOp, arith::NegFOp, arith::OrIOp,
                  arith::RemFOp, arith::RemSIOp, arith::RemUIOp, arith::ShLIOp,
                  arith::ShRSIOp, arith::ShRUIOp, arith::SIToFPOp,
                  arith::SubFOp, arith::SubIOp, arith::TruncFOp,
                  arith::TruncIOp, arith::UIToFPOp, arith::XOrIOp>(op))
      return true;
    if (llvm::isa<math::AbsFOp, math::AbsIOp, math::AtanOp, math::Atan2Op,
                  math::CeilOp, math::CopySignOp, math::CosOp, math::SinOp,
                  math::CountLeadingZerosOp, math::CountTrailingZerosOp,
                  math::CtPopOp, math::ErfOp, math::ExpOp, math::Exp2Op,
                  math::ExpM1Op, math::FloorOp, math::FmaOp, math::LogOp,
                  math::Log10Op, math::Log1pOp, math::Log2Op, math::PowFOp,
                  math::RsqrtOp, math::SqrtOp, math::TanhOp>(op))
      return true;
    if (llvm::isa<triton::IntToPtrOp, triton::PtrToIntOp, triton::BitcastOp,
                  triton::FpToFpOp, triton::AddPtrOp,
                  triton::PureExternElementwiseOp>(op))
      return true;
    return false;
  }
};

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

    materializeConstExprs();

    TypeInference typeInference;
    typeInference.run(mod);
  }

  void materializeConstExprs() {
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
