#ifndef TRITON_LANG_OPS
#define TRITON_LANG_OPS

include "mlir/Dialect/Arith/IR/ArithBase.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/TritonLang/IR/TritonLangDialect.td"

//
// Op Base
//
class TL_Op<string mnemonic, list<Trait> traits = []> :
    Op<TritonLang_Dialect, mnemonic, traits> {
}

//
// Make Range Op
//
// TODO: should have ConstantLike as Trait
def TL_MakeRangeOp : TL_Op<"make_range", []> {
    let summary = "make range";

    let description = [{
        Returns an 1D int32 tensor.

        Values span from $start to $end (exclusive), with step = 1
    }];

    let arguments = (ins I32:$start, I32:$end);

    let results = (outs TT_IntTensor:$result);

    let assemblyFormat = "$start `,` $end attr-dict `:` type($result)";
}

def TL_CvtShapeOp : TL_Op<"cvt_shape", []> {
    let summary = "convert shape";

    let description = [{
        Returns an 1D int32 tensor.

        Values span from $start to $end (exclusive), with step = 1
    }];

    let arguments = (ins TT_Tensor:$src);

    let results = (outs TT_Tensor:$result);

    let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

#endif // TRITO_LANG_OPS
