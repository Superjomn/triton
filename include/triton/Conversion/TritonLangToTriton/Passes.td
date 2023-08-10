#ifndef TRITON_LANG_CONVERSION_PASSES
#define TRITON_LANG_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTritonLangToTriton : Pass<"convert-triton-lang-to-triton", "mlir::ModuleOp"> {
    let summary = "Convert TritonLang to Triton";
    let description = [{

    }];
    let constructor = "mlir::triton_lang::createConvertTritonLangToTritonPass()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::math::MathDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::triton_lang::TritonLangDialect",
                             "mlir::triton::TritonDialect" ];

   let options = [
        Option<"kernel_name", "kernel_name", "std::string", "kernel_name", "Name of the kernel to specialize">,
        ListOption<"specs", "specs", "int32_t", "Specialization for constexprs">
   ];
}

#endif
