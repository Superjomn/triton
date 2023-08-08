module {
// Test the basic tt.any addresses, it could work with existing Triton address related ops
tt.func @aot_with_any_buffer(%buf: !tt.ptr<!tt.any>) {
  %x = tt.get_program_id x : i32
  %xx = tt.broadcast %x : (i32) -> tensor<256xi32>
  %pt = tt.broadcast %buf : (!tt.ptr<!tt.any>) -> tensor<256x!tt.ptr<!tt.any>>
  %27 = tt.addptr %pt, %xx : tensor<256x!tt.ptr<!tt.any>>, tensor<256xi32>

  tt.return
}

tt.func @aot_with_constexpr_as_int(%buf: !tt.ptr<!tt.any>, %BLOCK: !tt.constexpr) {
      %pt = tt.broadcast %buf : (!tt.ptr<!tt.any>) -> tensor<256x!tt.ptr<!tt.any>>

      // !tt.constexpr acts like an "Any" type, and there are some ops to convert it to real types
      %block = tt.cvt_dtype %BLOCK : !tt.constexpr -> i64

      %blocks = tt.broadcast %block : (i64) -> tensor<256xi64>
      %27 = tt.addptr %pt, %blocks : tensor<256x!tt.ptr<!tt.any>>, tensor<256xi64>

      tt.return
  }

  tt.func @aot_with_constexpr_as_function(%buf: !tt.ptr<!tt.any>, %some_fn: !tt.constexpr) {
      %pt = tt.broadcast %buf : (!tt.ptr<!tt.any>) -> tensor<256x!tt.ptr<!tt.any>>

      // the return type could only be deduced from python AST
      %ret = "tt.func_call" (%some_fn, %pt) {} : (!tt.constexpr, tensor<256x!tt.ptr<!tt.any>>) -> tensor<256x!tt.ptr<!tt.any>>

      tt.return
  }

  tt.func @aot_compare_constexpr_with_string(%SOME_STR : !tt.constexpr) {
      %res = tt.compare_str_with_constexpr "hello", %SOME_STR : i1

      tt.return
  }

  // basic example for matmul
  // Suppose we have the python kernel as below
  // def matmul(A, B, C, m:int, n:int, k:int, ACTIVATION:tl.constexpr):
  //     ...
  //     if ACTIVATION == "relu": ...
  //
  // The ACTIVATION is a constexpr holding an string acutally.
  tt.func @faked_tanh(%x: !tt.any) -> !tt.any {
    tt.return %x : !tt.any
  }

  tt.func @aot_matmul(%A:!tt.ptr<!tt.any>, %B:!tt.ptr<!tt.any>, %C:!tt.ptr<!tt.any>, %m:i32, %n:i32, %k:i32, %ACTIVATION:!tt.constexpr) {

    // ...
    %cond = tt.compare_str_with_constexpr "relu", %ACTIVATION : i1

    // we can deduce the constant tensor's type from python AST, at least we can tell int from float
    %acc_s = arith.constant 0.000000e+00 : f32
    %acc = tt.broadcast %acc_s : (f32) -> tensor<256xf32>

    // deal with the following python code:
    // if ACTIVATION == "relu":
    //   acc = relu(acc)
    %acc_up = scf.if %cond -> (tensor<256x!tt.any>) {
      %0 = tt.cvt_dtype %acc : tensor<256xf32> -> !tt.any
      %v = tt.call @faked_tanh(%0) : (!tt.any) -> !tt.any
      %v0 = tt.cvt_dtype %v : !tt.any -> tensor<256x!tt.any>
      scf.yield %v0 : tensor<256x!tt.any>
    } else {
      %1 = tt.cvt_dtype %acc : tensor<256xf32> -> tensor<256x!tt.any>
      scf.yield %1 : tensor<256x!tt.any>
    }

    tt.return
  }
}

module {
  tt.func @add_kernel__Pfp32_Pfp32_Pfp32_i32_i32_i32__(%arg0: !tt.ptr<!tt.any>, %arg1: !tt.ptr<!tt.any>, %arg2: !tt.ptr<!tt.any>, %arg3: i32, %arg4: i32, %arg5: i32) {
    %0 = tt.get_program_id x : i32
    %c256_i32 = arith.constant 256 : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %3 = tt.broadcast %1 : (i32) -> tensor<256xi32>
    %4 = arith.addi %3, %2 : tensor<256xi32>
    %5 = tt.broadcast %arg3 : (i32) -> tensor<256xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<256xi32>
    %7 = tt.broadcast %arg0 : (!tt.ptr<!tt.any>) -> tensor<256x!tt.ptr<!tt.any>>
    %8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<!tt.any>>, tensor<256xi32>
    %9 = tt.broadcast %arg1 : (!tt.ptr<!tt.any>) -> tensor<256x!tt.ptr<!tt.any>>
    %10 = tt.addptr %9, %4 : tensor<256x!tt.ptr<!tt.any>>, tensor<256xi32>
    %cst = arith.constant 0.000000e+00 : f32
    %cst0 = tt.cvt_dtype %cst : f32 -> !tt.any
    %11 = tt.broadcast %cst0 : (!tt.any) -> tensor<256x!tt.any>
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %12 = arith.index_cast %c0_i32 : i32 to index
    %13 = arith.index_cast %arg4 : i32 to index
    %14 = arith.index_cast %c32_i32 : i32 to index
    %15:3 = scf.for %arg6 = %12 to %13 step %14 iter_args(%arg7 = %11, %arg8 = %8, %arg9 = %10) -> (tensor<256x!tt.any>, tensor<256x!tt.ptr<!tt.any>>, tensor<256x!tt.ptr<!tt.any>>) {
      %cst_0 = arith.constant 0.000000e+00 : f32
      %cst_00 = tt.cvt_dtype %cst_0 : f32 -> !tt.any
      %18 = tt.broadcast %cst_00 : (!tt.any) -> tensor<256x!tt.any>
      %19 = tt.load %arg8, %6, %18 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256x!tt.any>
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_10 = tt.cvt_dtype %cst_1 : f32 -> !tt.any
      %20 = tt.broadcast %cst_10 : (!tt.any) -> tensor<256x!tt.any>
      %21 = tt.load %arg9, %6, %20 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256x!tt.any>
      %22 = tt.binary %19, %21, "add" : tensor<256x!tt.any>, tensor<256x!tt.any> -> tensor<256x!tt.any>
      //%22 = arith.addf %19, %21 : tensor<256x!tt.any>
      //%23 = arith.addf %arg7, %22 : tensor<256x!tt.any>
      %23 = tt.binary %arg7, %22, "add" : tensor<256x!tt.any>, tensor<256x!tt.any> -> tensor<256x!tt.any>

      %24 = tt.broadcast %arg5 : (i32) -> tensor<256xi32>
      %25 = tt.addptr %arg8, %24 : tensor<256x!tt.ptr<!tt.any>>, tensor<256xi32>
      %26 = tt.broadcast %arg5 : (i32) -> tensor<256xi32>
      %27 = tt.addptr %arg9, %26 : tensor<256x!tt.ptr<!tt.any>>, tensor<256xi32>
      scf.yield %23, %25, %27 : tensor<256x!tt.any>, tensor<256x!tt.ptr<!tt.any>>, tensor<256x!tt.ptr<!tt.any>>
    }
    %16 = tt.broadcast %arg2 : (!tt.ptr<!tt.any>) -> tensor<256x!tt.ptr<!tt.any>>
    %17 = tt.addptr %16, %4 : tensor<256x!tt.ptr<!tt.any>>, tensor<256xi32>
    tt.store %17, %15#0, %6 : tensor<256x!tt.any>
    tt.return
  }
}

module {
  tt.func @add_kernel__Pfp32_Pfp32_Pfp32_i32_i32_i32__(%arg0: !tt.ptr<!tt.any>, %arg1: !tt.ptr<!tt.any>, %arg2: !tt.ptr<!tt.any>, %arg3: i32, %arg4: i32, %arg5: i32) {
    %0 = tt.get_program_id x : i32
    %c256_i32 = arith.constant 256 : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %3 = tt.broadcast %1 : (i32) -> tensor<256xi32>
    %4 = arith.addi %3, %2 : tensor<256xi32>
    %5 = tt.broadcast %arg3 : (i32) -> tensor<256xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<256xi32>
    %7 = tt.broadcast %arg0 : (!tt.ptr<!tt.any>) -> tensor<256x!tt.ptr<!tt.any>>
    %8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<!tt.any>>, tensor<256xi32>
    %9 = tt.broadcast %arg1 : (!tt.ptr<!tt.any>) -> tensor<256x!tt.ptr<!tt.any>>
    %10 = tt.addptr %9, %4 : tensor<256x!tt.ptr<!tt.any>>, tensor<256xi32>
    %cst = arith.constant 0.000000e+00 : f32
    %cst0 = tt.cvt_dtype %cst : f32 -> !tt.any
    %11 = tt.broadcast %cst0 : (!tt.any) -> tensor<256x!tt.any>
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %12 = arith.index_cast %c0_i32 : i32 to index
    %13 = arith.index_cast %arg4 : i32 to index
    %14 = arith.index_cast %c32_i32 : i32 to index
    %15:3 = scf.for %arg6 = %12 to %13 step %14 iter_args(%arg7 = %11, %arg8 = %8, %arg9 = %10) -> (tensor<256x!tt.any>, tensor<256x!tt.ptr<!tt.any>>, tensor<256x!tt.ptr<!tt.any>>) {
      %cst_0 = arith.constant 0.000000e+00 : f32
      %cst_00 = tt.cvt_dtype %cst_0 : f32 -> !tt.any
      %18 = tt.broadcast %cst_00 : (!tt.any) -> tensor<256x!tt.any>
      %19 = tt.load %arg8, %6, %18 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256x!tt.any>
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_10 = tt.cvt_dtype %cst_1 : f32 -> !tt.any
      %20 = tt.broadcast %cst_10 : (!tt.any) -> tensor<256x!tt.any>
      %21 = tt.load %arg9, %6, %20 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256x!tt.any>

      // for reusing arith.addf, we convert all arguments from tt.any to f32, and then convert the result back to tt.any
      %f19 = tt.cvt_dtype %19 : tensor<256x!tt.any> -> tensor<256xf32>
      %f21 = tt.cvt_dtype %21 : tensor<256x!tt.any> -> tensor<256xf32>
      %22 = arith.addf %f19, %f21 : tensor<256xf32>
      %a22 = tt.cvt_dtype %22 : tensor<256xf32> -> tensor<256x!tt.any>

      %arg7_0 = tt.cvt_dtype %arg7 : tensor<256x!tt.any> -> tensor<256xf32>
      %23 = arith.addf %arg7_0, %22 : tensor<256xf32>
      %a23 = tt.cvt_dtype %23 : tensor<256xf32> -> tensor<256x!tt.any>

      %24 = tt.broadcast %arg5 : (i32) -> tensor<256xi32>
      %25 = tt.addptr %arg8, %24 : tensor<256x!tt.ptr<!tt.any>>, tensor<256xi32>
      %26 = tt.broadcast %arg5 : (i32) -> tensor<256xi32>
      %27 = tt.addptr %arg9, %26 : tensor<256x!tt.ptr<!tt.any>>, tensor<256xi32>
      scf.yield %a23, %25, %27 : tensor<256x!tt.any>, tensor<256x!tt.ptr<!tt.any>>, tensor<256x!tt.ptr<!tt.any>>
    }
    %16 = tt.broadcast %arg2 : (!tt.ptr<!tt.any>) -> tensor<256x!tt.ptr<!tt.any>>
    %17 = tt.addptr %16, %4 : tensor<256x!tt.ptr<!tt.any>>, tensor<256xi32>
    tt.store %17, %15#0, %6 : tensor<256x!tt.any>
    tt.return
  }
}
