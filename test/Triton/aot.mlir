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
