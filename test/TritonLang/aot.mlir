module {
  tt.func public @kernel0(%arg0: !tt.ptr<f32, 1>, %arg1: !tt.ptr<f32, 1>, %arg2: !tt.ptr<f32, 1>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %0 = tl.make_range %c0_i32, %arg9 : tensor<?xi32>
    %c0_i32_0 = arith.constant 0 : i32
    %1 = tl.make_range %c0_i32_0, %arg9 : tensor<?xi32>
    %c0_i32_1 = arith.constant 0 : i32
    %2 = tl.make_range %c0_i32_1, %arg9 : tensor<?xi32>
    %3 = tt.expand_dims %0 {axis = 1 : i32} : (tensor<?xi32>) -> tensor<?x1xi32>
    %4 = tl.splat %arg5 %arg9 : (i32, i32) -> tensor<?x1xi32>
    %5 = arith.muli %3, %4 : tensor<?x1xi32>
    %6 = tl.splat %arg1 %arg9 : (!tt.ptr<f32, 1>, i32) -> tensor<?x1x!tt.ptr<f32, 1>>
    %7 = tt.addptr %6, %5 : tensor<?x1x!tt.ptr<f32, 1>>, tensor<?x1xi32>
    %8 = tt.expand_dims %2 {axis = 0 : i32} : (tensor<?xi32>) -> tensor<1x?xi32>
    %9 = tl.splat %arg6 %arg9 : (i32, i32) -> tensor<1x?xi32>
    %10 = arith.muli %8, %9 : tensor<1x?xi32>
    %11 = tl.broadcast %7 %arg9, %arg9 : tensor<?x1x!tt.ptr<f32, 1>> -> tensor<?x?x!tt.ptr<f32, 1>>
    %12 = tl.broadcast %10 %arg9, %arg9 : tensor<1x?xi32> -> tensor<?x?xi32>
    %13 = tt.addptr %11, %12 : tensor<?x?x!tt.ptr<f32, 1>>, tensor<?x?xi32>
    %14 = tt.load %13 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<?x?xf32>
    %15 = tt.expand_dims %2 {axis = 1 : i32} : (tensor<?xi32>) -> tensor<?x1xi32>
    %16 = tl.splat %arg7 %arg9 : (i32, i32) -> tensor<?x1xi32>
    %17 = arith.muli %15, %16 : tensor<?x1xi32>
    %18 = tl.splat %arg2 %arg9 : (!tt.ptr<f32, 1>, i32) -> tensor<?x1x!tt.ptr<f32, 1>>
    %19 = tt.addptr %18, %17 : tensor<?x1x!tt.ptr<f32, 1>>, tensor<?x1xi32>
    %20 = tt.expand_dims %1 {axis = 0 : i32} : (tensor<?xi32>) -> tensor<1x?xi32>
    %21 = tl.splat %arg8 %arg9 : (i32, i32) -> tensor<1x?xi32>
    %22 = arith.muli %20, %21 : tensor<1x?xi32>
    %23 = tl.broadcast %19 %arg9, %arg9 : tensor<?x1x!tt.ptr<f32, 1>> -> tensor<?x?x!tt.ptr<f32, 1>>
    %24 = tl.broadcast %22 %arg9, %arg9 : tensor<1x?xi32> -> tensor<?x?xi32>
    %25 = tt.addptr %23, %24 : tensor<?x?x!tt.ptr<f32, 1>>, tensor<?x?xi32>
    %26 = tt.load %25 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<?x?xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %27 = tl.splat %cst %arg9, %arg9 : (f32, i32, i32) -> tensor<?x?xf32>
    %28 = tt.dot %14, %26, %27 {allowTF32 = true} : tensor<?x?xf32> * tensor<?x?xf32> -> tensor<?x?xf32>
    //%29 = tt.call @"mul__fp32Sint32[constexpr[1]]_int32[constexpr[1]]S_fp32Sint32[constexpr[1]]_int32[constexpr[1]]S__"(%28, %28) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    %29 = arith.mulf %28, %28 : tensor<?x?xf32>
    %30 = tt.expand_dims %0 {axis = 1 : i32} : (tensor<?xi32>) -> tensor<?x1xi32>
    %31 = tl.splat %arg3 %arg9 : (i32, i32) -> tensor<?x1xi32>
    %32 = arith.muli %30, %31 : tensor<?x1xi32>
    %33 = tl.splat %arg0 %arg9 : (!tt.ptr<f32, 1>, i32) -> tensor<?x1x!tt.ptr<f32, 1>>
    %34 = tt.addptr %33, %32 : tensor<?x1x!tt.ptr<f32, 1>>, tensor<?x1xi32>
    %35 = tt.expand_dims %1 {axis = 0 : i32} : (tensor<?xi32>) -> tensor<1x?xi32>
    %36 = tl.splat %arg4 %arg9 : (i32, i32) -> tensor<1x?xi32>
    %37 = arith.muli %35, %36 : tensor<1x?xi32>
    %38 = tl.broadcast %34 %arg9, %arg9 : tensor<?x1x!tt.ptr<f32, 1>> -> tensor<?x?x!tt.ptr<f32, 1>>
    %39 = tl.broadcast %37 %arg9, %arg9 : tensor<1x?xi32> -> tensor<?x?xi32>
    %40 = tt.addptr %38, %39 : tensor<?x?x!tt.ptr<f32, 1>>, tensor<?x?xi32>
    tt.store %40, %29 {cache = 1 : i32, evict = 1 : i32} : tensor<?x?xf32>
    tt.return
  }
  // tt.func private @"mul__fp32Sint32[constexpr[1]]_int32[constexpr[1]]S_fp32Sint32[constexpr[1]]_int32[constexpr[1]]S__"(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> attributes {noinline = false} {
    //%0 = arith.mulf %arg0, %arg1 : tensor<?x?xf32>
    //tt.return %0 : tensor<?x?xf32>
  //}
}
