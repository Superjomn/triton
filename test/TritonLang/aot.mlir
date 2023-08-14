module {
  tt.func public @kernel0(%arg0: !tt.ptr<f32, 1>, %arg1: !tt.ptr<f32, 1>, %arg2: !tt.ptr<f32, 1>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %0 = tl.make_range %c0_i32, %arg9 : tensor<?xi32>
    %3 = tt.expand_dims %0 {axis = 1 : i32} : (tensor<?xi32>) -> tensor<?x1xi32>
    //%c0_i32_0 = arith.constant 0 : i32
    //%1 = tl.make_range %c0_i32_0, %arg9 : tensor<?xi32>
    //%c0_i32_1 = arith.constant 0 : i32
    //%2 = tl.make_range %c0_i32_1, %arg9 : tensor<?xi32>
    // %4 = tt.splat %arg5 : (i32) -> tensor<?x1xi32>
    // %5 = arith.muli %3, %4 : tensor<?x1xi32>
    // %6 = tt.splat %arg1 : (!tt.ptr<f32, 1>) -> tensor<?x1x!tt.ptr<f32, 1>>
    // %7 = tt.addptr %6, %5 : tensor<?x1x!tt.ptr<f32, 1>>, tensor<?x1xi32>
    // %8 = tt.expand_dims %2 {axis = 0 : i32} : (tensor<?xi32>) -> tensor<1x?xi32>
    // %9 = tt.splat %arg6 : (i32) -> tensor<1x?xi32>
    // %10 = arith.muli %8, %9 : tensor<1x?xi32>
    // %11 = tt.broadcast %7 : (tensor<?x1x!tt.ptr<f32, 1>>) -> tensor<?x?x!tt.ptr<f32, 1>>
    // %12 = tt.broadcast %10 : (tensor<1x?xi32>) -> tensor<?x?xi32>
    // %13 = tt.addptr %11, %12 : tensor<?x?x!tt.ptr<f32, 1>>, tensor<?x?xi32>
    // %14 = tt.load %13 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<?x?xf32>
    // %15 = tt.expand_dims %2 {axis = 1 : i32} : (tensor<?xi32>) -> tensor<?x1xi32>
    // %16 = tt.splat %arg7 : (i32) -> tensor<?x1xi32>
    // %17 = arith.muli %15, %16 : tensor<?x1xi32>
    // %18 = tt.splat %arg2 : (!tt.ptr<f32, 1>) -> tensor<?x1x!tt.ptr<f32, 1>>
    // %19 = tt.addptr %18, %17 : tensor<?x1x!tt.ptr<f32, 1>>, tensor<?x1xi32>
    // %20 = tt.expand_dims %1 {axis = 0 : i32} : (tensor<?xi32>) -> tensor<1x?xi32>
    // %21 = tt.splat %arg8 : (i32) -> tensor<1x?xi32>
    // %22 = arith.muli %20, %21 : tensor<1x?xi32>
    // %23 = tt.broadcast %19 : (tensor<?x1x!tt.ptr<f32, 1>>) -> tensor<?x?x!tt.ptr<f32, 1>>
    // %24 = tt.broadcast %22 : (tensor<1x?xi32>) -> tensor<?x?xi32>
    // %25 = tt.addptr %23, %24 : tensor<?x?x!tt.ptr<f32, 1>>, tensor<?x?xi32>
    // %26 = tt.load %25 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<?x?xf32>
    // %cst = arith.constant 0.000000e+00 : f32
    // %cst_2 = tt.splat %cst : (f32) -> tensor<?x?xf32>
    // %27 = tt.dot %14, %26, %cst_2 {allowTF32 = true} : tensor<?x?xf32> * tensor<?x?xf32> -> tensor<?x?xf32>
    // %28 = arith.mulf %27, %27 : tensor<?x?xf32>
    // //%28 = tt.call @"mul__fp32S-1_-1S_fp32S-1_-1S__"(%27, %27) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    // %29 = tt.expand_dims %0 {axis = 1 : i32} : (tensor<?xi32>) -> tensor<?x1xi32>
    // %30 = tt.splat %arg3 : (i32) -> tensor<?x1xi32>
    // %31 = arith.muli %29, %30 : tensor<?x1xi32>
    // %32 = tt.splat %arg0 : (!tt.ptr<f32, 1>) -> tensor<?x1x!tt.ptr<f32, 1>>
    // %33 = tt.addptr %32, %31 : tensor<?x1x!tt.ptr<f32, 1>>, tensor<?x1xi32>
    // %34 = tt.expand_dims %1 {axis = 0 : i32} : (tensor<?xi32>) -> tensor<1x?xi32>
    // %35 = tt.splat %arg4 : (i32) -> tensor<1x?xi32>
    // %36 = arith.muli %34, %35 : tensor<1x?xi32>
    // %37 = tt.broadcast %33 : (tensor<?x1x!tt.ptr<f32, 1>>) -> tensor<?x?x!tt.ptr<f32, 1>>
    // %38 = tt.broadcast %36 : (tensor<1x?xi32>) -> tensor<?x?xi32>
    // %39 = tt.addptr %37, %38 : tensor<?x?x!tt.ptr<f32, 1>>, tensor<?x?xi32>
    // tt.store %39, %28 {cache = 1 : i32, evict = 1 : i32} : tensor<?x?xf32>
    tt.return
  }

  //tt.func private @"mul__fp32S-1_-1S_fp32S-1_-1S__"(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> attributes {noinline = false} {
    //%0 = arith.mulf %arg0, %arg1 : tensor<?x?xf32>
    //tt.return %0 : tensor<?x?xf32>
  //}
}
