import importlib.util
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import List

import triton
import triton.language as tl

desc = """
Triton compiler for C++ JIT.

This program compiles the kernel with name `kernel-name` in the file at the
provided `path` into self-contained C++ source-code that embeds the MLIR code.

signature is provided as a list of (optionally divisibility-hinted) types
or constexpr values, e.g.

`cppjit.py --kernel-name kernel --signature "*f32, i32, i32, i32" --out-name kernel /path/to/kernel.py`

will compile triton.JITFunction of name `kernel` inside the file `/path/to/kernel.py`.
Said kernel will be specialized such that argument 0 is f32 buffer, argument 1, 2, 3 are i32 scalars.
Note that, the `tl.constexpr` arguments should be specialized too.
"""

if __name__ == "__main__":

    # command-line arguments
    parser = ArgumentParser(description=desc)
    parser.add_argument("path", help="Path to Python source containing desired kernel in its scope. File will be executed.")
    parser.add_argument("--kernel-name", "-n", type=str, default="", help="Name of the kernel to compile", required=True)
    parser.add_argument("--out-name", "-on", type=str, default=None, help="Out name for the compiled kernel")
    parser.add_argument("--out-path", "-o", type=Path, default=None, help="Out filename")
    parser.add_argument("--signature", "-s", type=str, help="Signature of the kernel", required=True)
    args = parser.parse_args()

    out_name = args.out_name if args.out_name else args.kernel_name
    out_path = args.out_path if args.out_path else out_name

    # execute python sources and extract functions wrapped in JITFunction
    arg_path = Path(args.path)
    sys.path.insert(0, str(arg_path.parent))
    spec = importlib.util.spec_from_file_location(arg_path.stem, arg_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    kernel = getattr(mod, args.kernel_name)

    # validate and parse signature
    signature = list(map(lambda s: s.strip(" "), args.signature.split(",")))

    def constexpr(s):
        '''
        The format: type:c, e.g. i32:c
        '''
        s = s.strip(" ")
        if ':' in s:
            s = s.split(":")[0]
            if s.startswith("i"):
                s = f"int{s[1:]}"
                return tl.dtype(s)

    constexprs = {i: constexpr(s) for i, s in enumerate(signature)}
    constexprs = {k: v for k, v in constexprs.items() if v is not None}

    # treat constexpr as normal argument
    signature = {i: s.split(":")[0] for i, s in enumerate(signature)}
    print('signature', signature)
    print('constants', constexprs)

    # compile ast into cubin
    config = triton.compiler.instance_descriptor(divisible_by_16=set(), equal_to_1=set())

    # deferred parameters with num_wraps, num_stages, constexprs
    ccinfo = triton.compile(kernel, signature=signature, constants=constexprs, configs=[config], num_warps=2)
    print(ccinfo)
