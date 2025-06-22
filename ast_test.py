import ast
import inspect
import cupy as cp
import numpy as np
import textwrap
from types import FunctionType
from typing import Any, Dict, Tuple, Union
from httomo_backends.cufft import CufftType, cufft_estimate_1d, cufft_estimate_2d
from tomobar.methodsDIR_CuPy import RecToolsDIRCuPy


Shape = Union[int, Tuple[int, ...]]


class FakeCuPyArray:
    @staticmethod
    def new_ast_node(
        line_number: int, shape: ast.expr | None, dtype: ast.expr | ast.Constant
    ) -> ast.AST:
        return (
            ast.parse(
                f"FakeCuPyArray({line_number}, {ast.unparse(shape)}, dtype={ast.unparse(dtype)})"
            )
            .body[0]
            .value
        )

    def __init__(self, line_number_of_creation: int, shape, dtype="float32"):
        self.line_number_of_creation = line_number_of_creation
        self.shape = shape
        self.dtype = dtype
        print(
            f"[CREATE] FakeCuPyArray(line_no={self.line_number_of_creation}, shape={self.shape}, dtype={self.dtype})"
        )

    def __del__(self):
        print(
            f"[DELETE] FakeCuPyArray(line_no={self.line_number_of_creation}, shape={self.shape}, dtype={self.dtype})"
        )

    def __repr__(self):
        return f"FakeCuPyArray(line_no={self.line_number_of_creation}, shape={self.shape}, dtype={self.dtype})"

    def _binary_op(self, other):
        if isinstance(other, FakeCuPyArray):
            line_no = other.line_number_of_creation
            result_shape = np.broadcast_shapes(self.shape, other.shape)
            result_dtype = np.result_type(self.dtype, other.dtype)
        else:
            line_no = self.line_number_of_creation
            result_shape = self.shape
            result_dtype = np.result_type(self.dtype, type(other))
        return FakeCuPyArray(line_no, result_shape, str(result_dtype))

    def __add__(self, other):
        return self._binary_op(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._binary_op(other)

    def __rsub__(self, other):
        return self._binary_op(other)

    def __mul__(self, other):
        return self._binary_op(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._binary_op(other)

    def __rtruediv__(self, other):
        return self._binary_op(other)

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)

        key = key + (slice(None),) * (len(self.shape) - len(key))

        new_shape = []
        for dim_size, idx in zip(self.shape, key):
            if isinstance(idx, int):
                continue
            elif isinstance(idx, slice):
                idx = slice(
                    int(idx.start) if idx.start is not None else None,
                    int(idx.stop) if idx.stop is not None else None,
                    int(idx.step) if idx.step is not None else None,
                )
                start, stop, step = idx.indices(dim_size)
                length = max(0, (stop - start + (step - 1)) // step)
                new_shape.append(length)
            else:
                raise TypeError(f"Unsupported index type: {type(idx)}")

        return FakeCuPyArray(self.line_number_of_creation, new_shape, self.dtype)

    def __setitem__(self, key, value):
        pass

    def astype(self, dtype):
        return FakeCuPyArray(self.line_number_of_creation, self.shape, dtype)


class CuPyMemoryPrintTransformer(ast.NodeTransformer):
    def __init__(self, output_function_name, cupy_aliases=set[str]):
        self.output_function_name = output_function_name
        self.cupy_aliases = cupy_aliases
        self.custom_kernels = []

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if isinstance(node.func, ast.Attribute) and isinstance(
            node.func.value, ast.Name
        ):
            lib_name = node.func.value.id
            func_name = node.func.attr

            if lib_name in self.cupy_aliases and func_name in [
                "zeros",
                "ones",
                "empty",
                "full",
            ]:
                shape = node.args[0] if node.args else None
                dtype = next(
                    (x.value for x in node.keywords if x.arg == "dtype"),
                    ast.Constant(value="float32"),
                )

                return FakeCuPyArray.new_ast_node(node.lineno, shape, dtype)

            if lib_name in self.cupy_aliases and func_name == "pad":
                return ast.Call(
                    func=ast.Name(id="call_with_temp_arg", ctx=ast.Load()),
                    args=[
                        node.args[0],
                        ast.Lambda(
                            args=ast.arguments(
                                posonlyargs=[],
                                args=[ast.arg(arg="x")],
                                kwonlyargs=[],
                                defaults=[],
                            ),
                            body=FakeCuPyArray.new_ast_node(
                                node.lineno,
                                ast.Call(
                                    func=ast.Name(id="tuple", ctx=ast.Load()),
                                    args=[
                                        ast.GeneratorExp(
                                            elt=ast.BinOp(
                                                left=ast.BinOp(
                                                    left=ast.Name(
                                                        id="s", ctx=ast.Load()
                                                    ),
                                                    op=ast.Add(),
                                                    right=ast.Subscript(
                                                        value=ast.Name(
                                                            id="p", ctx=ast.Load()
                                                        ),
                                                        slice=ast.Constant(value=0),
                                                        ctx=ast.Load(),
                                                    ),
                                                ),
                                                op=ast.Add(),
                                                right=ast.Subscript(
                                                    value=ast.Name(
                                                        id="p", ctx=ast.Load()
                                                    ),
                                                    slice=ast.Constant(value=1),
                                                    ctx=ast.Load(),
                                                ),
                                            ),
                                            generators=[
                                                ast.comprehension(
                                                    target=ast.Tuple(
                                                        elts=[
                                                            ast.Name(
                                                                id="s",
                                                                ctx=ast.Store(),
                                                            ),
                                                            ast.Name(
                                                                id="p",
                                                                ctx=ast.Store(),
                                                            ),
                                                        ],
                                                        ctx=ast.Store(),
                                                    ),
                                                    iter=ast.Call(
                                                        func=ast.Name(
                                                            id="zip", ctx=ast.Load()
                                                        ),
                                                        args=[
                                                            ast.Attribute(
                                                                value=ast.Name(
                                                                    id="x",
                                                                    ctx=ast.Load(),
                                                                ),
                                                                attr="shape",
                                                                ctx=ast.Load(),
                                                            ),
                                                            node.args[1],
                                                        ],
                                                        keywords=[],
                                                    ),
                                                    ifs=[],
                                                    is_async=0,
                                                )
                                            ],
                                        )
                                    ],
                                    keywords=[],
                                ),
                                ast.Attribute(
                                    value=ast.Name(id="x", ctx=ast.Load()),
                                    attr="dtype",
                                    ctx=ast.Load(),
                                ),
                            ),
                        ),
                    ],
                    keywords=[],
                )

            if lib_name in self.cupy_aliases and func_name == "exp":
                return ast.Call(
                    func=ast.Name(id="call_with_temp_arg", ctx=ast.Load()),
                    args=[
                        node.args[0],
                        ast.Lambda(
                            args=ast.arguments(
                                posonlyargs=[],
                                args=[ast.arg(arg="x")],
                                kwonlyargs=[],
                                defaults=[],
                            ),
                            body=FakeCuPyArray.new_ast_node(
                                node.lineno,
                                ast.Attribute(
                                    value=ast.Name(id="x", ctx=ast.Load()),
                                    attr="shape",
                                    ctx=ast.Load(),
                                ),
                                ast.Attribute(
                                    value=ast.Name(id="x", ctx=ast.Load()),
                                    attr="dtype",
                                    ctx=ast.Load(),
                                ),
                            ),
                        ),
                    ],
                    keywords=[],
                )

            if lib_name == "module" and func_name == "get_function":
                self.custom_kernels.append(node.args[0].value)
                return self.generic_visit(node)

        if isinstance(node.func, ast.Name) and node.func.id in [
            "rfft",
            "irfft",
            "fft",
            "ifft2",
        ]:
            array_arg = node.args[0]
            shape = ast.Attribute(
                value=ast.copy_location(array_arg, array_arg),
                attr="shape",
                ctx=ast.Load(),
            )
            dtype = ast.Attribute(
                value=ast.copy_location(array_arg, array_arg),
                attr="dtype",
                ctx=ast.Load(),
            )

            return ast.Call(
                func=ast.Name(id="fake_fft", ctx=ast.Load()),
                args=[
                    ast.Constant(value=node.func.id),
                    FakeCuPyArray.new_ast_node(node.lineno, shape, dtype),
                ],
                keywords=[],
            )

        if isinstance(node.func, ast.Name) and node.func.id == "rfftfreq":
            shape = ast.BinOp(
                left=ast.BinOp(
                    left=node.args[0],
                    op=ast.FloorDiv(),
                    right=ast.Constant(value=2),
                ),
                op=ast.Add(),
                right=ast.Constant(value=1),
            )
            dtype = ast.Constant(value="float64")
            return FakeCuPyArray.new_ast_node(node.lineno, shape, dtype)

        if isinstance(node.func, ast.Name) and node.func.id == "calc_filter":
            shape = ast.BinOp(
                left=ast.BinOp(
                    left=node.args[0],
                    op=ast.FloorDiv(),
                    right=ast.Constant(value=2),
                ),
                op=ast.Add(),
                right=ast.Constant(value=1),
            )
            dtype = ast.Constant(value="float32")
            return FakeCuPyArray.new_ast_node(node.lineno, shape, dtype)

        if isinstance(node.func, ast.Name) and node.func.id in self.custom_kernels:
            return ast.Call(
                func=ast.Name(id="noop", ctx=ast.Load()),
                args=[],
                keywords=[],
            )

        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if node.name in ["fake_fft", "call_with_temp_arg", "noop"]:
            return self.generic_visit(node)

        node.name = self.output_function_name
        node.args.args = [arg for arg in node.args.args if arg.arg != "self"]
        for arg in node.args.args:
            if (
                isinstance(arg.annotation, ast.Attribute)
                and arg.annotation.attr == "ndarray"
            ):
                arg.annotation = ast.Name(id="FakeCuPyArray", ctx=ast.Load())
        node.returns = None

        def shape_index(index):
            return ast.Subscript(
                value=ast.Attribute(
                    value=ast.Name(id="input", ctx=ast.Load()),
                    attr="shape",
                    ctx=ast.Load(),
                ),
                slice=ast.Constant(value=index),
                ctx=ast.Load(),
            )

        def call_cufft_estimate_1d(cufft_type):
            return ast.Assign(
                targets=[ast.Name(id="plan_size", ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id="cufft_estimate_1d", ctx=ast.Load()),
                    args=[],
                    keywords=[
                        ast.keyword(arg="nx", value=shape_index(2)),
                        ast.keyword(
                            arg="fft_type",
                            value=ast.Attribute(
                                value=ast.Name(id="CufftType", ctx=ast.Load()),
                                attr=cufft_type,
                                ctx=ast.Load(),
                            ),
                        ),
                        ast.keyword(arg="batch", value=shape_index(1)),
                    ],
                ),
            )

        def call_cufft_estimate_2d(cuff_type):
            return ast.Assign(
                targets=[ast.Name(id="plan_size", ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id="cufft_estimate_2d", ctx=ast.Load()),
                    args=[],
                    keywords=[
                        ast.keyword(arg="nx", value=shape_index(2)),
                        ast.keyword(arg="ny", value=shape_index(1)),
                        ast.keyword(
                            arg="fft_type",
                            value=ast.Attribute(
                                value=ast.Name(id="CufftType", ctx=ast.Load()),
                                attr=cuff_type,
                                ctx=ast.Load(),
                            ),
                        ),
                    ],
                ),
            )

        function_def = ast.FunctionDef(
            name="call_with_temp_arg",
            args=ast.arguments(
                args=[ast.arg(arg="input"), ast.arg(arg="func")],
                posonlyargs=[],
                defaults=[],
                kwonlyargs=[],
            ),
            body=[
                ast.Return(
                    value=ast.Call(
                        func=ast.Name(id="func", ctx=ast.Load()),
                        args=[ast.Name(id="input", ctx=ast.Load())],
                        keywords=[],
                    )
                ),
            ],
            decorator_list=[],
        )
        node.body.insert(0, function_def)

        function_def = ast.FunctionDef(
            name="fake_fft",
            args=ast.arguments(
                args=[ast.arg(arg="fft_function"), ast.arg(arg="input")],
                posonlyargs=[],
                defaults=[],
                kwonlyargs=[],
            ),
            body=[
                ast.Nonlocal(names=["fft_plan_cache"]),
                ast.Match(
                    subject=ast.Name(id="fft_function", ctx=ast.Load()),
                    cases=[
                        ast.match_case(
                            pattern=ast.MatchValue(ast.Constant(value="rfft")),
                            body=[call_cufft_estimate_1d("CUFFT_R2C")],
                        ),
                        ast.match_case(
                            pattern=ast.MatchValue(ast.Constant(value="irfft")),
                            body=[call_cufft_estimate_1d("CUFFT_C2R")],
                        ),
                        ast.match_case(
                            pattern=ast.MatchValue(ast.Constant(value="fft")),
                            body=[call_cufft_estimate_1d("CUFFT_C2C")],
                        ),
                        ast.match_case(
                            pattern=ast.MatchValue(ast.Constant(value="ifft2")),
                            body=[call_cufft_estimate_2d("CUFFT_C2C")],
                        ),
                    ],
                ),
                ast.Expr(
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="fft_plan_cache", ctx=ast.Load()),
                            attr="append",
                            ctx=ast.Load(),
                        ),
                        args=[ast.Name(id="plan_size", ctx=ast.Load())],
                        keywords=[],
                    )
                ),
                ast.Return(value=ast.Name(id="input", ctx=ast.Load())),
            ],
            decorator_list=[],
        )
        node.body.insert(0, function_def)

        assign_node = ast.Assign(
            targets=[ast.Name(id="fft_plan_cache", ctx=ast.Store())],
            value=ast.List(elts=[], ctx=ast.Load()),
        )
        node.body.insert(0, assign_node)

        noop_def = ast.FunctionDef(
            name="noop",
            args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], defaults=[]),
            body=[ast.Pass()],
            decorator_list=[],
        )
        node.body.insert(0, noop_def)

        new_body = []
        for stmt in node.body:
            if isinstance(stmt, ast.Return):
                stmt = ast.Assign(
                    targets=[ast.Name(id="return_value", ctx=ast.Store())],
                    value=stmt.value,
                )
            new_body.append(stmt)

        node.body = new_body

        return self.generic_visit(node)

    def visit_Attribute(self, node):
        current = node
        while isinstance(current, ast.Attribute):
            current = current.value

        if isinstance(current, ast.Name) and current.id == "self":
            return ast.Subscript(
                value=ast.Name(id="kwargs", ctx=ast.Load()),
                slice=ast.Constant(value=node.attr),
                ctx=ast.Load(),
            )

        return self.generic_visit(node)


def infer_aliases_from_globals(func: FunctionType) -> set[str]:
    return {
        name
        for name, _ in func.__globals__.items()
        if name == "cupy" or name == "cp" or name == "xp"
    }


def memory_estimator_from_function(func: FunctionType) -> FunctionType:
    output_function_name = f"_calc_memory_bytes_{func.__name__}"
    source = inspect.getsource(func)
    source = textwrap.dedent(source)
    tree = ast.parse(source)

    transformer = CuPyMemoryPrintTransformer(
        output_function_name, infer_aliases_from_globals(func)
    )
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)

    print(ast.unparse(tree))

    global_namespace = dict(func.__globals__)
    global_namespace["FakeCuPyArray"] = FakeCuPyArray
    global_namespace["CufftType"] = CufftType
    global_namespace["cufft_estimate_1d"] = cufft_estimate_1d
    global_namespace["cufft_estimate_2d"] = cufft_estimate_2d

    namespace: Dict[str, Any] = {}
    exec(compile(tree, filename="<ast>", mode="exec"), global_namespace, namespace)
    return namespace[output_function_name]


original_func = RecToolsDIRCuPy.__dict__["FOURIER_INV"]
estimator = memory_estimator_from_function(original_func)


import os
import tomophantom
from tomophantom import TomoP3D

model = 13  # select a model number from the library
N_size = 256  # Define phantom dimensions using a scalar value (cubic phantom)
path = os.path.dirname(tomophantom.__file__)
path_library3D = os.path.join(path, "phantomlib", "Phantom3DLibrary.dat")

phantom_tm = TomoP3D.Model(model, N_size, path_library3D)

# Projection geometry related parameters:
Horiz_det = int(np.sqrt(2) * N_size)  # detector column count (horizontal)
Vert_det = N_size  # detector row count (vertical) (no reason for it to be > N)
angles_num = int(0.3 * np.pi * N_size)  # angles number
angles = np.linspace(0.0, 179.9, angles_num, dtype="float32")  # in degrees
angles_rad = angles * (np.pi / 180.0)

projData3D_analyt = TomoP3D.ModelSino(
    model, N_size, Horiz_det, Vert_det, angles, path_library3D
)
input_data_labels = ["detY", "angles", "detX"]

Fourier_cupy = estimator(
    FakeCuPyArray(-1, projData3D_analyt.shape, projData3D_analyt.dtype),
    recon_mask_radius=0.95,
    data_axes_labels_order=input_data_labels,
    recon_size=Horiz_det,
    centre_of_rotation=1281,
    angles_vec=angles_rad,
)
