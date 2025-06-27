import ast
import inspect
import cupy as cp
import numpy as np
import textwrap
from types import FunctionType
from typing import Any, Dict, Tuple, Union
from httomo_backends.cufft import CufftType, cufft_estimate_1d, cufft_estimate_2d
from tomobar.methodsDIR_CuPy import RecToolsDIRCuPy
from fake_cupy_array import FakeCuPyArray


class CuPyMemoryPrintTransformer(ast.NodeTransformer):
    def __init__(self, output_function_name, cupy_aliases=set[str]):
        self.output_function_name = output_function_name
        self.cupy_aliases = cupy_aliases
        self.custom_kernel_modules = []
        self.custom_kernels = []

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if isinstance(node.func, ast.Name) and node.func.id in self.custom_kernels:
            return ast.Call(
                func=ast.Name(id="noop", ctx=ast.Load()),
                args=[],
                keywords=[],
            )

        node.args.append(
            ast.Dict(
                keys=[
                    ast.Constant(value="line_number"),
                    ast.Constant(value="memory_usage"),
                    ast.Constant(value="fft_plan_cache"),
                ],
                values=[
                    ast.Constant(value=node.lineno),
                    ast.Name(id="memory_usage", ctx=ast.Load()),
                    ast.Name(id="fft_plan_cache", ctx=ast.Load()),
                ],
            )
        )
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if node.name in ["noop"]:
            return self.generic_visit(node)

        node.name = self.output_function_name

        new_args = []
        for arg in node.args.args:
            if arg.arg == "self":
                continue

            if arg.arg == "data":
                new_args.append(ast.arg(arg="data_shape", annotation=None))
                new_args.append(ast.arg(arg="data_dtype", annotation=None))
                continue

            new_args.append(arg)

        node.args.args = new_args

        node.returns = None

        new_body = [
            ast.FunctionDef(
                name="noop",
                args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], defaults=[]),
                body=[ast.Pass()],
                decorator_list=[],
            ),
            ast.Assign(
                targets=[ast.Name(id="memory_usage", ctx=ast.Store())],
                value=ast.Dict(
                    keys=[
                        ast.Constant(value="peak_memory"),
                        ast.Constant(value="current_peak_memory"),
                    ],
                    values=[
                        ast.Constant(value=0),
                        ast.Constant(value=0),
                    ],
                ),
            ),
            ast.Assign(
                targets=[ast.Name(id="fft_plan_cache", ctx=ast.Store())],
                value=ast.List(elts=[], ctx=ast.Load()),
            ),
            ast.Assign(
                targets=[ast.Name(id="data", ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id="ndarray", ctx=ast.Load()),
                    args=[
                        ast.Name(id="data_shape", ctx=ast.Load()),
                        ast.Name(id="data_dtype", ctx=ast.Load()),
                    ],
                    keywords=[],
                ),
            ),
        ]

        for statement in node.body:
            if (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)
            ):
                continue

            if isinstance(statement, ast.Return):
                statement = ast.Assign(
                    targets=[ast.Name(id="return_value", ctx=ast.Store())],
                    value=statement.value,
                )
            new_body.append(statement)

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

    def visit_Assign(self, node):
        if (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "load_cuda_module"
        ):
            self.custom_kernel_modules.append(node.targets)
            node.value = None

        if (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id in self.custom_kernel_modules
            and node.value.func.attr == "get_function"
        ):
            self.custom_kernels.append(node.targets)
            node.value = None

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

print(
    estimator(
        projData3D_analyt.shape,
        projData3D_analyt.dtype,
        recon_mask_radius=0.95,
        data_axes_labels_order=input_data_labels,
        recon_size=Horiz_det,
        centre_of_rotation=1281,
        angles_vec=angles_rad,
    )
)
