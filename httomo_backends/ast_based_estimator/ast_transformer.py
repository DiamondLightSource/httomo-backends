import ast
import inspect
import textwrap
from types import FunctionType
from typing import Any, Dict


class MemoryEstimatorTransformer(ast.NodeTransformer):
    def __init__(self, output_function_name):
        self.output_function_name = output_function_name
        self.function_calls_in_statement = set()

    def visit_FunctionDef(self, node):
        new_args = []
        for arg in node.args.args:
            if arg.arg == "self":
                continue

            if arg.arg == "data":
                new_args.append(ast.arg(arg="data_shape", annotation=None))
                new_args.append(ast.arg(arg="data_dtype", annotation=None))
                continue

            new_args.append(arg)

        new_body = [
            ast.Assign(
                targets=[ast.Name(id="data", ctx=ast.Store())],
                value=ast.Call(
                    lineno=node.lineno,
                    func=ast.Attribute(
                        value=ast.Name(id="cp", ctx=ast.Load()),
                        attr="ndarray",
                        ctx=ast.Load(),
                    ),
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

            visited_statement = self.generic_visit(statement)
            new_body.append(visited_statement)

        new_body.append(
            ast.Return(
                value=ast.Subscript(
                    value=ast.Name(id="memory_usage", ctx=ast.Load()),
                    slice=ast.Constant(value="peak_memory"),
                    ctx=ast.Load(),
                )
            )
        )

        node.name = self.output_function_name
        node.args.args = new_args
        node.body = new_body
        node.returns = ast.Name(id="int", ctx=ast.Load())

        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        call_name = None
        if isinstance(node.func, ast.Name):
            call_name = node.func.id
        if isinstance(node.func, ast.Attribute) and isinstance(
            node.func.value, ast.Name
        ):
            call_name = node.func.attr

        if call_name in ["globals"]:
            return node

        self.function_calls_in_statement.add(node.func)

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


def memory_estimator_from_function(func: FunctionType) -> FunctionType:
    output_function_name = f"_calc_memory_bytes_{func.__name__}"
    source = inspect.getsource(func)
    source = textwrap.dedent(source)
    tree = ast.parse(source)

    transformer = MemoryEstimatorTransformer(output_function_name)
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)

    print(ast.unparse(tree))

    memory_usage = {"peak_memory": 0, "current_peak_memory": 0}
    fft_plan_cache = {}

    global_namespace = dict(func.__globals__)
    global_namespace["memory_usage"] = memory_usage
    global_namespace["fft_plan_cache"] = fft_plan_cache

    namespace: Dict[str, Any] = {}
    exec(compile(tree, filename="<ast>", mode="exec"), global_namespace, namespace)
    namespace[output_function_name].__globals__[output_function_name] = namespace[
        output_function_name
    ]
    return namespace[output_function_name]


# original_func = RecToolsDIRCuPy.__dict__["FOURIER_INV"]
# estimator = memory_estimator_from_function(original_func)


# import os
# import tomophantom
# from tomophantom import TomoP3D

# model = 13  # select a model number from the library
# N_size = 256  # Define phantom dimensions using a scalar value (cubic phantom)
# path = os.path.dirname(tomophantom.__file__)
# path_library3D = os.path.join(path, "phantomlib", "Phantom3DLibrary.dat")

# phantom_tm = TomoP3D.Model(model, N_size, path_library3D)

# # Projection geometry related parameters:
# Horiz_det = int(np.sqrt(2) * N_size)  # detector column count (horizontal)
# Vert_det = N_size  # detector row count (vertical) (no reason for it to be > N)
# angles_num = int(0.3 * np.pi * N_size)  # angles number
# angles = np.linspace(0.0, 179.9, angles_num, dtype="float32")  # in degrees
# angles_rad = angles * (np.pi / 180.0)

# projData3D_analyt = TomoP3D.ModelSino(
#     model, N_size, Horiz_det, Vert_det, angles, path_library3D
# )
# input_data_labels = ["detY", "angles", "detX"]

# print(
#     estimator(
#         projData3D_analyt.shape,
#         projData3D_analyt.dtype,
#         recon_mask_radius=0.95,
#         data_axes_labels_order=input_data_labels,
#         recon_size=Horiz_det,
#         centre_of_rotation=1281,
#         angles_vec=angles_rad,
#     )
# )
