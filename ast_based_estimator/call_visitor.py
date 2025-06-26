import ast
from fake_cupy_array import FakeCuPyArray


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

            return FakeCuPyArray.new_ast_node(
                node.lineno,
                ast.Name(id="memory_usage", ctx=ast.Load()),
                shape,
                dtype,
            )

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
                            ast.Name(id="memory_usage", ctx=ast.Load()),
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
                            ast.Name(id="memory_usage", ctx=ast.Load()),
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
                FakeCuPyArray.new_ast_node(
                    node.lineno,
                    ast.Name(id="memory_usage", ctx=ast.Load()),
                    shape,
                    dtype,
                ),
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
        return FakeCuPyArray.new_ast_node(
            node.lineno, ast.Name(id="memory_usage", ctx=ast.Load()), shape, dtype
        )

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
        return FakeCuPyArray.new_ast_node(
            node.lineno, ast.Name(id="memory_usage", ctx=ast.Load()), shape, dtype
        )

    if isinstance(node.func, ast.Name) and node.func.id in self.custom_kernels:
        return ast.Call(
            func=ast.Name(id="noop", ctx=ast.Load()),
            args=[],
            keywords=[],
        )

    return None