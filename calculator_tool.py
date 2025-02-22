# -*- coding: utf-8 -*-

from smolagents import Tool


def plus(a: int, b: int) -> int:
    """Sum two numbers.

    Args:
        a: First number to add
        b: Second number to add

    Returns:
        The sum of a and b

    Example:
        >>> plus(2, 3)
        5
    """
    return a + b


def multiply(a: int, b: int) -> int:
    """Multiply two numbers.

    Args:
        a: First number to multiply
        b: Second number to multiply

    Returns:
        The product of a and b

    Example:
        >>> multiply(2, 3)
        6
    """
    return a * b


class PlusTool(Tool):
    name = "plus"
    description = "Sum two numbers."
    function = plus
    inputs = [("a", "int"), ("b", "int")]
    output_type = "int"

    def __call__(self, a: int, b: int) -> int:
        return a + b


class MultiplyTool(Tool):
    name = "multiply"
    description = "Multiply two numbers."
    function = multiply
    inputs = [("a", "int"), ("b", "int")]
    output_type = "int"
    def __call__(self, a: int, b: int) -> int:
        return a * b

def calculator_tool_format(tool):
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "num1": {"type": "integer", "description": "The first number"},
                    "num2": {"type": "integer", "description": "The second number"},
                },
                "required": ["num1", "num2"],
            },
        },
    }

if __name__ == '__main__':
    print(calculator_tool_format(PlusTool))

# import inspect
# from typing import Any, get_origin, get_args
#
# class Tool:
#     """Generic reusable code tool with automatic metadata extraction"""
#
#     def __init__(self, name, description, func, arguments, outputs):
#         self.name = name
#         self.description = description
#         self.func = func
#         self.arguments = arguments
#         self.outputs = outputs
#
#     def __call__(self, *args, **kwargs):
#         return self.func(*args, **kwargs)
#
#     def to_string(self):
#         args_str = ", ".join([f"{name}: {type_}" for name, type_ in self.arguments])
#         return (
#             f"Tool: {self.name}\n"
#             f"Description: {self.description}\n"
#             f"Arguments: {args_str}\n"
#             f"Outputs: {self.outputs}"
#         )
#
# def tool(func):
#     """Universal decorator that converts functions into Tool instances"""
#     def get_type_name(t):
#         """Handles both basic types and complex generics"""
#         if t is inspect.Parameter.empty:
#             return 'Any'
#         if (origin := get_origin(t)) is not None:
#             origin_name = get_type_name(origin)
#             args = get_args(t)
#             if args:
#                 return f"{origin_name}[{', '.join(get_type_name(a) for a in args)}]"
#             return origin_name
#         return t.__name__ if hasattr(t, '__name__') else str(t)
#
#     # Extract function metadata
#     sig = inspect.signature(func)
#     args_list = []
#     for name, param in sig.parameters.items():
#         param_type = get_type_name(param.annotation)
#         args_list.append((name, param_type))
#     return_type = get_type_name(sig.return_annotation)
#     description = func.__doc__.strip() if func.__doc__ else "No description provided"
#     return Tool(
#         name=func.__name__,
#         description=description,
#         func=func,
#         arguments=args_list,
#         outputs=return_type
#     )
#
# # --- Example Usage ---
#
# @tool
# def string_processor(text: str, operations: list[str]) -> str:
#     """Applies multiple string operations sequentially"""
#     processed_text = text
#     for op in operations:
#         if op == 'upper':
#             processed_text = processed_text.upper()
#         elif op == 'lower':
#             processed_text = processed_text.lower()
#         # Add additional operations as needed
#     return processed_text
#
# @tool
# def data_transformer(
#         input_data: dict[str, Any],
#         schema: dict[str, type],
#         strict: bool = False
# ) -> list[tuple]:
#     """Converts dictionary data to typed tuples according to schema"""
#     transformed_data = []
#     for key, value in input_data.items():
#         expected_type = schema.get(key, type(value))
#         if strict and not isinstance(value, expected_type):
#             continue
#         transformed_data.append((key, value))
#     return transformed_data
#
# @tool
# def complex_example(
#         a: int,
#         b: list[tuple[str, float]],
#         c: dict[str, Any] = None
# ) -> tuple[bool, str]:
#     """Example function with complex type annotations"""
#     return (True, "Success")
#
# # --- Test the Implementation ---
# print(string_processor.to_string())