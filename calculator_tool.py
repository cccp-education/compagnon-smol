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