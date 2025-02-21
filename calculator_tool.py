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


class MultiplyTool(Tool):
    name = "multiply"
    description = "Multiply two numbers."
    function = multiply
    inputs = [("a", "int"), ("b", "int")]
    output_type = "int"

