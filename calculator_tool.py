# -*- coding: utf-8 -*-
from smolagents import tool


# class SmollAgentsToolError(Exception):
#     """Exception raised for errors in the SmollAgents tools."""
#     pass

@tool
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


@tool
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
