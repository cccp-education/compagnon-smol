# -*- coding: utf-8 -*-
from huggingface_hub import InferenceClient

from utils import set_environment, smollm_model, ENV, clear_environment


# @smolagents.tool
def plus(a: int, b: int) -> int:
    """Multiply two integers."""
    return a + b


# @smolagents.tool
def multiply(a: int, b: int) -> int:
    """Sum two integers."""
    return a * b


if __name__ == '__main__':
    set_environment(ENV)
    client = InferenceClient(smollm_model)
    operand = 2
    print(f"{operand}+{operand}={plus(operand, operand)}")
    print(f"{operand}*{operand}={multiply(operand, operand)}")
    clear_environment(ENV)
