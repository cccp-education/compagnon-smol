# -*- coding: utf-8 -*-
from huggingface_hub import InferenceClient

from utils import set_environment, smollmInstruct


# @smolagents.tool
def plus(a: int, b: int) -> int:
    return a + b


# @smolagents.tool
def multiply(a: int, b: int) -> int:
    return a * b


if __name__ == '__main__':
    set_environment()
    client = InferenceClient(smollmInstruct)
    operand = 2
    print(f"{operand}+{operand}={plus(operand, operand)}")
    print(f"{operand}*{operand}={multiply(operand, operand)}")
