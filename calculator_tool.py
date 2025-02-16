# -*- coding: utf-8 -*-
from huggingface_hub import InferenceClient

from utils import set_environment, smollm_instruct_model


# @smolagents.tool
def plus(a: int, b: int) -> int:
    return a + b


# @smolagents.tool
def multiply(a: int, b: int) -> int:
    return a * b


if __name__ == '__main__':
    set_environment()
    client = InferenceClient(smollm_instruct_model)
    operand = 2
    print(f"{operand}+{operand}={plus(operand, operand)}")
    print(f"{operand}*{operand}={multiply(operand, operand)}")
