# -*- coding: utf-8 -*-

from assertpy import assert_that
from huggingface_hub import InferenceClient
from loguru import logger
from pytest import fixture, mark

from calculator_tool import (
    multiply, plus, PlusTool, MultiplyTool)
from utils import (
    set_environment, clear_environment,
    ENV, hf_base_client, hf_instruct_client, llama_instruct_client, tool_to_hf_format)

mark.asyncio


@fixture(scope="function")
def env():
    set_environment(ENV)
    yield
    clear_environment(ENV)


@fixture(scope="class")
def hf_base() -> InferenceClient:
    return hf_base_client()


@fixture(scope="class")
def hf_instruct() -> InferenceClient:
    return hf_instruct_client()


@fixture(scope="class")
def llama_instruct() -> InferenceClient:
    return llama_instruct_client()


class TestCalculatorTool:

    @staticmethod
    def test_trivial_multiply():
        assert_that(multiply(2, 2)).is_equal_to(4)

    @staticmethod
    def test_trivial_plus():
        assert_that(plus(2, 2)).is_equal_to(4)

    @staticmethod
    def test_plus_tool_description():
        logger.info(f"Tool description: {PlusTool.description}")
        (assert_that(PlusTool.description)
         .contains_ignoring_case("Sum two numbers."))

    @staticmethod
    def test_multiply_tool_description():
        logger.info(MultiplyTool.description)
        assert_that(MultiplyTool.description).contains_ignoring_case("Multiply two numbers.")

    @staticmethod
    def test_plus_tool(env, hf_instruct: InferenceClient):
        logger.info(PlusTool.description)
        result = hf_instruct.chat.completions.create(
            messages=[{"role": "user", "content": "Calculate 2 + 2"}],
            tools=[tool_to_hf_format(PlusTool),
                   tool_to_hf_format(MultiplyTool)],
            tool_choice="auto"
        )
        logger.info(f"Result: {result}")

    #     logger.info(multiply.to_string())
    # results = {}
    # operand = 2
    #
    # prompt_plus = f"Calculate {operand} + {operand}"
    # content_plus = ""
    # # for chunk in
    # res = blocking_chat_completion(hf_base_client, prompt_plus)
    # # delta = chunk.choices[0].delta.content or ""
    # # content_plus += delta
    # logger.info(res)
    # results[f"plus_{operand}"] = {
    #     "input": (operand, operand),
    #     "result": content_plus,
    #     "description": f"{operand} + {operand} = {content_plus}"
    # }

    # print(results[f"plus_{operand}"]["result"])

    # @staticmethod
    # async def test_plus(env, hf_instruct):
    #     results = {}
    #     operand = 2
    #
    #     prompt_plus = f"Calculate {operand} + {operand}"
    #     content_plus = ""
    #     for chunk in await chat_completion(hf_instruct, prompt_plus):
    #         delta = chunk.choices[0].delta.content or ""
    #         content_plus += delta
    #     results[f"plus_{operand}"] = {
    #         "input": (operand, operand),
    #         "result": content_plus,
    #         "description": f"{operand} + {operand} = {content_plus}"
    #     }
    #
    #     print(results[f"plus_{operand}"]["result"])

    # prompt_multiply = f"Calculate {operand} * {operand}"
    # content_multiply = ""
    # async for chunk in chat_completion(hf_instruct, prompt_multiply):
    #     delta = chunk.choices[0].delta.content or ""
    #     content_multiply += delta
    # results[f"multiply_{operand}"] = {
    #     "input": (operand, operand),
    #     "result": content_multiply,
    #     "description": f"{operand} * {operand} = {content_multiply}"
    # }
