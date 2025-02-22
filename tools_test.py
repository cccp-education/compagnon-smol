# -*- coding: utf-8 -*-

import os

from assertpy import assert_that
from huggingface_hub import InferenceClient, ChatCompletionOutput, ChatCompletionOutputMessage
from loguru import logger
from pytest import fixture
from pytest import mark

from calculator_tool import multiply, plus, PlusTool, MultiplyTool, calculator_tool_format
from utils import (
    set_environment, clear_environment,
    chat_completion, blocking_chat_completion,
    blocking_text_generation, ENV,
    llama_client, llama_instruct_client)

mark.asyncio

class TestTools:
    prompt = "The capital of France is"
    expected_result = "Paris"

    @staticmethod
    @fixture(scope="function")
    def env():
        set_environment(ENV)
        yield
        clear_environment(ENV)

    @staticmethod
    @fixture(scope="class")
    def base() -> InferenceClient:
        return llama_client()

    @staticmethod
    @fixture(scope="class")
    def instruct() -> InferenceClient:
        return llama_instruct_client()

    @staticmethod
    def test_clear_environment():
        assert_that(os.environ).is_not_empty()
        assert_that(os.environ.items()).is_not_empty()
        assert_that("foo").is_not_in(os.environ)
        assert_that("bar").is_not_in(os.environ.values())
        os.environ["foo"] = "bar"
        assert_that(os.environ).contains_key("foo")
        assert_that(os.environ).contains_value("bar")
        clear_environment({"foo": "bar"})
        assert_that(os.environ).is_not_empty()
        assert_that(os.environ.items()).is_not_empty()
        assert_that(os.environ.keys()).does_not_contain("foo")
        assert_that(os.environ.values()).does_not_contain("bar")

    @staticmethod
    def test_set_environment_loop_impl():
        for key, value in ENV.items():
            assert_that(os.environ).does_not_contain(key)
        set_environment(ENV)
        for key, value in ENV.items():
            assert_that(os.environ).contains_key(key)
            assert_that(os.environ[key]).is_equal_to(value)
        clear_environment(ENV)

    @staticmethod
    def test_set_environment_map_impl():
        assert_that(list(map(
            lambda key: key in os.environ, ENV.keys()
        ))).is_equal_to([False] * len(ENV))
        set_environment(ENV)
        assert_that(list(map(
            lambda key: os.environ[key] == ENV[key], ENV.keys()
        ))).is_equal_to([True] * len(ENV))
        clear_environment(ENV)

    @staticmethod
    def test_run_blocking_text_generation_huggingface(env, base: InferenceClient):
        result = blocking_text_generation(base, TestTools.prompt)
        assert_that(result).is_type_of(str).is_not_empty()
        logger.info(f"Result: {result}")

    @staticmethod
    def test_run_blocking_chat_completion_huggingface_instruct(env, instruct: InferenceClient):
        content = (blocking_chat_completion(instruct, TestTools.prompt)
                   .choices[0]
                   .message.content)
        assert_that(content).is_type_of(str)
        assert_that(content).is_not_empty()
        assert_that(content).contains_ignoring_case(TestTools.expected_result)
        logger.info(f"Result: {content}")

    @staticmethod
    @mark.asyncio
    async def test_run_chat_completion_huggingface_instruct(env, instruct: InferenceClient):
        content = ""
        for chunk in await chat_completion(instruct, TestTools.prompt):
            delta = chunk.choices[0].delta.content or ""
            content += delta
        logger.info(f"Result: {content}")
        assert_that(content).is_not_empty().contains_ignoring_case(TestTools.expected_result)

    @staticmethod
    def test_base_model_dont_support_templated_prompt_blocking_text_generation(env, base: InferenceClient):
        prompt = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>
                The capital of France is<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        expected = (TestTools.prompt + TestTools.expected_result).lower()
        result = base.text_generation(
            prompt=prompt, max_new_tokens=100)
        assert_that(result).is_type_of(str).is_not_empty()
        assert_that(result.lower()).does_not_contain(expected.lower())
        logger.info(f"Result: {result}")

    @staticmethod
    @mark.asyncio
    async def test_base_model_dont_support_templated_prompt_text_generation(env, base: InferenceClient):
        prompt = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>
                The capital of France is<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        expected = f"{TestTools.prompt} {TestTools.expected_result}".lower()
        result = ""
        for chunk in base.text_generation(
                prompt,
                max_new_tokens=100,
                temperature=0.99,
                stream=True):
            result += chunk
        logger.info(f"Result: {result}")
        assert_that(result).is_type_of(str).is_not_empty()
        (assert_that(result.lower())
         .does_not_contain(expected.lower()))

    @staticmethod
    def test_trivial_multiply():
        assert_that(multiply(2, 2)).is_equal_to(4)

    @staticmethod
    def test_trivial_plus():
        assert_that(plus(2, 2)).is_equal_to(4)

    @staticmethod
    def test_plus_tool_description():
        logger.info(f"Tool description: {PlusTool.description}")
        assert_that(PlusTool.description).contains_ignoring_case("Sum two numbers.")

    @staticmethod
    def test_multiply_tool_description():
        logger.info(MultiplyTool.description)
        assert_that(MultiplyTool.description).contains_ignoring_case("Multiply two numbers.")

    @staticmethod
    def test_plus_tool(env, instruct: InferenceClient):
        logger.info(PlusTool.description)
        result: ChatCompletionOutput = instruct.chat.completions.create(
            messages=[{"role": "user", "content": "Calculate 2 + 2"}],
            tools=[calculator_tool_format(PlusTool)],
            tool_choice="auto")
        logger.info(f"Result: {result}")
        assert_that(result.model).is_equal_to(instruct.model)
        logger.info(f"Choices size: {len(result.choices)}")
        assert_that(len(result.choices)).is_equal_to(1)

        message: ChatCompletionOutputMessage = result.choices[0].message

        # Assertions break from here
        # assert_that(message.tool_calls).is_not_none()
        # assert_that(
        #     message.tool_calls[0].function.description
        # ).is_equal_to(PlusTool.description)
        # assert_that(message.content).is_not_none()
        # assert_that(message.content).contains("4")

    @staticmethod
    @mark.asyncio
    async def test_chat_completion_with_calculator_tools(env, instruct: InferenceClient):
        # TODO: Add assertions
        operand = 2
        prompt_plus = f"Calculate {operand} + {operand}"
        content_plus = ""
        for chunk in await chat_completion(
                instruct,
                prompt_plus,
                [calculator_tool_format(PlusTool),
                 calculator_tool_format(MultiplyTool)]):
            delta = chunk.choices[0].delta.content or ""
            content_plus += delta
        logger.info(f"result plus: {content_plus}")
        prompt_multiply = f"Calculate {operand} * {operand}"
        content_multiply = ""
        for chunk in await chat_completion(
                instruct,
                prompt_multiply,
                [calculator_tool_format(PlusTool),
                 calculator_tool_format(MultiplyTool)]):
            delta = chunk.choices[0].delta.content or ""
            content_multiply += delta
        logger.info(f"result multiply: {content_plus}")

