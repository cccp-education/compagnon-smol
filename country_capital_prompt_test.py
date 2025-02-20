# -*- coding: utf-8 -*-
from typing import Generator

from assertpy import assert_that
from huggingface_hub import InferenceClient
from loguru import logger
from pytest import fixture
from pytest import mark

from utils import (
    set_environment, clear_environment,
    ENV, hf_base_client, hf_instruct_client,
    chat_completion, blocking_chat_completion, blocking_text_generation)

mark.asyncio


@fixture(scope="function")
def env() -> Generator:
    set_environment(ENV)
    yield
    clear_environment(ENV)


@fixture(scope="class")
def hf_base():
    return hf_base_client()


@fixture(scope="class")
def hf_instruct():
    return hf_instruct_client()


class TestCountryCapitalPrompt:
    prompt = "The capital of France is"
    expected_result = "Paris"

    @staticmethod
    def test_run_blocking_text_generation_huggingface(
            env, hf_base: InferenceClient):
        logger.info(f"env.keys() : {ENV.keys()}")
        assert_that(ENV).contains_key("SMOLLM2_MODEL")
        result = blocking_text_generation(hf_base, TestCountryCapitalPrompt.prompt)
        assert_that(result).is_type_of(str).is_not_empty()
        logger.info(f"Result: {result}")

    @staticmethod
    def test_run_blocking_chat_completion_huggingface_instruct(env, hf_instruct):
        content = (blocking_chat_completion(hf_instruct, TestCountryCapitalPrompt.prompt)
                   .choices[0]
                   .message.content)
        assert_that(content).is_type_of(str)
        assert_that(content).is_not_empty()
        assert_that(content).contains_ignoring_case(TestCountryCapitalPrompt.expected_result)
        logger.info(f"Result: {content}")

    @staticmethod
    @mark.asyncio
    async def test_run_chat_completion_huggingface_instruct(env, hf_instruct):
        content = ""
        try:
            for chunk in await chat_completion(hf_instruct, TestCountryCapitalPrompt.prompt):
                delta = chunk.choices[0].delta.content or ""
                content += delta
                print(delta, end='', flush=True)
        except Exception as e:
            assert_that(e).is_none()
        assert_that(content).contains_ignoring_case(TestCountryCapitalPrompt.expected_result)

    @staticmethod
    def test_base_model_dont_support_templated_prompt_blocking_text_generation(env, hf_base):
        prompt = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>
            The capital of France is<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        expected = (TestCountryCapitalPrompt.prompt + TestCountryCapitalPrompt.expected_result).lower()
        result = hf_base.text_generation(
            prompt=prompt, max_new_tokens=100)
        assert_that(result).is_type_of(str).is_not_empty()
        assert_that(result.lower()).does_not_contain(expected.lower())
        logger.info(f"Result: {result}")

    @staticmethod
    @mark.asyncio
    async def test_base_model_dont_support_templated_prompt_text_generation(env, hf_base):
        prompt = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>
            The capital of France is<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        expected = f"{TestCountryCapitalPrompt.prompt} {TestCountryCapitalPrompt.expected_result}".lower()
        result = ""
        for chunk in hf_base.text_generation(
                prompt,
                max_new_tokens=100,
                temperature=0.99,
                stream=True):
            result += chunk
            print(chunk, end='', flush=True)
        assert_that(result).is_type_of(str).is_not_empty()
        (assert_that(result.lower())
         .does_not_contain(expected.lower()))
