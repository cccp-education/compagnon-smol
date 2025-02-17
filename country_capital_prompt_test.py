# -*- coding: utf-8 -*-
from assertpy import assert_that
from huggingface_hub import InferenceClient
from pytest import fixture
from pytest import mark

from country_capital_prompt import blocking_text_generation, blocking_chat_completion, chat_completion
from utils import smollm_instruct_model, set_environment, clear_environment, ENV

mark.asyncio


@fixture(scope="function")
def env_setup():
    set_environment(ENV)
    yield
    clear_environment(ENV)


@fixture(scope="class")
def client():
    return InferenceClient(smollm_instruct_model)


class TestCountryCapitalPrompt:
    @staticmethod
    @fixture(scope="class")
    def prompt():
        return "The capital of France is"

    # noinspection PyUnusedLocal
    @staticmethod
    def test_run_blocking_text_generation(env_setup, client, prompt):
        result = blocking_text_generation(client, prompt)
        assert_that(result).is_type_of(str)
        assert_that(result).is_not_empty()
        assert_that(result).contains_ignoring_case("Paris")
        print(result)

    # noinspection PyUnusedLocal
    @staticmethod
    def test_run_blocking_chat_completion(env_setup, client, prompt):
        result = (blocking_chat_completion(client, prompt)
                  .choices[0]
                  .message.content)
        assert_that(result).is_type_of(str)
        assert_that(result).is_not_empty()
        assert_that(result).contains_ignoring_case("Paris")
        print(result)

    # noinspection PyUnusedLocal
    @staticmethod
    @mark.asyncio
    async def test_run_text_generation(env_setup, client, prompt):
        result = ""
        try:
            for chunk in await chat_completion(client, prompt):
                content = chunk.choices[0].delta.content or ""
                result += content
                print(content, end='', flush=True)
        except Exception as e:
            assert_that(e).is_none()
        assert_that(result).contains_ignoring_case("Paris")
