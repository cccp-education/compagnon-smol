# -*- coding: utf-8 -*-
from assertpy import assert_that
from pytest import fixture
from pytest import mark

from utils import (
    set_environment, clear_environment,
    ENV, hf_base_client, hf_instruct_client,
    blocking_text_generation,
    blocking_chat_completion, chat_completion)

mark.asyncio


@fixture(scope="function")
def env():
    set_environment(ENV)
    yield
    clear_environment(ENV)


@fixture(scope="class")
def hf_base(): return hf_base_client()


@fixture(scope="class")
def hf_instruct(): return hf_instruct_client()


class TestCountryCapitalPrompt:

    @staticmethod
    @fixture(scope="class")
    def prompt():
        return "The capital of France is"

    @staticmethod
    def test_run_blocking_text_generation_huggingface(env, hf_base, prompt):
        result = blocking_text_generation(hf_base, prompt)
        assert_that(result).is_type_of(str)
        assert_that(result).is_not_empty()
        print(result)

    @staticmethod
    def test_run_blocking_chat_completion_huggingface_instruct(env, hf_instruct, prompt):
        content = (blocking_chat_completion(hf_instruct, prompt)
                   .choices[0]
                   .message.content)
        assert_that(content).is_type_of(str)
        assert_that(content).is_not_empty()
        assert_that(content).contains_ignoring_case("Paris")
        print(content)

    @staticmethod
    @mark.asyncio
    async def test_run_chat_completion_huggingface_instruct(env, hf_instruct, prompt):
        content = ""
        try:
            for chunk in await chat_completion(hf_instruct, prompt):
                delta = chunk.choices[0].delta.content or ""
                content += delta
                print(delta, end='', flush=True)
        except Exception as e:
            assert_that(e).is_none()
        assert_that(content).contains_ignoring_case("Paris")
