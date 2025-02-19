# -*- coding: utf-8 -*-

from assertpy import assert_that
from pytest import fixture, mark

from utils import ENV, set_environment, clear_environment, hf_base_client, hf_instruct_client
from huggingface_hub import InferenceClient

class TestWeatherTool:
    @staticmethod
    @fixture(scope="function")
    def env():
        set_environment(ENV)
        yield
        clear_environment(ENV)

    @staticmethod
    @fixture(scope="class")
    def hf_base() -> InferenceClient:
        return hf_base_client()

    @staticmethod
    @fixture(scope="class")
    def hf_instruct() -> InferenceClient:
        return hf_instruct_client()


    @staticmethod
    def test_canary():
        assert_that(2 + 2).is_equal_to(4)
