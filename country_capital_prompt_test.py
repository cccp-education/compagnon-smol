# -*- coding: utf-8 -*-
from assertpy import assert_that
from huggingface_hub import InferenceClient
from pytest import fixture
from pytest import mark

from country_capital_prompt import blocking_text_generation, blocking_chat_completion, text_generation
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
        result = (blocking_chat_completion(
            client,
            prompt)
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
        # Capture la sortie standard pour vérifier le texte généré
        import sys
        from io import StringIO

        # Rediriger stdout dans un buffer
        stdout = StringIO()
        sys.stdout = stdout

        # Exécuter la fonction asynchrone
        await text_generation(client, prompt)

        # Restaurer stdout
        sys.stdout = sys.__stdout__

        # Récupérer la sortie capturée
        output = stdout.getvalue()
        assert_that(output).contains_ignoring_case("Paris")
