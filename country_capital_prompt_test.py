# -*- coding: utf-8 -*-
from assertpy import assert_that
from huggingface_hub import InferenceClient
from pytest import fixture

from country_capital_prompt import run_blocking_text_generation, run_blocking_chat_completion
from utils import smollm_instruct_model, set_environment, clear_environment, ENV


@fixture(scope="function")
def env_setup():
    set_environment(ENV)
    yield
    clear_environment(ENV)


@fixture(scope="class")
def client():
    return InferenceClient(smollm_instruct_model)


# noinspection PyUnusedLocal
class TestCountryCapitalPrompt:
    @staticmethod
    @fixture(scope="class")
    def prompt():
        return "The capital of france is"

    @staticmethod
    def test_run_blocking_text_generation(env_setup, client, prompt):
        result = run_blocking_text_generation(
            client,
            prompt
        )
        print(result)
        assert_that(result).contains_ignoring_case("Paris")

    @staticmethod
    def test_run_blocking_chat_completion(env_setup, client, prompt):
        result = (run_blocking_chat_completion(
            client,
            prompt)
                  .choices[0]
                  .message.content)
        print(result)
        assert_that(result).contains_ignoring_case("Paris")

    # @staticmethod
    # @pytest.mark.asyncio  # Marquer le test comme asynchrone
    # async def test_run_text_generation(env_setup, hf_client, prompt):
    #     # Capture la sortie standard pour vérifier le texte généré
    #     import sys
    #     from io import StringIO
    #
    #     # Rediriger stdout dans un buffer
    #     stdout = StringIO()
    #     sys.stdout = stdout
    #
    #     # Exécuter la fonction asynchrone
    #     await run_text_generation(
    #         hf_client,
    #         prompt
    #     )
    #
    #     # Restaurer stdout
    #     sys.stdout = sys.__stdout__
    #
    #     # Récupérer la sortie capturée
    #     output = stdout.getvalue()
    #     assert_that(output).contains_ignoring_case("Paris")
