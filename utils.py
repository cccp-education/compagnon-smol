# -*- coding: utf-8 -*-

import os

from assertpy import assert_that

from config import (CODESTRAL_API_KEY, HUGGINGFACE_API_KEY,
                    GOOGLE_API_KEY, MISTRAL_API_KEY)

ENV = {
    "HUGGINGFACE_API_KEY": HUGGINGFACE_API_KEY,
    "GOOGLE_API_KEY": GOOGLE_API_KEY,
    "MISTRAL_API_KEY": MISTRAL_API_KEY,
    "CODESTRAL_API_KEY": CODESTRAL_API_KEY,
}

HF_TOKEN = ENV["HUGGINGFACE_API_KEY"]

talk = {
    'greetings': "Bonjour le monde!",
    'topic': """Le sujet qui m'intéresse ce sont les LLMs(Large Language Model) et les agents."""
}


def set_environment():
    diff = {key: value for key, value in ENV.items() if key not in os.environ}
    if len(diff) > 0: os.environ.update(diff)


def clear_environment(env_vars):
    for key in env_vars.keys():
        os.environ.pop(key, None)
        assert_that(os.environ.keys()).does_not_contain(key)
