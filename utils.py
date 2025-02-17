# -*- coding: utf-8 -*-

import os

from assertpy import assert_that

from config import (
    CODESTRAL_API_KEY, HUGGINGFACE_API_KEY,
    GOOGLE_API_KEY, MISTRAL_API_KEY,
    SMOLLM2_MODEL)

ENV = {
    "HUGGINGFACE_API_KEY": HUGGINGFACE_API_KEY,
    "GOOGLE_API_KEY": GOOGLE_API_KEY,
    "MISTRAL_API_KEY": MISTRAL_API_KEY,
    "CODESTRAL_API_KEY": CODESTRAL_API_KEY,
    "SMOLLM2_MODEL": SMOLLM2_MODEL
}

HF_TOKEN = ENV["HUGGINGFACE_API_KEY"]
smollm_instruct_model = ENV["SMOLLM2_MODEL"]

def set_environment(env_vars):
    diff = {key: value for key, value in env_vars.items() if key not in os.environ}
    if len(diff) > 0: os.environ.update(diff)


def clear_environment(env_vars):
    for key in env_vars.keys():
        os.environ.pop(key, None)
        assert_that(os.environ.keys()).does_not_contain(key)
