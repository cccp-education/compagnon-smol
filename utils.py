# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from typing import Iterable, Dict

from assertpy import assert_that
from huggingface_hub import (
    ChatCompletionOutput, ChatCompletionStreamOutput,
    InferenceClient)
from loguru import logger

from calculator_tool import calculator_tool_format, PlusTool, MultiplyTool
from config import (
    CODESTRAL_API_KEY, HUGGINGFACE_API_KEY,
    GOOGLE_API_KEY, MISTRAL_API_KEY)

logger.add("sys.stdout", level="INFO")

HUGGINGFACE_API_ENV_KEY = "HUGGINGFACE_API_KEY"
GOOGLE_API_ENV_KEY = "GOOGLE_API_KEY"
MISTRAL_API_ENV_KEY = "MISTRAL_API_KEY"
CODESTRAL_API_ENV_KEY = "CODESTRAL_API_KEY"
llama_model = "meta-llama/Llama-3.2-3B"
llama_instruct_model = "meta-llama/Llama-3.2-3B-Instruct"

ENV: Dict[str, str] = {
    HUGGINGFACE_API_ENV_KEY: HUGGINGFACE_API_KEY,
    GOOGLE_API_ENV_KEY: GOOGLE_API_KEY,
    MISTRAL_API_ENV_KEY: MISTRAL_API_KEY,
    CODESTRAL_API_ENV_KEY: CODESTRAL_API_KEY,
}

HF_TOKEN = ENV[HUGGINGFACE_API_ENV_KEY]


def set_environment(env_vars) -> None:
    diff = {key: value for key, value in env_vars.items() if key not in os.environ}
    if len(diff) > 0:
        os.environ.update(diff)
        logger.debug(f"Environment variables set: {diff.keys()}")  # Log the changes
    else:
        logger.debug("No new environment variables to set.")


def clear_environment(env_vars):
    for key in env_vars.keys():
        os.environ.pop(key, None)
        assert_that(os.environ.keys()).does_not_contain(key)
    logger.debug(f"Environment variables cleared: {env_vars.keys()}")


def llama_instruct_client() -> InferenceClient:
    logger.debug("Creating Llama3.2 instruct client.")
    return InferenceClient(llama_instruct_model)


def llama_client() -> InferenceClient:
    logger.debug("Creating Hugging Face base client.")
    return InferenceClient(llama_model)


# noinspection PyShadowingNames
def blocking_text_generation(client: InferenceClient, prompt: str) -> str:
    logger.info(f"Blocking text generation started for prompt: {prompt}")
    result = client.text_generation(
        prompt,
        max_new_tokens=100,
        temperature=0.98)
    logger.info(f"Blocking text generation finished. Result: {result}")  # Log the result
    return result


# noinspection PyShadowingNames
def blocking_chat_completion(
        client: InferenceClient,
        prompt: str) -> ChatCompletionOutput | Iterable[ChatCompletionStreamOutput]:
    logger.info(f"Blocking chat completion started for prompt: {prompt}")
    result = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        max_tokens=1024,
    )
    logger.info(f"Blocking chat completion finished. Result: {result}")
    return result


# noinspection PyShadowingNames
async def text_generation(client: InferenceClient, prompt: str):
    logger.info(f"Streaming text generation started for prompt: {prompt}")
    for chunk in client.text_generation(
            prompt,
            max_new_tokens=100,
            temperature=0.99,
            stream=True):
        logger.debug(f"Received chunk: {chunk}")  # Log each chunk
    logger.info("Streaming text generation finished.")


# noinspection PyShadowingNames
async def chat_completion(
        client: InferenceClient,
        prompt: str
) -> ChatCompletionOutput | Iterable[ChatCompletionStreamOutput]:
    logger.info(f"Streaming chat completion started for prompt: {prompt}")
    return client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        max_tokens=1024,
        tools=[calculator_tool_format(PlusTool),
               calculator_tool_format(MultiplyTool)],
        tool_choice="auto",
    )


# noinspection PyShadowingNames
async def display_chat_completion(
        client: InferenceClient,
        prompt: str):
    for chunk in await chat_completion(client, prompt): print(
        chunk.choices[0].delta.content,
        end='',
        flush=True
    )


# TODO: Add connection
def gemini_client():
    return 0
