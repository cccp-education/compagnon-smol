# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from typing import Iterable, Dict

from assertpy import assert_that
from huggingface_hub import (
    ChatCompletionOutput, ChatCompletionStreamOutput,
    InferenceClient)
from loguru import logger

from config import (
    CODESTRAL_API_KEY, HUGGINGFACE_API_KEY,
    GOOGLE_API_KEY, MISTRAL_API_KEY,
    SMOLLM2_MODEL, SMOLLM2_INSTRUCT_MODEL, LLAMA_3_2_INSTRUCT_MODEL)

logger.add("sys.stdout", level="INFO")

ENV: Dict[str, str] = {
    "HUGGINGFACE_API_KEY": HUGGINGFACE_API_KEY,
    "GOOGLE_API_KEY": GOOGLE_API_KEY,
    "MISTRAL_API_KEY": MISTRAL_API_KEY,
    "CODESTRAL_API_KEY": CODESTRAL_API_KEY,
    "SMOLLM2_MODEL": SMOLLM2_MODEL,
    "SMOLLM2_INSTRUCT_MODEL": SMOLLM2_INSTRUCT_MODEL,
    "LLAMA_3_2_INSTRUCT_MODEL": LLAMA_3_2_INSTRUCT_MODEL
}

HF_TOKEN = ENV["HUGGINGFACE_API_KEY"]
smollm_model = ENV["SMOLLM2_MODEL"]
smollm_instruct_model = ENV["SMOLLM2_INSTRUCT_MODEL"]
llama_instruct_model = ENV["LLAMA_3_2_INSTRUCT_MODEL"]


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


def hf_instruct_client() -> InferenceClient:
    logger.debug("Creating Hugging Face instruct client.")
    return InferenceClient(smollm_instruct_model)


def llama_instruct_client() -> InferenceClient:
    logger.debug("Creating Llama3.2 instruct client.")
    return InferenceClient(llama_instruct_model)


def hf_base_client() -> InferenceClient:
    logger.debug("Creating Hugging Face base client.")
    return InferenceClient(smollm_model)


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
        # print(chunk, end='', flush=True)
        logger.debug(f"Received chunk: {chunk}")  # Log each chunk
    logger.info("Streaming text generation finished.")


# noinspection PyShadowingNames
async def chat_completion(
        client: InferenceClient,
        prompt: str) -> ChatCompletionOutput | Iterable[ChatCompletionStreamOutput]:
    logger.info(f"Streaming chat completion started for prompt: {prompt}")
    return client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        max_tokens=1024,
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

def tool_to_hf_format(tool):
    return {
        "type": "function",
        "function": {
            "name": tool.name,  # changed to access the name attribute directly
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "num1": {"type": "integer", "description": "The first number"},
                    "num2": {"type": "integer", "description": "The second number"},
                },
                "required": ["num1", "num2"],
            },
        },
    }

# TODO: Add connection
def gemini_client():
    return 0
