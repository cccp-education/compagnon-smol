# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
from typing import Iterable

from huggingface_hub import InferenceClient, ChatCompletionOutput, ChatCompletionStreamOutput

from utils import set_environment, ENV, smollm_instruct_model


def blocking_text_generation(
        client: InferenceClient,
        prompt: str) -> str:
    return client.text_generation(
        prompt,
        max_new_tokens=100,
        temperature=0.98)


async def text_generation(
        client: InferenceClient,
        prompt: str):
    for chunk in client.text_generation(
            prompt,
            max_new_tokens=100,
            temperature=0.99,
            stream=True):
        print(chunk, end='', flush=True)


def blocking_chat_completion(
        client: InferenceClient,
        prompt: str) -> ChatCompletionOutput | Iterable[ChatCompletionStreamOutput]:
    return client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        max_tokens=1024,
    )


async def chat_completion(
        client: InferenceClient,
        prompt: str) -> ChatCompletionOutput | Iterable[ChatCompletionStreamOutput]:
    return client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        max_tokens=1024,
    )


async def display_chat_completion(
        client: InferenceClient,
        prompt: str):
    for chunk in await chat_completion(client, prompt):
        print(
            chunk.choices[0].delta.content,
            end='',
            flush=True
        )


if __name__ == '__main__':
    set_environment(ENV)
    client = InferenceClient(smollm_instruct_model)
    prompt = "The capital of france is"
    asyncio.run(display_chat_completion(client, prompt))

# prompt = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>
# The capital of Paris is<|eot_id|><|start_header_id|>assistant<|end_header_id|>
# """
# output = client.text_generation(
#     prompt,
#     max_new_tokens=100,
# )
#
# print(output)
#
# output = client.chat.completions.create(
#     messages=[{"role": "user", "content": "The capital of france is"}],
#     stream=False,
#     max_tokens=1024,
# )
#
# print(output.choices[0].message.content)

# from huggingface_hub import InferenceClient
#
# from utils import set_environment, smollm_instruct_model
#
# if __name__ == '__main__':
#     set_environment()
#
#     client = InferenceClient(smollm_instruct_model)
#
#     output = client.text_generation(
#         "The capital of france is",
#         max_new_tokens=100,
#         temperature=0.99,
#         stream=True)
#
#     print(output)
#
#     # prompt = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>
#     # The capital of Paris is<|eot_id|><|start_header_id|>assistant<|end_header_id|>
#     # """
#     # output = client.text_generation(
#     #     prompt,
#     #     max_new_tokens=100,
#     # )
#     #
#     # print(output)
#     #
#     # output = client.chat.completions.create(
#     #     messages=[{"role": "user", "content": "The capital of france is"}],
#     #     stream=False,
#     #     max_tokens=1024,
#     # )
#     #
#     # print(output.choices[0].message.content)
