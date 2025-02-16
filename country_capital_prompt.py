# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
from typing import Iterable

from huggingface_hub import InferenceClient, ChatCompletionOutput, ChatCompletionStreamOutput

from utils import set_environment, ENV, smollm_instruct_model


def run_blocking_text_generation(
        client: InferenceClient,
        prompt: str) -> str:
    return client.text_generation(
        prompt,
        max_new_tokens=100,
        temperature=0.98)


async def run_text_generation(
        client: InferenceClient,
        prompt: str):
    output = client.text_generation(
        prompt,
        max_new_tokens=100,
        temperature=0.99,
        stream=True)
    for chunk in output:
        print(chunk, end='', flush=True)


def run_blocking_chat_completion(
        client: InferenceClient,
        prompt: str) -> ChatCompletionOutput | Iterable[ChatCompletionStreamOutput]:
    return client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        max_tokens=1024,
    )

if __name__ == '__main__':
    set_environment(ENV)
    client = InferenceClient(smollm_instruct_model)
    prompt = "The capital of france is"
    # run_blocking_text_generation(client, prompt)
    asyncio.run(run_text_generation(client, prompt))
    # run_blocking_chat_completion(client, prompt)

# print(output)

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
