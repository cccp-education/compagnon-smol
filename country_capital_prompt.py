# -*- coding: utf-8 -*-

from __future__ import annotations

import asyncio

from huggingface_hub import InferenceClient

from utils import (
    set_environment, ENV, smollm_model,
    smollm_instruct_model, display_chat_completion)

if __name__ == '__main__':
    set_environment(ENV)
    base = InferenceClient(smollm_model)
    instruct = InferenceClient(smollm_instruct_model)

    # weird_prompt = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>
    # The capital of Paris is<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    # """
    # print(base.text_generation(
    #     prompt=weird_prompt, max_new_tokens=100, temperature=0.99
    # ))

    prompt = "The capital of france is"
    print(
        instruct.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            max_tokens=1024,
        ).choices[0].message.content
    )

    asyncio.run(display_chat_completion(instruct, prompt))
