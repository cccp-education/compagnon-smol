# -*- coding: utf-8 -*-

from huggingface_hub import InferenceClient

from utils import set_environment, smollmInstruct

if __name__ == '__main__':
    set_environment()

    client = InferenceClient(smollmInstruct)

    output = client.text_generation("The capital of france is", max_new_tokens=100)

    print(output)

    prompt = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>
    The capital of Paris is<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    output = client.text_generation(
        prompt,
        max_new_tokens=100,
    )

    print(output)

    output = client.chat.completions.create(
        messages=[{"role": "user", "content": "The capital of france is"}],
        stream=False,
        max_tokens=1024,
    )

    print(output.choices[0].message.content)
