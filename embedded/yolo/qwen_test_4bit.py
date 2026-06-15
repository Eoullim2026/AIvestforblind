import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

print("cuda:", torch.cuda.is_available())

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

messages = [
    {
        "role": "system",
        "content": (
            "You translate short English image captions into natural Korean. "
            "Do not add information. Output only one Korean sentence."
        ),
    },
    {
        "role": "user",
        "content": (
            "English: a person standing in a room\n"
            "Korean: 방 안에 사람이 서 있습니다.\n\n"
            "English: a laptop computer sitting on top of a desk\n"
            "Korean:"
        ),
    },
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer([text], return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=False,
        temperature=None,
        top_p=None,
    )

generated = outputs[0][inputs.input_ids.shape[-1]:]
answer = tokenizer.decode(generated, skip_special_tokens=True)

print("answer:", answer.strip())
