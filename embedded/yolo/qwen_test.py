import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto"
)

messages = [
    {
        "role": "system",
        "content": (
            "너는 시각장애인 보행 보조 장치의 화면해설 문장을 만드는 도우미다. "
            "영어 캡션을 한국어 한 문장으로 자연스럽게 바꿔라. "
            "원문에 없는 물체, 위험, 거리, 방향은 추가하지 마라."
        )
    },
    {
        "role": "user",
        "content": "English caption: a laptop computer sitting on top of a desk"
    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer([text], return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=False
    )

generated = outputs[0][inputs.input_ids.shape[-1]:]
answer = tokenizer.decode(generated, skip_special_tokens=True)

print("answer:", answer.strip())
