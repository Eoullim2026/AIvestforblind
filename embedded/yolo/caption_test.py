import os
import time
import shutil
import subprocess

import cv2
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from gtts import gTTS


CAMERA_INDEX = 0
FRAME_PATH = "/tmp/caption_frame.jpg"
TTS_PATH = "/tmp/caption_tts.mp3"

MODEL_NAME = "Salesforce/blip-image-captioning-base"


def capture_frame():
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError(f"카메라를 열 수 없습니다: /dev/video{CAMERA_INDEX}")

    # 카메라 밝기/노출 안정화용으로 몇 프레임 버림
    for _ in range(10):
        ret, frame = cap.read()
        time.sleep(0.05)

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError("프레임 캡처 실패")

    cv2.imwrite(FRAME_PATH, frame)
    print(f"[CAPTURE] saved: {FRAME_PATH}")

    return FRAME_PATH


def load_caption_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[MODEL] device = {device}")
    print("[MODEL] loading BLIP...")

    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)

    model.to(device)
    model.eval()

    print("[MODEL] loaded")

    return processor, model, device


def generate_caption(image_path, processor, model, device):
    image = Image.open(image_path).convert("RGB")

    inputs = processor(image, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=30,
            num_beams=3
        )

    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption.strip()


def speak_text(text):
    print(f"[TTS] {text}")

    tts = gTTS(text=text, lang="en")
    tts.save(TTS_PATH)

    if shutil.which("mpg123"):
        subprocess.run(["mpg123", "-q", TTS_PATH])
    elif shutil.which("paplay"):
        subprocess.run(["paplay", TTS_PATH])
    else:
        print("[TTS] mpg123 또는 paplay가 없어 음성 재생은 생략합니다.")


def main():
    print("=== Caption Test Start ===")

    image_path = capture_frame()

    processor, model, device = load_caption_model()

    start = time.time()
    caption = generate_caption(image_path, processor, model, device)
    elapsed = time.time() - start

    print(f"[CAPTION] {caption}")
    print(f"[TIME] caption elapsed: {elapsed:.2f}s")

    speak_text(caption)

    print("=== Caption Test Done ===")


if __name__ == "__main__":
    main()
