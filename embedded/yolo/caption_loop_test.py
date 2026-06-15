import time
import shutil
import subprocess
import threading
import queue

import cv2
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from gtts import gTTS


CAMERA_INDEX = 0
MODEL_NAME = "Salesforce/blip-image-captioning-base"

CAPTION_INTERVAL = 5.0
TTS_PATH = "/tmp/caption_loop_tts.mp3"

caption_queue = queue.Queue(maxsize=1)
caption_running = False
stop_flag = False


def speak_text(text):
    print(f"[TTS] {text}")

    try:
        tts = gTTS(text=text, lang="en")
        tts.save(TTS_PATH)

        if shutil.which("mpg123"):
            subprocess.run(["mpg123", "-q", TTS_PATH])
        elif shutil.which("paplay"):
            subprocess.run(["paplay", TTS_PATH])
        else:
            print("[TTS] mpg123 또는 paplay가 없어 음성 재생은 생략합니다.")
    except Exception as e:
        print(f"[TTS ERROR] {e}")


def load_caption_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[MODEL] device = {device}")
    print("[MODEL] loading BLIP...")

    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model = BlipForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        use_safetensors=False
    )

    model.to(device)
    model.eval()

    print("[MODEL] loaded")
    return processor, model, device


def generate_caption(frame, processor, model, device):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)

    inputs = processor(image, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=30,
            num_beams=3
        )

    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption.strip()


def caption_worker(processor, model, device):
    global caption_running

    while not stop_flag:
        try:
            frame = caption_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        caption_running = True
        start = time.time()

        try:
            caption = generate_caption(frame, processor, model, device)
            elapsed = time.time() - start

            print(f"[CAPTION] {caption}")
            print(f"[TIME] caption elapsed: {elapsed:.2f}s")

            speak_text(caption)

        except Exception as e:
            print(f"[CAPTION ERROR] {e}")

        finally:
            caption_running = False
            caption_queue.task_done()


def main():
    global stop_flag

    print("=== Caption Loop Test Start ===")

    processor, model, device = load_caption_model()

    worker = threading.Thread(
        target=caption_worker,
        args=(processor, model, device),
        daemon=True
    )
    worker.start()

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError(f"카메라를 열 수 없습니다: /dev/video{CAMERA_INDEX}")

    last_caption_time = 0.0
    frame_count = 0
    fps_start = time.time()

    try:
        while True:
            ret, frame = cap.read()

            if not ret or frame is None:
                print("[CAMERA] frame read failed")
                time.sleep(0.1)
                continue

            frame_count += 1
            now = time.time()

            if now - fps_start >= 5.0:
                fps = frame_count / (now - fps_start)
                print(f"[FPS] camera loop fps: {fps:.2f}")
                frame_count = 0
                fps_start = now

            if now - last_caption_time >= CAPTION_INTERVAL:
                if caption_running:
                    print("[SKIP] previous caption still running")
                elif caption_queue.full():
                    print("[SKIP] caption queue full")
                else:
                    caption_queue.put(frame.copy())
                    print("[QUEUE] frame submitted for caption")

                last_caption_time = now

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[STOP] Ctrl+C received")

    finally:
        stop_flag = True
        cap.release()
        print("=== Caption Loop Test Done ===")


if __name__ == "__main__":
    main()
