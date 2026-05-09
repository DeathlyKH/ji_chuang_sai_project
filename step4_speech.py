#!/usr/bin/env python3
"""
Step 4: Whisper + Silero VAD
Fast + Accurate + Anti-hallucination
"""

import numpy as np
import threading
import time
import os
from collections import deque

try:
    import pyaudio
    PYAUDIO_OK = True
except ImportError:
    PYAUDIO_OK = False

try:
    import whisper
    WHISPER_OK = True
except ImportError:
    WHISPER_OK = False

try:
    import torch
    import torchaudio
    TORCHAUDIO_OK = True
except ImportError:
    TORCHAUDIO_OK = False


class SpeechModule:
    COMMAND_MAP = {
        "你好小熊": "hello_bear",
        "你好": "hello_bear",
        "小熊": "hello_bear",
        "嗨小熊": "hello_bear",
        "你好吗": "hello_bear",
        "拍照": "take_photo",
        "拍": "take_photo",
        "照": "take_photo",
        "茄子": "take_photo",
        "合影": "take_photo",
        "再见": "goodbye",
        "拜拜": "goodbye",
        "bye": "goodbye",
    }

    def __init__(self, model_size="tiny"):
        self.is_running = False
        self.current_command = ""
        self.command_time = 0
        self.last_trigger = 0
        self.cooldown = 1.5
        self._audio_buffer = []
        self._buffer_lock = threading.Lock()

        self.whisper_model = None
        self.vad_model = None
        self.mode = "simulate"

        if not WHISPER_OK or not PYAUDIO_OK:
            print("[Speech] Whisper/PyAudio unavailable. Simulate mode.")
            return

        # Load Silero VAD
        if TORCHAUDIO_OK:
            try:
                self.vad_model, utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    onnx=False
                )
                self.vad_model.eval()
                print("[Speech] Silero VAD loaded.")
            except Exception as e:
                print(f"[Speech] VAD load failed: {e}")
                self.vad_model = None
        else:
            self.vad_model = None

        # Load Whisper
        try:
            print(f"[Speech] Loading Whisper '{model_size}'...")
            self.whisper_model = whisper.load_model(model_size)
            self.mode = "whisper"
            print("[Speech] Whisper ready!")
            print("[Speech] Speak clearly in quiet environment.")
            print("[Speech] Put mic 5cm from your mouth.")
        except Exception as e:
            print(f"[Speech] Whisper failed: {e}")
            self.mode = "simulate"

    def start(self):
        self.is_running = True
        if self.mode == "simulate":
            threading.Thread(target=self._sim_loop, daemon=True).start()
        else:
            threading.Thread(target=self._record_loop, daemon=True).start()
            threading.Thread(target=self._recognize_loop, daemon=True).start()

    def stop(self):
        self.is_running = False

    def _record_loop(self):
        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=512
        )

        # Silero VAD expects 30ms frames at 16kHz = 480 samples
        # We collect 1.5s chunks for Whisper
        COLLECT_SECONDS = 1.5
        collect_frames = int(16000 * COLLECT_SECONDS / 512)
        temp_buffer = []

        print("[Speech] Mic ON.")

        while self.is_running:
            try:
                data = stream.read(512, exception_on_overflow=False)
                chunk = np.frombuffer(data, dtype=np.int16)
                temp_buffer.append(chunk)

                if len(temp_buffer) >= collect_frames:
                    # Concatenate
                    audio_np = np.concatenate(temp_buffer).astype(np.float32) / 32768.0
                    temp_buffer = temp_buffer[collect_frames // 3:]  # Overlap 2/3

                    # VAD check
                    if self._has_speech(audio_np):
                        with self._buffer_lock:
                            self._audio_buffer.append(audio_np)

            except Exception as e:
                print(f"[Speech] Record error: {e}")

        stream.stop_stream()
        stream.close()
        pa.terminate()

    def _has_speech(self, audio_np):
        """Silero VAD: check if audio contains speech"""
        if self.vad_model is None:
            # Fallback: energy check
            return np.sqrt(np.mean(audio_np ** 2)) > 0.02

        try:
            # Convert to torch tensor
            tensor = torch.from_numpy(audio_np)
            # VAD expects specific format
            speech_prob = self.vad_model(tensor, 16000).item()
            return speech_prob > 0.5
        except Exception:
            # Fallback
            return np.sqrt(np.mean(audio_np ** 2)) > 0.02

    def _recognize_loop(self):
        while self.is_running:
            time.sleep(0.2)

            audio = None
            with self._buffer_lock:
                if len(self._audio_buffer) > 0:
                    audio = self._audio_buffer.pop(0)

            if audio is None:
                continue

            print("\n[LISTENING] ...", end="", flush=True)

            try:
                result = self.whisper_model.transcribe(
                    audio,
                    language="zh",
                    fp16=False,
                    verbose=None,
                    condition_on_previous_text=False,
                    initial_prompt="以下是普通话句子。",
                    temperature=0.0,
                    no_speech_threshold=0.6,
                    logprob_threshold=-1.0,
                )
                text = result.get("text", "").strip()
                # Clean
                for c in " ,。?？！!\"'":
                    text = text.replace(c, "")

                if text and len(text) >= 2:
                    print(f" Heard: '{text}'")
                    self._match_command(text)
                else:
                    print(" (silence)")

            except Exception as e:
                print(f"[Speech] Error: {e}")

    def _sim_loop(self):
        import random
        while self.is_running:
            time.sleep(5)
            cmd = random.choice(["hello_bear", "take_photo", "goodbye"])
            self.current_command = cmd
            self.command_time = time.time()
            print(f"[Sim] {cmd}")

    def _match_command(self, text):
        now = time.time()
        if now - self.last_trigger < self.cooldown:
            return

        for keyword, cmd in self.COMMAND_MAP.items():
            if keyword in text:
                self.current_command = cmd
                self.command_time = now
                self.last_trigger = now
                print(f"[Speech] >>> {cmd} <<<")
                return

        print(f"[Speech] (no match: '{text}')")

    def get_command(self):
        now = time.time()
        if now - self.command_time > 3.0:
            return None
        cmd = self.current_command
        self.current_command = ""
        return cmd if cmd else None

    def get_state(self):
        return {"mode": self.mode}


if __name__ == "__main__":
    import cv2
    speech = SpeechModule(model_size="tiny")
    speech.start()

    cap = cv2.VideoCapture(0)
    print("\nSpeak: ni hao xiao xiong / pai zhao / zai jian")
    print("Press 'q' to quit")

    last_cmd = ""
    show_until = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        cmd = speech.get_command()
        if cmd:
            last_cmd = cmd
            show_until = time.time() + 2.0

        state = speech.get_state()
        lines = [
            f"Mode: {state['mode']}",
            f"Last: {last_cmd}",
            "Speak: ni hao xiao xiong",
            "Speak: pai zhao",
            "Speak: zai jian",
            "Press 'q' to quit"
        ]
        for i, t in enumerate(lines):
            cv2.putText(frame, t, (10, 30 + i * 28),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 136), 2)

        if time.time() < show_until and last_cmd:
            cv2.putText(frame, f">>> {last_cmd} <<<", (w // 2 - 120, h // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 200, 255), 3)

        cv2.imshow("Whisper+VAD Speech", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    speech.stop()
    cap.release()
    cv2.destroyAllWindows()