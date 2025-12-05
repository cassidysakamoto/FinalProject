# pip install pyaudio SpeechRecognition
import time, struct, sys
import pyaudio
import speech_recognition as sr

# Audio and framing
RATE = 16000
FRAME_MS = 20
SAMPLES_PER_FRAME = int(RATE * FRAME_MS / 1000)     # 320
BYTES_PER_SAMPLE = 2
BYTES_PER_FRAME = SAMPLES_PER_FRAME * BYTES_PER_SAMPLE
CHUNK_FRAMES = int(2000 / FRAME_MS)                 # 2 s => 100 frames

# Logic
RETRY_LIMIT = 3
ENERGY_MIN = 200

# Domain replacements and light punctuation
DOMAIN_REPLACE = {
    "r c": "RC",
    "p w m": "PWM",
    "k h z": "kHz",
    "h z": "Hz",
    "m s": "ms",
    "so c": "SoC",
}

recognizer = sr.Recognizer()
pa = pyaudio.PyAudio()

def frame_energy(frame_bytes: bytes) -> int:
    n = len(frame_bytes) // 2
    if n == 0:
        return 0
    total = 0
    for i in range(0, len(frame_bytes), 2):
        total += abs(struct.unpack("<h", frame_bytes[i:i+2])[0])
    return total // n

def recognize_google_raw(raw_pcm: bytes) -> str:
    audio = sr.AudioData(raw_pcm, RATE, BYTES_PER_SAMPLE)
    return recognizer.recognize_google(audio)

def normalize_text(text: str) -> str:
    t = " " + text.lower() + " "
    for k, v in DOMAIN_REPLACE.items():
        t = t.replace(f" {k} ", f" {v} ")
    t = " ".join(t.split())
    if t and t[-1] not in ".!?":
        t += "."
    return t[0].upper() + t[1:] if t else t

def main():
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=RATE,
                     input=True, frames_per_buffer=SAMPLES_PER_FRAME)
    if not stream.is_active():
        print("Input stream not active")
        sys.exit(1)

    print("Listening at 16 kHz with 20 ms frames. Ctrl+C to stop.\n")

    frames, partials = [], []
    chunk_id = 0
    t_chunk_start = time.time()
    warned_quiet = False

    try:
        while True:
            frame = stream.read(SAMPLES_PER_FRAME, exception_on_overflow=False)

            # Input checks
            if len(frame) != BYTES_PER_FRAME:
                print("Warning dropped frame due to size mismatch")
                continue

            if frame_energy(frame) < ENERGY_MIN:
                if not warned_quiet:
                    print("Warning mic level low in current chunk")
                    warned_quiet = True
            else:
                warned_quiet = False

            frames.append(frame)

            # 2 s aggregation
            if len(frames) == CHUNK_FRAMES:
                raw = b"".join(frames)
                frames.clear()
                t_chunk_end = time.time()

                if not raw:
                    print(f"[chunk {chunk_id}] Empty chunk, skipping")
                    t_chunk_start = time.time()
                    continue

                # Retry logic with warnings
                text, err = "", None
                for attempt in range(1, RETRY_LIMIT + 1):
                    try:
                        text = recognize_google_raw(raw)
                        break
                    except sr.UnknownValueError:
                        err = "speech not understood"
                        print(f"Warning chunk {chunk_id} could not understand speech")
                        break
                    except sr.RequestError as e:
                        err = f"api {e}"
                        if attempt < RETRY_LIMIT:
                            print(f"Warning chunk {chunk_id} retry {attempt} of {RETRY_LIMIT} due to {e}")
                            time.sleep(0.25)
                        else:
                            print(f"Warning chunk {chunk_id} failed after {RETRY_LIMIT} attempts due to {e}")

                # Partial output
                timing = f"[chunk {chunk_id} {t_chunk_start:.3f} to {t_chunk_end:.3f}]"
                if text:
                    out = normalize_text(text)
                    partials.append(out)
                    print(f"{timing} {out}")
                else:
                    print(f"{timing} (no text, {err})")

                chunk_id += 1
                t_chunk_start = time.time()
                warned_quiet = False

    except KeyboardInterrupt:
        pass
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
        full = " ".join(partials)
        print("\n=== Summary ===")
        print(f"Chunks processed: {chunk_id}")
        print(f"Transcript: {full if full else '(none)'}")

if __name__ == "__main__":
    main()
