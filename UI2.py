# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 12:53:59 2025

@author: cassi
"""

# =========================
# BACKEND: Mic + Transcription
# =========================

# Use sounddevice library - live audio recording with NumPy arrays
# has callback as a built-in feature
import sounddevice as sd
# print(sd.query_devices())  # Lists all audio devices with their IDs
# print(sd.default.device)   # Shows default input/output device IDs

# print(sd.query_devices(9, 'input'))

# Use numpy library - audio processing, needed for mic level
import numpy as np

# Other imports for transcription
import time, struct, sys, threading
import speech_recognition as sr

# ===========================================================

# Class for microphone input- reads audio in chunks
# ===========================================================
class MicrophoneInput:
    def __init__(self, threshold=0.0000001, sample_rate=48000, chunk_dur=0.02):
        self.threshold = threshold   # noise gate threshold
        self.rate = sample_rate      # audio sample rate
        self.chunk_dur = chunk_dur   # duration (s) of each audio chunk
        self.chunk_size = int(self.rate * chunk_dur)  # samples per chunk
        self.stream = None    # initialize stream as none (stream is the mic stream)
        self.callback = None   # initialize callback (audio_callback) as none
        # NEW: callback for mic level so UI can display it
        self.level_callback = None

    # processing audio input, called every time audio comes in from mic
    # indata is the audio chunk, frames are # of frames in chunk
    def audio_callback(self, indata, frames, time_info, status):

        # Check for errors
        if status:
            print(f"Stream status: {status}")
        # print("audio_callback called")

        audio_chunk = indata.flatten()  # convert audio into 1D NumPy array

        # Noise Gate:
        # root mean square = mean of NumPy array (each value is squared to turn them all pos.)
        # average volume
        rms = np.sqrt(np.mean(audio_chunk**2))  # loudness level

        # Mic Level Meter (visual feedback)
        bar = "#" * int(rms * 50)  # volume bar using #
        print(f"Mic Level: {bar} ({rms:.3f})")

        # NEW: send level to UI if a level_callback is set
        if self.level_callback:
            self.level_callback(float(rms))

        # If volume is lower than threshold...
        if rms < self.threshold:
            pass  # mark as silence but still pick up quiet noise

        # Send audio to transcriber  **connect code

        if self.callback:

            # Calls process_audio()!
            self.callback(audio_chunk)  # pass chunk to another function

    # Start listening to mic
    def start_stream(self, callback=None, level_callback=None):

        # save callback
        self.callback = callback
        # NEW: save level callback too
        self.level_callback = level_callback

        # Importing from the sounddevice library
        # InputStream is a class in sd library - allows an audio input stream
        self.stream = sd.InputStream(
            device=9,          # Device ID
            channels=1,        # 1 channel: Mono audio input
            samplerate=self.rate,  # sample rate for audio
            # telling sd library to call audio_callback whenever it receives audio
            callback=self.audio_callback  # callback is set equal to the audio_callback function
        )
        # Starting audio stream- calling function built into sounddevice library
        self.stream.start()
        print("Microphone stream started")

    def stop_stream(self):

        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            print("Microphone stream stopped")


# Part 2: transcription
# ===========================================================

# Audio and framing
RATE = 48000
FRAME_MS = 20
SAMPLES_PER_FRAME = int(RATE * FRAME_MS / 1000)     # 960 samples per 20 ms
BYTES_PER_SAMPLE = 2
CHUNK_FRAMES = int(6000 / FRAME_MS)                 # 6 s => 100 frames

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

# NEW: global callback so UI can receive finished text chunks
transcript_callback = None

# Helper functions for processing + recognition
# ===========================================================

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

# Integration between mic stream and transcription
# ===========================================================

frames = []          # will hold individual short chunks (~20ms)
chunk_id = 0
partials = []        # partial transcripts
t_chunk_start = time.time()

# Runs every time mic produces a chunk of audio
def process_audio(chunk_np):
    global frames, chunk_id, t_chunk_start

    # Convert float32 (-1.0 to 1.0) → 16-bit PCM bytes
    frame_bytes = (chunk_np * 32767).astype(np.int16).tobytes()
    frames.append(frame_bytes)

    # When enough 20 ms frames = ~2 s of audio, transcribe that chunk
    if len(frames) >= CHUNK_FRAMES:
        raw = b"".join(frames)
        frames.clear()
        t_chunk_end = time.time()

        # Run transcription in background so mic callback isn’t blocked
        threading.Thread(
            target=transcribe_chunk,
            args=(raw, chunk_id, t_chunk_start, t_chunk_end),
            daemon=True,
        ).start()

        chunk_id += 1
        t_chunk_start = time.time()

def transcribe_chunk(raw, chunk_id, t_start, t_end):
    """Performs transcription with retry logic"""
    global transcript_callback

    err = None
    text = ""
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
                print(f"Warning chunk {chunk_id} retry {attempt}/{RETRY_LIMIT}")
                time.sleep(0.25)
            else:
                print(f"Warning chunk {chunk_id} failed after {RETRY_LIMIT} attempts: {e}")

    timing = f"[chunk {chunk_id} {t_start:.3f}-{t_end:.3f}]"
    if text:
        out = normalize_text(text)
        partials.append(out)
        print(f"{timing} {out}")
        # NEW: send text to UI if a handler is registered
        if transcript_callback:
            try:
                transcript_callback(out)
            except Exception as e:
                print(f"Error in transcript_callback: {e}", file=sys.stderr)
    else:
        print(f"{timing} (no text, {err})")


# =========================
# UI + Notes App
# (from UI.py, adapted to use Mic + STT backend above)
# =========================

import json, uuid, random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Callable, Optional
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ---------------- Basics ----------------
DATA_DIR = Path.cwd() / "stt_ui_data"
DATA_DIR.mkdir(exist_ok=True)
NOTES_FILE = DATA_DIR / "notes.json"
CONFIG_FILE = DATA_DIR / "config.json"
DICT_FILE = DATA_DIR / "dictionary.json"

def now_ms() -> int:
    return int(time.time() * 1000)

def format_time(ms: int) -> str:
    s = ms // 1000
    m, sec = (s % 3600) // 60, s % 60
    return f"{m:02d}:{sec:02d}"

def slugify(name: str) -> str:
    keep = "".join(c if (c.isalnum() or c in " -_") else "_" for c in name)
    return "_".join(keep.strip().split())

# ---------------- Data model ----------------
@dataclass
class Note:
    id: str
    title: str
    content: str
    tags: List[str]
    createdAt: int
    updatedAt: int
    durationMs: int = 0

def load_notes() -> List[Note]:
    if NOTES_FILE.exists():
        try:
            raw = json.loads(NOTES_FILE.read_text(encoding="utf-8"))
            return [Note(**n) for n in raw]
        except Exception:
            pass
    return []

def save_notes(notes: List[Note]) -> None:
    NOTES_FILE.write_text(json.dumps([asdict(n) for n in notes], indent=2), encoding="utf-8")

# --- config / dictionary helpers ---
def load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    # defaults
    return {
        "autosave_enabled": True,
        "autosave_delay_ms": 800,
        "noise_gate_threshold": 0.002,
        "chunk_ms": 2000,
        "max_retries": 3,
        "use_local_model": True,
    }

def save_config(cfg: dict) -> None:
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

def load_dictionary() -> List[str]:
    if DICT_FILE.exists():
        try:
            data = json.loads(DICT_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [str(w) for w in data]
        except Exception:
            pass
    return []

def save_dictionary(words: List[str]) -> None:
    DICT_FILE.write_text(json.dumps(sorted(set(words)), indent=2), encoding="utf-8")

# ---------------- (Original) Recorder adapter ----------------
class RecorderAdapter:
    def start(self, on_level: Callable[[float], None]) -> None: ...
    def stop(self) -> None: ...
    def is_running(self) -> bool: return False

class DummyRecorder(RecorderAdapter):
    """UI dev stub: fakes a level so you can design the meter & flows."""
    def __init__(self, root: tk.Tk):
        self.root = root
        self._running = False
        self._cb: Optional[Callable[[float], None]] = None

    def start(self, on_level: Callable[[float], None]) -> None:
        self._running, self._cb = True, on_level
        self._tick()

    def _tick(self):
        if not self._running: return
        # Simulate quiet->good->loud range
        level = max(0.0, min(0.6, random.uniform(0.01, 0.5)))
        if self._cb: self._cb(level)
        self.root.after(80, self._tick)

    def stop(self) -> None:
        self._running = False
        self._cb = None

    def is_running(self) -> bool:
        return self._running


# ---------------- UI ----------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("STT Notes")
        self.geometry("1080x680")
        self.minsize(940, 580)

        # Data
        self.notes: List[Note] = load_notes()
        self.selected_id: Optional[str] = None

        # Config / dictionary
        self.config_data: dict = load_config()
        self.dictionary: List[str] = load_dictionary()

        # Autosave / edit tracking
        self.loading_note = False
        self.last_edit_ms = 0
        self.last_autosave_ms = 0

        # Recording state (timer/level)
        self.rec_start_ms = 0
        self.elapsed_ms = 0
        self.level_val = 0.0
        self.recording = False  # NEW: replace recorder.is_running()

        # REAL mic input using the backend MicrophoneInput
        noise_gate = float(self.config_data.get("noise_gate_threshold", 0.002) or 0.0)
        self.mic = MicrophoneInput(threshold=noise_gate, sample_rate=RATE)

        # UI state vars
        self.query_var = tk.StringVar()
        self.title_var = tk.StringVar()
        self.tags_var = tk.StringVar()

        # Build UI
        self._build()
        self._bind_shortcuts()
        self._refresh_list()

        # Connect backend transcription to this UI
        global transcript_callback
        def ui_transcript_handler(text: str):
            # Ensure we touch widgets only on the Tk main thread
            self.after(0, lambda: self._append_transcript(text))
        transcript_callback = ui_transcript_handler

        # Start UI ticker
        self._ui_tick()

    # ----- layout -----
    def _build(self):
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # Sidebar
        side = ttk.Frame(self, padding=8)
        side.grid(row=0, column=0, sticky="nsew")
        side.rowconfigure(2, weight=1)

        ttk.Label(side, text="Your Notes", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky="w")
        ent = ttk.Entry(side, textvariable=self.query_var)
        self.search_entry = ent
        ent.grid(row=1, column=0, sticky="ew", pady=(6, 8))
        ent.bind("<KeyRelease>", lambda e: self._refresh_list())

        self.listbox = tk.Listbox(side, activestyle="dotbox")
        self.listbox.grid(row=2, column=0, sticky="nsew")
        self.listbox.bind("<<ListboxSelect>>", self._on_select)
        # Allow Delete key to delete note
        self.listbox.bind("<Delete>", lambda e: self._delete())

        # Delete button under the list
        btn_row = ttk.Frame(side)
        btn_row.grid(row=3, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(btn_row, text="Delete Selected", command=self._delete).pack(side="left")

        # Main
        main = ttk.Frame(self, padding=10)
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(5, weight=1)

        # Controls
        top = ttk.Frame(main)
        top.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        top.columnconfigure(4, weight=1)

        self.start_btn = ttk.Button(top, text="Start Recording", command=self._start)
        self.start_btn.grid(row=0, column=0, padx=2)

        self.stop_btn = ttk.Button(top, text="Stop (00:00)", command=self._stop, state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=2)

        ttk.Button(top, text="Mark ⏱", command=self._mark).grid(row=0, column=2, padx=2)
        ttk.Button(top, text="Save", command=self._save).grid(row=0, column=3, padx=2)

        # Level meter
        meter = ttk.Frame(top)
        meter.grid(row=0, column=4, sticky="e")
        self.level_canvas = tk.Canvas(meter, width=180, height=10, bg="#E5E7EB", highlightthickness=0)
        self.level_canvas.pack(side="left", padx=(0, 6))
        self.level_bar = self.level_canvas.create_rectangle(0, 0, 0, 10, fill="#22C55E", width=0)
        self.level_label = ttk.Label(meter, text="Good")
        self.level_label.pack(side="left")

        # Title / tags
        self.title_entry = ttk.Entry(main, textvariable=self.title_var)
        self.title_entry.grid(row=1, column=0, sticky="ew", pady=(2, 2))
        self.title_entry.insert(0, "Title")

        self.tags_entry = ttk.Entry(main, textvariable=self.tags_var)
        self.tags_entry.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        self.tags_entry.insert(0, "tags, comma, separated")

        # Track edits from title/tags
        self.title_var.trace_add("write", self._on_content_edited)
        self.tags_var.trace_add("write", self._on_content_edited)

        # Text editor
        self.text = tk.Text(main, wrap="word")
        self.text.grid(row=5, column=0, sticky="nsew")
        self.text.bind("<<Modified>>", self._on_text_modified)

        # Export / New / Settings / Status
        foot = ttk.Frame(main)
        foot.grid(row=6, column=0, sticky="ew", pady=8)

        # Status bar (right)
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(foot, textvariable=self.status_var)
        status_label.pack(side="right")

        ttk.Button(foot, text="Settings", command=self._open_settings).pack(side="right", padx=(0, 8))
        ttk.Button(foot, text="Dictionary", command=self._open_dictionary).pack(side="right", padx=(0, 8))

        # New + export (left)
        ttk.Button(foot, text="New Note", command=self._new_note).pack(side="left", padx=(0, 8))
        ttk.Button(foot, text="Export .txt", command=lambda: self._export("txt")).pack(side="left")
        ttk.Button(foot, text="Export .md", command=lambda: self._export("md")).pack(side="left", padx=(6, 0))

    # ----- shortcuts -----
    def _bind_shortcuts(self):
        self.bind_all("<Control-n>", lambda e: self._new_note())
        self.bind_all("<Control-s>", lambda e: self._save())
        self.bind_all("<Control-f>", lambda e: self._focus_search())
        self.bind_all("<Control-d>", lambda e: self._open_dictionary())
        self.bind_all("<Control-comma>", lambda e: self._open_settings())

    def _focus_search(self):
        self.search_entry.focus_set()
        self.search_entry.select_range(0, "end")

    def _set_status(self, msg: str):
        self.status_var.set(msg)

    # ----- recording controls (now using real MicrophoneInput) -----
    def _start(self):
        if self.recording:
            return
        self.rec_start_ms = now_ms()
        self.elapsed_ms = 0

        # sounddevice --> MicrophoneInput.audio_callback --> process_audio()
        # Also feed RMS level into _on_level for the meter
        self.mic.start_stream(callback=process_audio, level_callback=self._on_level)

        self.recording = True
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self._set_status("Recording...")

    def _stop(self):
        if not self.recording:
            return
        self.mic.stop_stream()
        self.elapsed_ms = now_ms() - self.rec_start_ms
        self.recording = False
        self.stop_btn.configure(text=f"Stop ({format_time(0)})")
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self._set_status(f"Stopped. Duration {format_time(self.elapsed_ms)}")

    def _on_level(self, val: float):
        # Called from the mic callback thread; UI-safe update is done in _ui_tick
        self.level_val = max(0.0, float(val))

    def _mark(self):
        elapsed = (now_ms() - self.rec_start_ms) if self.recording else self.elapsed_ms
        self.text.insert("end", f"\n[{format_time(elapsed)}] ")
        self._on_content_edited()

    # ----- insert transcripts into the note -----
    def _append_transcript(self, text: str):
        """Append recognized text into the editor, with spacing."""
        existing = self.text.get("1.0", "end").strip()
        if existing:
            self.text.insert("end", " ")
        self.text.insert("end", text)
        self._on_content_edited()

    # ----- notes list / search -----
    def _filtered(self) -> List[Note]:
        q = self.query_var.get().strip().lower()
        notes = sorted(self.notes, key=lambda n: n.updatedAt, reverse=True)
        if not q:
            return notes
        out = []
        for n in notes:
            if (q in n.title.lower()
                or q in n.content.lower()
                or any(q in t.lower() for t in n.tags)):
                out.append(n)
        return out

    def _refresh_list(self):
        self.listbox.delete(0, "end")
        for n in self._filtered():
            date = time.strftime("%Y-%m-%d", time.localtime(n.createdAt / 1000))
            tags = ", ".join(n.tags) if n.tags else ""
            self.listbox.insert("end", f"{n.title}   • {date}   • {tags}")

    def _selected(self) -> Optional[Note]:
        sel = self.listbox.curselection()
        if not sel:
            return None
        idx = sel[0]
        items = self._filtered()
        if 0 <= idx < len(items):
            return items[idx]
        return None

    def _on_select(self, _evt=None):
        n = self._selected()
        if not n:
            return
        self.loading_note = True
        try:
            self.selected_id = n.id
            self.title_var.set(n.title)
            self.tags_var.set(", ".join(n.tags))
            self.text.delete("1.0", "end")
            self.text.insert("1.0", n.content)
            self.last_edit_ms = 0
        finally:
            self.loading_note = False

    # ----- edit tracking -----
    def _on_text_modified(self, _evt=None):
        if self.loading_note:
            self.text.edit_modified(False)
            return
        self.last_edit_ms = now_ms()
        self.text.edit_modified(False)

    def _on_content_edited(self, *args):
        if self.loading_note:
            return
        self.last_edit_ms = now_ms()

    def _has_meaningful_content(self) -> bool:
        if self.text.get("1.0", "end").strip():
            return True
        if self.title_var.get().strip():
            return True
        if self.tags_var.get().strip():
            return True
        return False

    # ----- new note / delete -----
    def _new_note(self):
        """Prepare UI to create a brand-new note."""
        self.loading_note = True
        try:
            self.selected_id = None  # so _save creates a new one
            self.title_var.set("")
            self.tags_var.set("")
            self.text.delete("1.0", "end")
            # reset elapsed recording duration for this note
            self.elapsed_ms = 0
            self.last_edit_ms = 0
        finally:
            self.loading_note = False
        self._set_status("New note")

    def _delete(self):
        n = self._selected()
        if not n:
            messagebox.showinfo("Delete", "Select a note to delete.")
            return

        if not messagebox.askyesno("Delete note",
                                   f"Delete '{n.title}'?\nThis cannot be undone."):
            return

        # Remove from list and persist
        self.notes = [note for note in self.notes if note.id != n.id]
        save_notes(self.notes)

        # Clear selection and editor
        self.loading_note = True
        try:
            self.selected_id = None
            self.title_var.set("")
            self.tags_var.set("")
            self.text.delete("1.0", "end")
        finally:
            self.loading_note = False

        self._refresh_list()
        self._set_status("Note deleted.")

    # ----- save / export -----
    def _save(self, *, via_autosave: bool = False, show_popup: bool = True):
        if not self._has_meaningful_content() and via_autosave:
            # don't autosave completely empty notes
            return

        title = (self.title_var.get() or "").strip() or f"Note {time.strftime('%Y-%m-%d %H:%M:%S')}"
        tags = [t.strip() for t in self.tags_var.get().split(",") if t.strip()]
        content = self.text.get("1.0", "end").rstrip()
        now_ts = now_ms()

        if self.selected_id:
            # Update existing note
            for i, n in enumerate(self.notes):
                if n.id == self.selected_id:
                    self.notes[i] = Note(
                        id=n.id,
                        title=title,
                        content=content,
                        tags=tags,
                        createdAt=n.createdAt,
                        updatedAt=now_ts,
                        durationMs=n.durationMs or self.elapsed_ms,
                    )
                    break
        else:
            # Create new note
            nid = str(uuid.uuid4())
            self.notes.insert(0, Note(
                id=nid,
                title=title,
                content=content,
                tags=tags,
                createdAt=now_ts,
                updatedAt=now_ts,
                durationMs=self.elapsed_ms,
            ))
            self.selected_id = nid

        save_notes(self.notes)
        # reset edit tracking
        self.last_edit_ms = 0
        self.last_autosave_ms = now_ts

        if show_popup and not via_autosave:
            messagebox.showinfo("Saved", "Note saved.")

        timestamp = time.strftime("%H:%M:%S")
        if via_autosave:
            self._set_status(f"Autosaved at {timestamp}")
        else:
            self._set_status(f"Saved at {timestamp}")

        self._refresh_list()

    def _export(self, kind: str):
        n = self._selected()
        if not n:
            messagebox.showinfo("Export", "Select a note first.")
            return
        default = f"{slugify(n.title)}.{kind}"
        path = filedialog.asksaveasfilename(defaultextension=f".{kind}", initialfile=default)
        if not path:
            return
        Path(path).write_text(n.content, encoding="utf-8")
        messagebox.showinfo("Export", f"Exported to {path}")
        self._set_status("Exported note")

    # ----- Settings / Dictionary -----
    def _open_settings(self):
        if hasattr(self, "_settings_win") and self._settings_win.winfo_exists():
            self._settings_win.lift()
            return

        win = tk.Toplevel(self)
        self._settings_win = win
        win.title("Settings")
        win.resizable(False, False)

        autosave_var = tk.BooleanVar(value=self.config_data.get("autosave_enabled", True))
        delay_var = tk.IntVar(value=self.config_data.get("autosave_delay_ms", 800))
        noise_var = tk.DoubleVar(value=self.config_data.get("noise_gate_threshold", 0.002))
        chunk_var = tk.IntVar(value=self.config_data.get("chunk_ms", 500))
        retries_var = tk.IntVar(value=self.config_data.get("max_retries", 3))
        local_var = tk.BooleanVar(value=self.config_data.get("use_local_model", True))

        row = 0
        ttk.Checkbutton(win, text="Enable autosave", variable=autosave_var).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=8, pady=(8, 4)
        )
        row += 1
        ttk.Label(win, text="Autosave delay (ms):").grid(row=row, column=0, sticky="e", padx=8, pady=4)
        ttk.Entry(win, textvariable=delay_var, width=8).grid(row=row, column=1, sticky="w", padx=8, pady=4)
        row += 1

        # placeholders for backend team
        ttk.Label(win, text="Noise gate threshold:").grid(row=row, column=0, sticky="e", padx=8, pady=4)
        ttk.Entry(win, textvariable=noise_var, width=8).grid(row=row, column=1, sticky="w", padx=8, pady=4)
        row += 1

        ttk.Label(win, text="Chunk length (ms):").grid(row=row, column=0, sticky="e", padx=8, pady=4)
        ttk.Entry(win, textvariable=chunk_var, width=8).grid(row=row, column=1, sticky="w", padx=8, pady=4)
        row += 1

        ttk.Label(win, text="Max retries per chunk:").grid(row=row, column=0, sticky="e", padx=8, pady=4)
        ttk.Entry(win, textvariable=retries_var, width=8).grid(row=row, column=1, sticky="w", padx=8, pady=4)
        row += 1

        ttk.Checkbutton(
            win, text="Prefer local model when available", variable=local_var
        ).grid(row=row, column=0, columnspan=2, sticky="w", padx=8, pady=4)
        row += 1

        def save_and_close():
            self.config_data["autosave_enabled"] = bool(autosave_var.get())
            self.config_data["autosave_delay_ms"] = max(100, int(delay_var.get() or 0))
            self.config_data["noise_gate_threshold"] = float(noise_var.get() or 0.0)
            self.config_data["chunk_ms"] = max(100, int(chunk_var.get() or 0))
            self.config_data["max_retries"] = max(0, int(retries_var.get() or 0))
            self.config_data["use_local_model"] = bool(local_var.get())
            save_config(self.config_data)
            self._set_status("Settings saved")
            win.destroy()

        ttk.Button(win, text="Save", command=save_and_close).grid(
            row=row, column=0, columnspan=2, pady=(8, 8)
        )

    def _open_dictionary(self):
        if hasattr(self, "_dict_win") and self._dict_win.winfo_exists():
            self._dict_win.lift()
            return

        win = tk.Toplevel(self)
        self._dict_win = win
        win.title("Domain Dictionary")
        win.resizable(False, False)

        frame = ttk.Frame(win, padding=8)
        frame.grid(row=0, column=0, sticky="nsew")

        ttk.Label(frame, text="Domain-specific words (one per entry):").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 4)
        )

        listbox = tk.Listbox(frame, height=10)
        listbox.grid(row=1, column=0, columnspan=2, sticky="nsew")
        for w in self.dictionary:
            listbox.insert("end", w)

        entry_var = tk.StringVar()
        entry = ttk.Entry(frame, textvariable=entry_var)
        entry.grid(row=2, column=0, sticky="ew", pady=(4, 4))
        frame.columnconfigure(0, weight=1)

        def add_word():
            w = entry_var.get().strip()
            if not w:
                return
            if w not in self.dictionary:
                self.dictionary.append(w)
                listbox.insert("end", w)
                save_dictionary(self.dictionary)
                self._set_status("Dictionary updated")
            entry_var.set("")

        def delete_selected():
            sel = listbox.curselection()
            if not sel:
                return
            idx = sel[0]
            word = listbox.get(idx)
            self.dictionary = [w for w in self.dictionary if w != word]
            listbox.delete(idx)
            save_dictionary(self.dictionary)
            self._set_status("Dictionary updated")

        ttk.Button(frame, text="Add", command=add_word).grid(
            row=2, column=1, sticky="w", padx=(4, 0)
        )
        ttk.Button(frame, text="Delete Selected", command=delete_selected).grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(4, 0)
        )

    # ----- UI ticker -----
    def _ui_tick(self):
        # meter bar & label
        pct = max(0.0, min(1.0, self.level_val * 1.8))
        self.level_canvas.coords(self.level_bar, 0, 0, int(180 * pct), 10)
        self.level_label.configure(
            text="Too quiet" if self.level_val < 0.03 else ("Too loud" if self.level_val > 0.35 else "Good")
        )
        # elapsed on Stop button while recording
        if self.recording:
            el = now_ms() - self.rec_start_ms
            self.stop_btn.configure(text=f"Stop ({format_time(el)})")

        # autosave check
        now_ts = now_ms()
        if (
            self.config_data.get("autosave_enabled", True)
            and self.last_edit_ms
            and self.last_edit_ms > self.last_autosave_ms
            and now_ts - self.last_edit_ms >= self.config_data.get("autosave_delay_ms", 800)
        ):
            self._save(via_autosave=True, show_popup=False)

        self.after(100, self._ui_tick)


if __name__ == "__main__":
    App().mainloop()