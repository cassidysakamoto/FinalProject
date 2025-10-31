# stt_notes_ui_only.py
# UI-focused Speech-to-Text Notes (Tkinter)
# - Start/Stop, elapsed timer, marker, notes list, search, tags, export

import json, time, uuid, random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Callable, Optional
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ---------------- Basics ----------------
DATA_DIR = Path.cwd() / "stt_ui_data"
DATA_DIR.mkdir(exist_ok=True)
NOTES_FILE = DATA_DIR / "notes.json"

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

# ---------------- Recorder adapter ----------------
class RecorderAdapter:
    """
    Minimal interface your audio team can implement and pass into the UI:
      - start(on_level: Callable[[float], None]) -> None
      - stop() -> None
      - is_running() -> bool
    on_level should be called with RMS-like value in [0, ~0.7] at ~10–20 Hz.
    """
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
    def __init__(self, recorder: RecorderAdapter | None = None):
        super().__init__()
        self.title("STT Notes (UI only)")
        self.geometry("1080x680")
        self.minsize(940, 580)

        # Data
        self.notes: List[Note] = load_notes()
        self.selected_id: Optional[str] = None

        # Recording state (timer/level only; no audio)
        self.recorder = recorder or DummyRecorder(self)
        self.rec_start_ms = 0
        self.elapsed_ms = 0
        self.level_val = 0.0

        # UI state vars
        self.query_var = tk.StringVar()
        self.title_var = tk.StringVar()
        self.tags_var = tk.StringVar()

        self._build()
        self._refresh_list()
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
        ent.grid(row=1, column=0, sticky="ew", pady=(6, 8))
        ent.bind("<KeyRelease>", lambda e: self._refresh_list())

        self.listbox = tk.Listbox(side, activestyle="dotbox")
        self.listbox.grid(row=2, column=0, sticky="nsew")
        self.listbox.bind("<<ListboxSelect>>", self._on_select)

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

        # Text editor
        self.text = tk.Text(main, wrap="word")
        self.text.grid(row=5, column=0, sticky="nsew")

        # Export
        foot = ttk.Frame(main)
        foot.grid(row=6, column=0, sticky="ew", pady=8)
        ttk.Button(foot, text="Export .txt", command=lambda: self._export("txt")).pack(side="left")
        ttk.Button(foot, text="Export .md", command=lambda: self._export("md")).pack(side="left", padx=(6, 0))

    # ----- recording controls (UI only) -----
    def _start(self):
        if self.recorder.is_running():
            return
        self.rec_start_ms = now_ms()
        self.elapsed_ms = 0
        self.recorder.start(self._on_level)
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")

    def _stop(self):
        if not self.recorder.is_running():
            return
        self.recorder.stop()
        self.elapsed_ms = now_ms() - self.rec_start_ms
        self.stop_btn.configure(text=f"Stop ({format_time(0)})")
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")

    def _on_level(self, val: float):
        self.level_val = max(0.0, float(val))

    def _mark(self):
        elapsed = (now_ms() - self.rec_start_ms) if self.recorder.is_running() else self.elapsed_ms
        self.text.insert("end", f"\n[{format_time(elapsed)}] ")

    # ----- notes list / search -----
    def _filtered(self) -> List[Note]:
        q = self.query_var.get().strip().lower()
        notes = sorted(self.notes, key=lambda n: n.updatedAt, reverse=True)
        if not q: return notes
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
            date = time.strftime("%Y-%m-%d", time.localtime(n.createdAt/1000))
            tags = ", ".join(n.tags) if n.tags else ""
            self.listbox.insert("end", f"{n.title}   • {date}   • {tags}")

    def _selected(self) -> Optional[Note]:
        sel = self.listbox.curselection()
        if not sel: return None
        idx = sel[0]
        items = self._filtered()
        if 0 <= idx < len(items): return items[idx]
        return None

    def _on_select(self, _evt=None):
        n = self._selected()
        if not n: return
        self.selected_id = n.id
        self.title_var.set(n.title)
        self.tags_var.set(", ".join(n.tags))
        self.text.delete("1.0", "end")
        self.text.insert("1.0", n.content)

    # ----- save / export -----
    def _save(self):
        title = (self.title_var.get() or "").strip() or f"Note {time.strftime('%Y-%m-%d %H:%M:%S')}"
        tags = [t.strip() for t in self.tags_var.get().split(",") if t.strip()]
        content = self.text.get("1.0", "end").rstrip()
        now = now_ms()

        if self.selected_id:
            for i, n in enumerate(self.notes):
                if n.id == self.selected_id:
                    self.notes[i] = Note(
                        id=n.id, title=title, content=content, tags=tags,
                        createdAt=n.createdAt, updatedAt=now,
                        durationMs=n.durationMs or self.elapsed_ms
                    )
                    break
        else:
            nid = str(uuid.uuid4())
            self.notes.insert(0, Note(
                id=nid, title=title, content=content, tags=tags,
                createdAt=now, updatedAt=now, durationMs=self.elapsed_ms
            ))
            self.selected_id = nid

        save_notes(self.notes)
        self._refresh_list()
        messagebox.showinfo("Saved", "Note saved.")

    def _export(self, kind: str):
        n = self._selected()
        if not n:
            messagebox.showinfo("Export", "Select a note first.")
            return
        default = f"{slugify(n.title)}.{kind}"
        path = filedialog.asksaveasfilename(defaultextension=f".{kind}", initialfile=default)
        if not path: return
        Path(path).write_text(n.content, encoding="utf-8")
        messagebox.showinfo("Export", f"Exported to {path}")

    # ----- UI ticker -----
    def _ui_tick(self):
        # meter bar & label
        pct = max(0.0, min(1.0, self.level_val * 1.8))
        self.level_canvas.coords(self.level_bar, 0, 0, int(180 * pct), 10)
        self.level_label.configure(
            text="Too quiet" if self.level_val < 0.03 else ("Too loud" if self.level_val > 0.35 else "Good")
        )
        # elapsed on Stop button while recording
        if self.recorder.is_running():
            el = now_ms() - self.rec_start_ms
            self.stop_btn.configure(text=f"Stop ({format_time(el)})")
        self.after(100, self._ui_tick)

if __name__ == "__main__":
    App().mainloop()