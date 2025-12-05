# FinalProject
Speech-to-Text Notes App
Authors: Cassidy Sakamoto · Justin Glabicki · Nathan Tan
Instructor: Dr. Fatema Nafa
Date: 12/5/2025

Overview:
The Speech-to-Text Notes App is an end-to-end Python application that captures microphone  audio in real time, transcribes it into text, and integrates the results into a full-featured note-taking interface. Built with a modular architecture separating audio capture, speech  recognition, and a Tkinter- based user interface, the app provides users with an accessible  tool for hands-free note creation and organization.
The project uses the sounddevice library for raw audio streaming, speech_recognition (Google  Web Speech API) for transcription, and Python’s built-in tkinter for the interactive note- editing environment. Notes are saved locally in JSON files, and users may export them in .txt or .md format.
This README reflects the final integrated implementation (Mic -> Transcription Pipeline -> UI/Notes System).

Goal of the Project:
The goal of this project is to build a complete Speech-to-Text Notes Application that allows users to:
- Record real-time microphone audio
- Automatically transcribe speech into text
- Edit, search, and organize notes
- Export notes in .txt or .md format

Features:
Real-Time Audio Capture
  - Uses sounddevice.InputStream to read audio in continuous 20 ms frames
  - Computes RMS loudness levels to drive a live mic level meter in the UI.
  - Adjustable noise-gate threshold (via Settings panel).
  - Modular callback design allows the UI to attach audio handlers.
Speech Transcription Pipeline
  - Frames (~6 seconds) of PCM audio are automatically assembled and sent for transcription.
  - Transcription performed using speech_recognition's Google Web Speech API.
  - Retry logic for network/API failures.
  - Automatic domain-specific text normalization:
    - Replaces phrases like "r c" -> "RC", "k h z" -> "kHz", etc.
  - Clean capitalized, punctuation-normalized sentences.
  - Transcript chunks are fowarded to the UI via a thread-safe callback.
Full Notes Management System
  - Create, edit, delete, and search notes by:
    - Title
    - Tags
    - Content keywords
  - Notes stored in notes.json with metadata:
    - Title
    - Content
    - Tags
    - Created / Updated timestamps
    - Recording Duration
Recording Tools
  - Start / Stop microphone capture with visible duration timer.
  - Insert timestamp markers ( [mm:ss] ) into text while recording.
  - Autosave system prevents data loss (configurable delay, default 800 ms).
Local Storage + Export
  - All notes stored locally in:
      stt_ui_data/
        notes.json
        config.json
        dictionary.json
  - Export options:
    - .txt
    - .md
  - Automatically slugifies titles to create clean filenames.
Customization Settings
  - Enable / disable autosave.
  - Adjust autosave delay, chunk size, noise gate threshold, retry limit, etc.
  - Optional domain dictionary:
      - Add / remove custom words used in your field.
      - Stored in dictionary.json.
Tkinter User Interface
  - Search bar and live-updating notes list.
  - Main editor pane for note content.
  - Status bar with autosave notifications.
  - Mic level bar and status ("Good", "Too quiet", "Too loud").
  - Keyboard shortcuts (Ctrl+S, Ctrl+N, Ctrl+F, etc.).

How to Run Application
1. Install Dependencies
Python 3.10+ recommended.
  pip install sounddevice speechrecognition numpy
Aditionally, your system must support Google Web Speech API (requires internet unless using a local model extension).
2. Run the Full Application
  python Speech-to-Text_Notes_app.py
  If you experience errors starting the mic or obtaining transcripts, try the secondary App    version:
  python UI2.py
This launches the Tkinter UI and prepares the microphone backend. The file can also be launched in an IDE of choice such as Spyder.
4. Using the UI
  - Click Start Recording
  - Speak into your microphone
  - View live mic levels as you talk
  - Transcribed text automatically inserts into your note
  - Use the Mark button to insert timestamps
  - Manage notes on the left sidebar
  - Export notes from the footer toolbar
If transcription does not work:
  - Your noise gate may be set too high.
  - Set it to a low value such as 0-0.002 in: Settings -> Noise Gate Threshold

Requirements & Dependencies
- Python 3.10+
- sounddevice - real-time audio capture
- speech_recognition - Google Web API transcription
- numpy - audio array operations
- tkinter - graphical user interface (built into Python)
- Local storage JSON files:
  - notes.json
  - config.json
  - dictionary.json
Data is stored automatically in stt_ui_data/

Explanation of Approach / Methodology
Architecture Overview
The system is build around three major components:
1. Microphone Audio Module
   - Streams ~20 ms audio frames using sounddevice.InputStream
   - Computes RMS level -> drives mic-level meter
   - Noise-gate suppresses silent frames
   - Forwards audio to the transcription pipeline
2. Transcription Pipeline
   - Combines ~6 seconds of audio into chunks
   - Sends audio to Google's Speech Recognition API
   - Retries failures up to max_retries
   - Applies domain-specific corrections (e.g., "k h z" -> "kHz")
   - Sends normalized text back to the UI thread
3. Tkinter Notes UI
   - Full notes editor (create, edit, delete, search)
   - Autosave system (default 800 ms delay)
   - Timestamp markers during recording
   - Export to .txt or .md
   - Editable domain dictionary for technical vocabulary
The system uses thread-safe callbacks to avoid blocking real-time audio capture or UI updates.

Results, Outputs, and Demo Screenshots
Noise Gate Tests
<img width="2156" height="500" alt="ThresholdHigh" src="https://github.com/user-attachments/assets/37a05093-89ec-4d8b-ba0b-97091e826753" />


Known Limitations / Future Improvements
- Google Web Speech API requires internet; offline transcription model optional but not yet implemented
- Large audio chunks (~6 seconds) may introduce slight delays in long recordings.
- Device index for microphone (device=9) may require local adjustment.
Possible future extensions:
- Whisper-based local transcription
- Audio playback for recorded segments
- Multi-note merging and tagging UI improvements
- Real-time word-by-word transcript preview

Credits
Project developed collaboratively by:
Cassidy Sakamoto · Justin Glabicki · Nathan Tan

Special thanks to Dr. Fatema Nafa for course guidance and support.
