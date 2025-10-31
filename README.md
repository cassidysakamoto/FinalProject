# FinalProject
Speech-to-Text Notes App (Iteration #03)
Authors: Cassidy Sakamoto · Justin Glabicki · Nathan Tan
Instructor: Dr. Fatema Nafa
WORK IN PROGRESS

Overview:
The Speech-to-Text Notes App is a Python-based project that records audio, converts it to text in real time, and provides a simple UI for editing, searching, and exporting notes.
It is implemented entirely in Python using sounddevice, speech_recognition, and tkinter, with modular separation between audio capture, transcription, and UI/notes storage.

Features:
Audio Input - Captures live microphone audio using sounddevice; applies noise-gate filtering; displays real-time mic-level meter.
Transcription - Real-time chunked transcription via speech_recognition. Includes retry and offline modes.
Notes Storage - Saves transcribed notes to notes.json with metadata (title, tags, timestamps, duration).
Search & Edit - Search by keyword, phrase, or tag; edit and autosave content; mark timestamps in text.
Export - Export any note to .txt or .md including metadata headers.
Error Handling - Handles missing mic input, network failures, and permission errors gracefully.

How to run:
UI Mode: 
  Download UI.py
  Run using an IDE
  Start/Stop Recording, save notes, and export notes using buttons on UI

Microphone Module Test:
  Download Project_FirstDraft.py
  Run using an IDE
  Displays live mic-level meter and demonstrates noise-gate filtering

Real-time Transcription Pipeline:
  Download final_project.py
  Run using an IDE
  Performs real-time speech-to-text, printing partial transcriptions and handling retry logic automatically
