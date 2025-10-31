# Real-time streaming audio-to-text with retry logic
# Install required packages: pip install SpeechRecognition pyaudio

import speech_recognition as sr
import time
import struct

RATE = 16000
FRAME_MS = 20
FRAME_SAMPLES = 320
CHUNK_FRAMES = 100
ENERGY_MIN = 200
ENERGY_MAX = 28000
RETRY_LIMIT = 3

# Initialize speech recognizer
recognizer = sr.Recognizer()
recognizer.energy_threshold = ENERGY_MIN
recognizer.dynamic_energy_threshold = True

# Streaming state
buffer_frames = []
outgoing_chunks = []
partials = []
messages = []
too_quiet = False
too_loud = False

# Network simulation state
pending = []
retry_count = {}
sent = []
network_up = True  # Set to False to test offline mode

print("=== REAL-TIME AUDIO TO TEXT ===")
print("Starting microphone... Speak clearly into your microphone.")
print("Press Ctrl+C to stop.\n")

try:
    with sr.Microphone(sample_rate=RATE) as source:
        print("Adjusting for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Ready! Start speaking...\n")
        
        chunk_counter = 0
        
        while True:
            try:
                # Listen for audio with short timeout for responsiveness
                audio = recognizer.listen(source, timeout=1, phrase_time_limit=2)
                
                # Get raw audio data
                raw_data = audio.get_raw_data()
                
                # Convert bytes to 16-bit integers for energy calculation
                samples = []
                for i in range(0, len(raw_data), 2):
                    if i + 1 < len(raw_data):
                        sample = struct.unpack('<h', raw_data[i:i+2])[0]
                        samples.append(sample)
                
                # Calculate average energy
                if len(samples) > 0:
                    s = sum(abs(sample) for sample in samples)
                    avg = s // len(samples)
                else:
                    avg = 0
                
                # Check audio levels
                if avg < ENERGY_MIN:
                    if not too_quiet:
                        messages.append("Mic: too quiet")
                        print("⚠ Mic: too quiet")
                    too_quiet = True
                else:
                    too_quiet = False
                
                if avg > ENERGY_MAX:
                    if not too_loud:
                        messages.append("Mic: clipping")
                        print("⚠ Mic: clipping")
                    too_loud = True
                else:
                    too_loud = False
                
                # Only process if sufficient energy
                if avg >= ENERGY_MIN:
                    buffer_frames.append(raw_data)
                    
                    # When we have enough frames, create a chunk and transcribe
                    if len(buffer_frames) >= 5:  # Reduced for faster response
                        # Combine frames into a single chunk
                        combined_audio = b''.join(buffer_frames)
                        
                        # Create AudioData object for recognition
                        audio_chunk = sr.AudioData(combined_audio, RATE, 2)
                        
                        outgoing_chunks.append(audio_chunk)
                        buffer_frames = []
                        
                        # Add to pending queue
                        chunk_id = chunk_counter
                        pending.append({"id": chunk_id, "data": audio_chunk})
                        retry_count[chunk_id] = 0
                        chunk_counter += 1
                        
                        # Try to transcribe (simulating network send)
                        if network_up:
                            idx = 0
                            while idx < len(pending):
                                item = pending[idx]
                                cid = item["id"]
                                
                                # Simulate occasional network failures (every 5th chunk fails first attempt)
                                will_fail = ((cid % 5) == 0) and (retry_count[cid] == 0)
                                
                                if not will_fail:
                                    try:
                                        # ACTUAL TRANSCRIPTION HERE
                                        text = recognizer.recognize_google(item["data"])
                                        
                                        if text:
                                            partials.append(text)
                                            print(f"✓ Transcribed: {text}")
                                            
                                        sent.append(item)
                                        pending.pop(idx)
                                        
                                    except sr.UnknownValueError:
                                        print("⚠ Could not understand audio")
                                        pending.pop(idx)
                                    except sr.RequestError as e:
                                        print(f"⚠ API error: {e}")
                                        # Retry this chunk
                                        tries = retry_count[cid]
                                        if tries < RETRY_LIMIT:
                                            retry_count[cid] = tries + 1
                                            pending.append(pending.pop(idx))
                                        else:
                                            messages.append(f"Failed to send chunk {cid}")
                                            print(f"✗ Failed to send chunk {cid}")
                                            pending.pop(idx)
                                else:
                                    # Simulate retry
                                    retry_count[cid] += 1
                                    pending.append(pending.pop(idx))
                        else:
                            if len(messages) == 0 or messages[-1] != "Offline: storing chunks":
                                messages.append("Offline: storing chunks")
                                print("⚠ Offline: storing chunks")
                
            except sr.WaitTimeoutError:
                # No speech detected in timeout period, continue listening
                continue
            except Exception as e:
                print(f"Error: {e}")
                continue

except KeyboardInterrupt:
    print("\n\n=== SESSION SUMMARY ===")
    print(f"\n=== PARTIAL UPDATES ({len(partials)} total) ===")
    for i, partial in enumerate(partials):
        print(f"{i + 1}) {partial}")
    
    print(f"\n=== STATUS MESSAGES ({len(messages)} total) ===")
    for msg in messages:
        print(f"- {msg}")
    
    print("\n=== SEND SUMMARY ===")
    print(f"Chunks sent: {len(sent)}")
    print("Retries used per chunk (id:count):")
    for cid in sorted(retry_count.keys()):
        print(f"{cid}:{retry_count[cid]}", end="  ")
    print(f"\nPending after flush: {len(pending)}")
    
    print("\n=== FULL TRANSCRIPT ===")
    full_transcript = " ".join(partials)
    print(full_transcript if full_transcript else "(no speech detected)")
    
    print("\nSession ended.")