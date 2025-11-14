#Use sounddevice library - live audio recording with NumPy arrays
#has callback as a built-in feature
import sounddevice as sd
#print(sd.query_devices())  # Lists all audio devices with their IDs
#print(sd.default.device)   # Shows default input/output device IDs

#print(sd.query_devices(9, 'input'))


#Use numpy library - audio processing, needed for mic level
import numpy as np

#Other imports for transcription
import time, struct, sys, threading
import speech_recognition as sr


#===========================================================
#Class for microphone input- reads audio in chunks
#===========================================================
class MicrophoneInput:
    
    def __init__(self, threshold=0.00001, sample_rate=48000, chunk_dur=0.02):
        self.threshold = threshold   # noise gate threshold
        self.rate = sample_rate      # audio sample rate
        self.chunk_dur = chunk_dur   # duration (s) of each audio chunk
        self.chunk_size = int(self.rate * chunk_dur)  # samples per chunk
        self.stream = None    #initialize stream as none (stream is the mic stream)
        self.callback = None   #initialize callback (audio_callback) as none

   
    #processing audio input, called every time audio comes in from mic
    #indata is the audio chunk, frames are # of frames in chunk
    def audio_callback(self, indata, frames, time_info, status):
       
        #Check for errors
        if status:
            print(f"Stream status: {status}")
        print("audio_callback called")

        audio_chunk = indata.flatten()  #convert audio into 1D NumPy array

        # Noise Gate:
        #root mean square = mean of NumPy array (each value is squared to turn them all pos.) 
        #average volume
        rms = np.sqrt(np.mean(audio_chunk**2))  #loudness level
       
        #Mic Level Meter (visual feedback)
        bar = "#" * int(rms * 50)  # volume bar using #
        print(f"Mic Level: {bar} ({rms:.3f})")

        #If volume is lower than threshold...
        if rms < self.threshold:
            return  #ignore quiet noise

        #Send audio to transcriber  **connect code
        
        if self.callback:
            
            #Calls process_audio()!
            self.callback(audio_chunk)  #pass chunk to another function


    #Start listening to mic
    def start_stream(self, callback=None):
        
        #save callback
        self.callback = callback
        
        #Importing from the sounddevice library
        #InputStream is a class in sd library - allows an audio input stream
        self.stream = sd.InputStream(
            device=10,    # Device ID
            channels=1,  #1 channel: Mono audio input
            samplerate=self.rate,  #sample rate for audio
            #telling sd library to call audio_callback whenever it recieves audio
            callback=self.audio_callback  #callback is set equal to the audio_callback function
        )
        #Starting audio stream- calling function built into sounddevice library
        self.stream.start()
        print("Microphone stream started")

    def stop_stream(self):
        
        if self.stream:
            self.stream.stop()
            self.stream.close()
            print("Microphone stream stopped")




#===========================================================
#Part 2: transcription (kept mostly identical)
#===========================================================

# Audio and framing
RATE = 48000
FRAME_MS = 20
SAMPLES_PER_FRAME = int(RATE * FRAME_MS / 1000)     # 960 samples per 20 ms
BYTES_PER_SAMPLE = 2
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


#-----------------------------------------------------------
#Helper functions for processing + recognition
#-----------------------------------------------------------

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


#-----------------------------------------------------------
#Integration between mic stream and transcription
#-----------------------------------------------------------

frames = []          # will hold individual short chunks (~20ms)
chunk_id = 0
partials = []        # partial transcripts
t_chunk_start = time.time()


#Runs every time mic produces a chunk of audio

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
            args=(raw, chunk_id, t_chunk_start, t_chunk_end)
        ).start()

        chunk_id += 1
        t_chunk_start = time.time()


def transcribe_chunk(raw, chunk_id, t_start, t_end):
    """Performs transcription with retry logic"""
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

    # Print results using your original formatting
    timing = f"[chunk {chunk_id} {t_start:.3f}-{t_end:.3f}]"
    if text:
        out = normalize_text(text)
        partials.append(out)
        print(f"{timing} {out}")
    else:
        print(f"{timing} (no text, {err})")



#===========================================================
#Main execution
#===========================================================

if __name__ == "__main__":
    mic = MicrophoneInput(threshold=0.00001, sample_rate=48000)
    
    #self.callback = process_audio ---> processes audio from chunk: 
        #from process_audio(audio_chunk) in audio_callback function
   
    # sounddevice --> audio_callback --> process_audio function
    mic.start_stream(callback=process_audio)

    try:
        input("Press Enter to stop...\n")
    finally:
        mic.stop_stream()
        print("\n=== Summary ===")
        print(f"Chunks processed: {chunk_id}")
        full = " ".join(partials)
        print(f"Transcript: {full if full else '(none)'}")