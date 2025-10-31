# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 13:43:58 2025

@author: cassi
"""


#Use sounddevice library - live audio recording with NumPy arrays
import sounddevice as sd

#Use numpy library - audio processing, needed for mic level
import numpy as np


#Class for microphone input- reads audio in chunks
class MicrophoneInput:
    
    def __init__(self, threshold=0.02):
        self.threshold = threshold   # noise gate threshold
        self.stream = None    #initialize stream as none (stream is the mic stream)
        self.callback = None   #initialize callback (audio_callback) as none

   
    #processing audio input, called every time audio comes in from mic
    #indata is the audio chunk, frames are # of frames in chunk
    def audio_callback(self, indata, frames, time, status):
       
        #Check for errors
        if status:
            print(f"Stream status: {status}")

        audio_chunk = indata.flatten()  #convert audio into 1D NumPy array

        # Noise Gate:
        
        #root mean square = mean of NumPy array (each value is squared to turn them all pos.) 
        #average volume
        rms = np.sqrt(np.mean(audio_chunk**2))  #loudness level
       
       #If volume is lower than threshold...
        if rms < self.threshold:
            return  #ignore quiet noise

        

print(f"Audio chunk received (volume={rms:.3f})")

     #Mic Level Meter (visual feedback)
        bar = "#" * int(rms * 50)  # volume bar using #
        print(f"Mic Level: {bar} ({rms:.3f})")

        #Send audio to transcriber  **connect code
        if self.callback:
            self.callback(audio_chunk)  #pass chunk to another function


#Start listening to mic
    def start_stream(self, callback=None):
        
        #save callback
        self.callback = callback
        
        #Importing from the sounddevice library
        #InputStream is a class in sd library - allows an audio input stream
        self.stream = sd.InputStream(
            channels=1,  #1 channel: Mono audio input
           
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




#Buffer -- stores recent chunks of audio to send as larger batch to transcription part of code
class AudioBuffer:
    def __init__(self, max_chunks=10):
        #Stores audio chunks in a list
        self.buffer = []           
       
        #10 is the max chunks of audio to hold at a time
        self.max_chunks = max_chunks

    def add_chunk(self, chunk):
        #add chunk to list
        self.buffer.append(chunk)
        
        #if the length of the list is greater than the max...
        if len(self.buffer) > self.max_chunks:
            # remove the first item (oldest recorded audio)
            self.buffer.pop(0)  

    # Create one long combined list of all the audio chunks lists in buffer
    def get_audio(self):
       
        if not self.buffer:
            return None
        #Create empty list
        combined = []
        for chunk in self.buffer:
            #Add each chunk to the list
            combined.extend(chunk)
        #Clear the buffer list
        self.buffer = []
        return combined
    

#----------Testing-------------------------------
if __name__ == "__main__":
    mic = MicrophoneInput(threshold=0.02)  #create microphone object
    buffer = AudioBuffer(max_chunks=5)    #only stores 5 chunks of audio at a time

#callback function
def process_audio(chunk):
    #calling add_chunk function in AudioBuffer class
    buffer.add_chunk(chunk)   #the chunked audio is stored here

'''
can pass process_audio function into MicrophoneInput class:
    
mic = MicrophoneInput(callback=process_audio)
    
'''
    

mic.start_stream(process_audio)

#When user presses enter, input function is finished and mic stream stops
input("Press Enter to stop...\n")

mic.stop_stream()







# Test callback function
def test_callback(chunk):
    print(f" got audio chunk of size {len(chunk)}")

# Start the mic and use the test callback
mic = MicrophoneInput()
mic.start_stream(callback=test_callback)

input("Press Enter to stop...\n")
mic.stop_stream()




