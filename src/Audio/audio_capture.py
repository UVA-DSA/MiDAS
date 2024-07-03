import pyaudio
import wave

# Parameters for audio recording
FORMAT = pyaudio.paInt16  # Audio format (16-bit)
CHANNELS = 1              # Number of audio channels (1 for mono, 2 for stereo)
RATE = 44100              # Sampling rate (samples per second)
CHUNK = 1024              # Number of frames per buffer
RECORD_SECONDS = 5        # Duration of recording
OUTPUT_FILENAME = "output.wav"  # Output file name


def capture_audio_transmit(audio_q, data_path):
    # Initialize PyAudio
    audio = pyaudio.PyAudio()

    # Open a new stream for audio input
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    print("Recording...")

    OUTPUT_FILENAME = data_path + "/surgeon_evaluation.wav"

    # Open the wave file in write mode
    wf = wave.open(OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)

    while True:
        try:
            data = stream.read(CHUNK)
            wf.writeframes(data)
            audio_q.put(data)
            
        except KeyboardInterrupt:
            print("Recording stopped by user")

            # Stop and close the stream
            stream.stop_stream()
            stream.close()
            audio.terminate()

            # Close the wave file
            wf.close()

            print("Recording saved to", OUTPUT_FILENAME)
            break
    
    
# test the above function

from queue import Queue   
import os
import time

audio_q = Queue()

data_path = os.path.join(os.getcwd(), "data")

if not os.path.exists(data_path):
    os.makedirs(data_path)
    
capture_audio_transmit(audio_q, data_path)