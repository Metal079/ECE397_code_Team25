import pyaudio
import wave
import datetime
from array import array

FORMAT = pyaudio.paInt16
CHANNELS = 2           # Number of channels
BITRATE = 44100        # Audio Bitrate
CHUNK_SIZE = 1024      # Chunk size to 
RECORDING_LENGTH = 10  # Recording Length in seconds

#Instantiate Pyaudio
audio = pyaudio.PyAudio()

#Select SEEED sound card
device_id = 2
print("Recording using Input Device ID "+str(device_id))

#Recording Prereqs/Open stream
stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=BITRATE,
        input=True,
        input_device_index = device_id,
        frames_per_buffer=CHUNK_SIZE
    )

#Recording block

#Recording data array
recording_frames = []

#Detect and record
#for i in range(0, int(BITRATE / CHUNK_SIZE * RECORDING_LENGTH)):
while True:
    data = stream.read(CHUNK_SIZE)
    data_chunk=array('h',data)
    vol=max(data_chunk)
    if(vol>=100):
        now = datetime.datetime.now()
        print("something said")
        print("\n")
        print(str(now))
        print("\n")
        recording_frames.append(data)
    else:
        print("nothing")
        print("\n")

#Stop stream/End recording
stream.stop_stream()
stream.close()
audio.terminate()

#Write to .wav file
WAVE_OUTPUT_FILENAME = str(now) + ".wav"
waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(FORMAT))
waveFile.setframerate(BITRATE)
waveFile.writeframes(b''.join(recording_frames))
waveFile.close()

    



    

