import alsaaudio   
import numpy as np
import array
import wave
import datetime
import audioop
import time
import numpy
import board
import adafruit_dotstar
from math import log10


# Variables
TYPE        = alsaaudio.PCM_CAPTURE
MODE        = alsaaudio.PCM_NONBLOCK
CHANNELS    = 2              # Set # of Channels (1 = Mono, 2 = Stereo)
BITRATE     = 44100          # Audio Bitrate
CHUNK_SIZE  = 1024           # Sampling Size
RECORDING_LENGTH = 5         # Sets recording length
db = 0                   # Set a decibel threshold
DOTSTAR_DATA = board.D5  # Initialize data pin
DOTSTAR_CLOCK = board.D6 # Initialize clock pin
recording_frames = []    # Recording data array


#Instantiate Adafruit Voice Bonnet LEDS
#adafruit_dotstar.Dotstar(clk pin, data pin, # of LEDS [0 to n]/Left to Right, brightness)
#Color scheme:
#Green (255, 0, 0)
#Blue (0, 255, 0)
#Red (0, 0, 255)
dots = adafruit_dotstar.DotStar(DOTSTAR_CLOCK, DOTSTAR_DATA, 3, brightness=0.1)


#Set LEDS to red
def redDots():
    dots[0] = (0, 0, 255)
    dots[1] = (0, 0, 255)
    dots[2] = (0, 0, 255)
    dots.show()


#Set LEDS to blue
def blueDots():
    dots[0] = (0, 255, 0)
    dots[1] = (0, 255, 0)
    dots[2] = (0, 255, 0)
    dots.show()


#Set LEDS to green
def greenDots():
    dots[0] = (255, 0, 0)
    dots[1] = (255, 0, 0)
    dots[2] = (255, 0, 0)
    dots.show()


# set up audio input
stream=alsaaudio.PCM(type=TYPE, mode=MODE)


# Write to .wav file in /home/collab/Desktop/Audio DB/
def waveOutput():
    blueDots()
    print("Writing to Audio DB")
    print("\n")
    now = datetime.datetime.now()
    print("Started writing at " + str(now))
    print("\n")
    WAVE_OUTPUT_FILENAME = "/home/collab/Desktop/Audio DB/" + str(now) + ".wav"
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(2)
    waveFile.setframerate(BITRATE)
    waveFile.writeframes(b''.join(recording_frames))
    waveFile.close()
    now = datetime.datetime.now()
    print("Finished writing at " + str(now))
    print("\n")
    

# Recording incoming audio stream for 2 seconds
def record():
    redDots()
    for i in range(0,int(BITRATE/CHUNK_SIZE*RECORDING_LENGTH)):
        now = datetime.datetime.now()
        print("Recording at " + str(now))
        print("\n")
        data=stream.read()[1]      
        recording_frames.append(data)
        

# Indefinite detection function
def detect():
    print("Listening!")
    print("\n")
    greenDots()
    data = stream.read()[1]
    rms = audioop.rms(data, 2)
    db = 20 * log10(rms)
    print("Current dB level: " + str(int(db)))
    print("\n")
    # Threshold detection; On break, begin recording
    if(db>=50):
        now = datetime.datetime.now()
        redDots()
        print("Sound Detected at " + str(now))
        print("\n")
        record()
    else:
        print("Nothing Detected...")
        print("\n")


#detect()
#record()
waveOutput()


#data = np.array(buffer, dtype='f')