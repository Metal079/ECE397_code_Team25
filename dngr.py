import pyaudio
import wave
import datetime
import audioop
import time
import signal
import board
import adafruit_dotstar
from digitalio import DigitalInOut, Direction, Pull
from math import log10
from array import array


# Variables
FORMAT = pyaudio.paInt16
CHANNELS = 2             # Number of channels
BITRATE = 44100          # Audio Bitrate
CHUNK_SIZE = 1024        # Chunk size to 
RECORDING_LENGTH = 2     # Recording Length in seconds
db = 0                   # Set a decibel threshold
DOTSTAR_DATA = board.D5  # Initialize data pin
DOTSTAR_CLOCK = board.D6 # Initialize clock pin
recording_frames = []    # Recording data array


##################################
#####Graceful Exit Class Test#####
##################################
class GracefulExiter():

    def __init__(self):
        self.state = False
        signal.signal(signal.SIGINT, self.change_state)

    def change_state(self, signum, frame):
        print("exit flag set to True (repeat to exit now)")
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        self.state = True

    def exit(self):
        return self.state
    
##################################
##################################


#Setup Adafruit Voice Bonnet button(Not in use)
button = DigitalInOut(board.D17)
button.direction = Direction.INPUT
button.pull = Pull.UP

    
# Instantiate Pyaudio
audio = pyaudio.PyAudio()


#Instantiate Adafruit Voice Bonnet LEDS
#adafruit_dotstar.Dotstar(clk pin, data pin, # of LEDS [0 to n]/Left to Right, brightness)
#Color scheme:
#Green (255, 0, 0)
#Blue (0, 255, 0)
#Red (0, 0, 255)
dots = adafruit_dotstar.DotStar(DOTSTAR_CLOCK, DOTSTAR_DATA, 3, brightness=0.1)


# Recording Prereqs/Open stream
stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=BITRATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE
    )


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
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
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
        data=stream.read(CHUNK_SIZE, exception_on_overflow=False)      
        recording_frames.append(data)

        
# Indefinite detection function
def detect():
    while True:
        print("Listening!")
        print("\n")
        greenDots()
        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        rms = audioop.rms(data, 2)
        db = 20 * log10(rms)
        print("Current dB level: " + str(int(db)))
        print("\n")
        # Threshold detection; On break, begin recording
        if(db>=80):
            now = datetime.datetime.now()
            redDots()
            print("Sound Detected at " + str(now))
            print("\n")
            break
        else:
            print("Nothing Detected...")
            print("\n")
            

flag = GracefulExiter()
# Main loop to detect, record, and output a .wav file            
while True:
    detect()
    record()
    waveOutput()
    recording_frames = [] # Resets the array
    ###Graceful Exit Test###
    if flag.exit():
        stream.stop_stream()
        stream.close()
        audio.terminate()
        dots.deinit()
        break




