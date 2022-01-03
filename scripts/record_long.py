import sounddevice as sd
from playsound import playsound

from scipy.io.wavfile import write
import sys

fs = 10000  # Sample rate

seconds = int(2.2*60*60)  # Duration of recording

if len(sys.argv) > 2:
    myrecording = sd.rec(int(int(sys.argv[2]) * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    write(sys.argv[1], fs, myrecording)  # Save as WAV file 
else:
    print(f'usage {sys.argv[0]} <filename> <duration in seconds>')
