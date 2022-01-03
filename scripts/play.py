import sounddevice as sd
from playsound import playsound

from scipy.io.wavfile import write
import sys
import time
import subprocess

if len(sys.argv) > 1:
    playsound(sys.argv[1])
else:
    print(f'usage {sys.argv[0]} <audio 1>')
