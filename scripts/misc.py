import time
import librosa
import sounddevice as sd
import math

start = time.time()
def resetTimer():
    global start
    start = time.time()
    
def elapsedTime():
    elapsed = math.ceil(time.time() - start)
    hours = math.floor(elapsed/3600)
    minutes = math.floor(elapsed%3600/60)
    seconds = elapsed%60
    print(f'Elapsed time: {hours} Hours, {minutes} Minutes and {seconds} Seconds ({time.ctime(time.time())})', flush=True)
    
def alert_me():
    gameover, _ = librosa.load("../data/misc/game_over.wav", duration=4.0, sr=48000)
    sd.play(gameover)
    sd.wait()
