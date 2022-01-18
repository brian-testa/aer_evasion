from google.cloud import speech
import vosk
import os
import librosa
import re
import wave
import uuid
from pathlib import Path
import soundfile as sf
import json
import copy

text_extractor = 'VOSK'
vosk_model_path = f'{os.path.dirname(os.path.realpath(__file__))}/../lib/vosk_models/vosk-model-small-en-us-0.15'
#vosk_model_path = f'{os.path.dirname(os.path.realpath(__file__))}/../lib/vosk_models/vosk-model-en-us-daanzu-20200905'
#vosk_model_path = f'{os.path.dirname(os.path.realpath(__file__))}/../lib/vosk_models/vosk-model-en-us-daanzu-20200905-lgraph'

google_app_credentials = f'{os.path.dirname(os.path.realpath(__file__))}/../lib/MyFirstProject-fae4c18cf725.json'


# Let's check those transcripts
def valid_transcript(v, confidence=1):
    return valid_transcript_ravdess(v, confidence) or valid_transcript_savee(v, confidence) or valid_transcript_tess(v, confidence)

tess_sentences = {'say the word back': 3, 'say the word bar': 4, 'say the word base': 5, 'say the word bath': 6, 'say the word bean': 7, 'say the word beg': 8, 'say the word bite': 9, 'say the word boat': 10, 'say the word bone': 11, 'say the word book': 12, 'say the word bought': 13, 'say the word burn': 14, 'say the word cab': 15, 'say the word calm': 16, 'say the word came': 17, 'say the word cause': 18, 'say the word chain': 19, 'say the word chair': 20, 'say the word chalk': 21, 'say the word chat': 22, 'say the word check': 23, 'say the word cheek': 24, 'say the word chief': 25, 'say the word choice': 26, 'say the word cool': 27, 'say the word dab': 28, 'say the word date': 29, 'say the word dead': 30, 'say the word death': 31, 'say the word deep': 32, 'say the word dime': 33, 'say the word dip': 34, 'say the word ditch': 35, 'say the word dodge': 36, 'say the word dog': 37, 'say the word doll': 38, 'say the word door': 39, 'say the word fail': 40, 'say the word fall': 41, 'say the word far': 42, 'say the word fat': 43, 'say the word fit': 44, 'say the word five': 45, 'say the word food': 46, 'say the word gap': 47, 'say the word gas': 48, 'say the word gaze': 49, 'say the word germ': 50, 'say the word get': 51, 'say the word gin': 52, 'say the word goal': 53, 'say the word good': 54, 'say the word goose': 55, 'say the word gun': 56, 'say the word half': 57, 'say the word hall': 58, 'say the word hash': 59, 'say the word hate': 60, 'say the word have': 61, 'say the word haze': 62, 'say the word hire': 63, 'say the word hit': 64, 'say the word hole': 65, 'say the word home': 66, 'say the word hurl': 67, 'say the word hush': 68, 'say the word jail': 69, 'say the word jar': 70, 'say the word join': 71, 'say the word judge': 72, 'say the word jug': 73, 'say the word juice': 74, 'say the word keen': 75, 'say the word keep': 76, 'say the word keg': 77, 'say the word kick': 78, 'say the word kill': 79, 'say the word king': 80, 'say the word kite': 81, 'say the word knock': 82, 'say the word late': 83, 'say the word laud': 84, 'say the word lean': 85, 'say the word learn': 86, 'say the word lease': 87, 'say the word lid': 88, 'say the word life': 89, 'say the word limb': 90, 'say the word live': 91, 'say the word loaf': 92, 'say the word long': 93, 'say the word lore': 94, 'say the word lose': 95, 'say the word lot': 96, 'say the word love': 97, 'say the word luck': 98, 'say the word make': 99, 'say the word match': 100, 'say the word merge': 101, 'say the word mess': 102, 'say the word met': 103, 'say the word mill': 104, 'say the word mob': 105, 'say the word mode': 106, 'say the word mood': 107, 'say the word moon': 108, 'say the word mop': 109, 'say the word mouse': 110, 'say the word nag': 111, 'say the word name': 112, 'say the word near': 113, 'say the word neat': 114, 'say the word nice': 115, 'say the word note': 116, 'say the word numb': 117, 'say the word pad': 118, 'say the word page': 119, 'say the word pain': 120, 'say the word pass': 121, 'say the word pearl': 122, 'say the word peg': 123, 'say the word perch': 124, 'say the word phone': 125, 'say the word pick': 126, 'say the word pike': 127, 'say the word pole': 128, 'say the word pool': 129, 'say the word puff': 130, 'say the word rag': 131, 'say the word raid': 132, 'say the word rain': 133, 'say the word raise': 134, 'say the word rat': 135, 'say the word reach': 136, 'say the word read': 137, 'say the word red': 138, 'say the word ring': 139, 'say the word ripe': 140, 'say the word road': 141, 'say the word room': 142, 'say the word rose': 143, 'say the word rot': 144, 'say the word rough': 145, 'say the word rush': 146, 'say the word said': 147, 'say the word sail': 148, 'say the word search': 149, 'say the word seize': 150, 'say the word sell': 151, 'say the word shack': 152, 'say the word shall': 153, 'say the word shawl': 154, 'say the word sheep': 155, 'say the word shirt': 156, 'say the word should': 157, 'say the word shout': 158, 'say the word size': 159, 'say the word soap': 160, 'say the word soup': 161, 'say the word sour': 162, 'say the word south': 163, 'say the word sub': 164, 'say the word such': 165, 'say the word sure': 166, 'say the word take': 167, 'say the word talk': 168, 'say the word tape': 169, 'say the word team': 170, 'say the word tell': 171, 'say the word thin': 172, 'say the word third': 173, 'say the word thought': 174, 'say the word thumb': 175, 'say the word time': 176, 'say the word tip': 177, 'say the word tire': 178, 'say the word ton': 179, 'say the word tool': 180, 'say the word tough': 181, 'say the word turn': 182, 'say the word vine': 183, 'say the word voice': 184, 'say the word void': 185, 'say the word vote': 186, 'say the word wag': 187, 'say the word walk': 188, 'say the word wash': 189, 'say the word week': 190, 'say the word wheat': 191, 'say the word when': 192, 'say the word which': 193, 'say the word whip': 194, 'say the word white': 195, 'say the word wife': 196, 'say the word wire': 197, 'say the word witch': 198, 'say the word yearn': 199, 'say the word yes': 200, 'say the word young': 201, 'say the word youth': 202}

def valid_transcript_tess(v, confidence=1):
    good_transcript = False

    if confidence <= 0.5:
        good_transcript = False
    else:
        for key in tess_sentences.keys():
            if key == v:
                good_transcript = True
                break
    
    return good_transcript
    
def valid_transcript_savee(v, confidence=1):
    good_transcript = False
        
    if re.match(r'please take this dirty tablecloth to the cleaners for me', v) is not None and confidence > .5:
        good_transcript = True
    # REPREAT AS NECESSARY...does not need to be regular expressions
    elif re.match(r'catastrophic economic cutbacks neglect the poor', v) is not None and confidence > .5:
        good_transcript = True

    elif re.match(r'will you tell me why', v) is not None and confidence > .5:
        good_transcript = True

    elif re.match(r'no return address whatsoever', v) is not None and confidence > .5:
        good_transcript = True

    elif re.match(r'he will allow a rare lie', v) is not None and confidence > .5:
        good_transcript = True

    elif re.match(r'laboratory astrophysics', v) is not None and confidence > .5:
        good_transcript = True

    elif re.match(r'don’t ask me to carry an oily rug like that', v) is not None and confidence > .5:
        good_transcript = True

    elif re.match(r'straw hats are out of fashion this year', v) is not None and confidence > .5:
        good_transcript = True

    elif re.match(r'who authorized the unlimited expense account', v) is not None and confidence > .5:
        good_transcript = True

    elif re.match(r'if people were more generous there would be no need for welfare', v) is not None and confidence > .5:
        good_transcript = True

    elif re.match(r'he stole a dime from a beggar', v) is not None and confidence > .5:
        good_transcript = True

    elif re.match(r'call and ambulance for medical assistance', v) is not None and confidence > .5:
        good_transcript = True

    elif re.match(r'those musicians harmonize marvelously', v) is not None and confidence > .5:
        good_transcript = True

    elif re.match(r'if the farm is rented the rent must be paid', v) is not None and confidence > .5:
        good_transcript = True

    elif re.match(r'how good is your endurance', v) is not None and confidence > .5:
        good_transcript = True

    elif re.match(r'basketball can be an entertaining sport', v) is not None and confidence > .5:
        good_transcript = True

    elif re.match(r'he would not carry a briefcase', v) is not None and confidence > .5:
        good_transcript = True

    elif re.match(r'i\'d ride the subway but i haven\'t enough change', v) is not None and confidence > .5:
        good_transcript = True

    elif re.match(r'salvation reconsidered', v) is not None and confidence > .5:
        good_transcript = True

    elif re.match(r'he always seemed to have money in his pocket', v) is not None and confidence > .5:
        good_transcript = True

    elif re.match(r'the oasis was a mirage', v) is not None and confidence > .5:
        good_transcript = True

    elif re.match(r'keep your seats boys i just want to put some finishing touches on this thing', v) is not None and confidence > .5:
        good_transcript = True

    elif re.match(r'right now may not be the best time for business merges', v) is not None and confidence > .5:
        good_transcript = True

    elif re.match(r'the pulsing glow of a cigarette', v) is not None and confidence > .5:
        good_transcript = True

    elif re.match(r'this is a problem that goes considerably beyond questions of salary and tenure', v) is not None and confidence > .5:
        good_transcript = True

    elif re.match(r'american newspaper reviewers like to call his plays nihilistic', v) is not None and confidence > .5:
        good_transcript = True

    elif re.match(r'withdraw all phony accusations at once', v) is not None and confidence > .5:
        good_transcript = True

    elif re.match(r'the small boy put the worm on the hook', v) is not None and confidence > .5:
        good_transcript = True

    elif re.match(r'special task forces rescue hostages from kidnappers', v) is not None and confidence > .5:
        good_transcript = True

    elif re.match(r'agricultural products are unevenly distributed', v) is not None and confidence > .5:
        good_transcript = True

    elif re.match(r'tornadoes often destroy acres of farmland', v) is not None and confidence > .5:
        good_transcript = True

    elif re.match(r'the best way to learn is to solve extra problems', v) is not None and confidence > .5:
        good_transcript = True

    elif re.match(r'greg buys fresh milk each weekday morning', v) is not None and confidence > .5:
        good_transcript = True

    elif re.match(r'before thursday\'s exam review every formula', v) is not None and confidence > .5:
        good_transcript = True

    elif re.match(r'the east(ern){0,1} coast is a place for pure pleasure and excitement', v) is not None and confidence > .5:
        good_transcript = True

    elif re.match(r'jeff thought you argued{0,1} in favor of a centrifuge purchase', v) is not None and confidence > .5:
        good_transcript = True

    elif re.match(r'the prospect of cutting back spending is an unpleasant one for any governor', v) is not None and confidence > .5:
        good_transcript = True

    elif re.match(r'his artistic accomplishments guaranteed him entry into any social gathering', v) is not None and confidence > .5:
        good_transcript = True

    elif re.match(r'don\'t ask me (the|to) carry an oily rag like that', v) is not None and confidence > .5:
        good_transcript = True

    elif re.match(r'but this doesn\'t detract from its merit as an interesting if not great film', v) is not None and confidence > .5:
        good_transcript = True

    elif re.match(r'agricultural products (or|are) unevenly distributed', v) is not None and confidence > .5:
        good_transcript = True

    elif re.match(r'right now may not be the best time for business mergers', v) is not None and confidence > .5:
        good_transcript = True

    elif re.match(r'this is a problem (the|that) goes considerably beyond question of salary and tenure', v) is not None and confidence > .5:
        good_transcript = True

    elif re.match(r'call an ambulance for medical assistance', v) is not None and confidence > .5:
        good_transcript = True

    elif re.match(r'we will achieve a morbid sense of what it is by realizing what it is not', v) is not None and confidence > .5:
        good_transcript = True

    elif re.match(r'the clumsy customers{0,1} spilled some expensive perfume', v) is not None and confidence > .5:
        good_transcript = True

    elif re.match(r'product development was proceeding too{0,1} slowly', v) is not None and confidence > .5:
        good_transcript = True

    elif re.match(r'destroy every file related to my audits', v) is not None and confidence > .5:
        good_transcript = True

    elif re.match(r'the nearest synagogue may not be within walking distance', v) is not None and confidence > .5:
        good_transcript = True

    elif re.match(r'he would allow a rare lie', v) is not None and confidence > .5:
        good_transcript = True

    elif re.match(r'the diagnosis was discouraging however he was not overly worried', v) is not None and confidence > .5:
        good_transcript = True

    elif re.match(r'they enjoy it when i auditioned', v) is not None and confidence > .5:
        good_transcript = True

    elif re.match(r'pretty soon a woman came along carrying a folded umbrella as a walking stick', v) is not None and confidence > .5:
        good_transcript = True

    elif re.match(r'the east(ern){0,1} coast is a place for (peer|pure) pleasure and excitement', v) is not None and confidence > .5:
        good_transcript = True

    elif re.match(r'his shoulder felt as if it were broken', v) is not None and confidence > .5:
        good_transcript = True

    elif re.match(r'calcium makes bones and teeth strong', v) is not None and confidence > .5:
        good_transcript = True

    elif re.match(r'as such it was beyond politics and had no need of justification (by|but) a message', v) is not None and confidence > .5:
        good_transcript = True

    elif re.match(r'kindergarten children decorate the(ir){0,1} (classrooms|costumes) (were|are|for) all holidays', v) is not None and confidence > .5:
        good_transcript = True

    elif re.match(r'a few years later the dome (fell in|felon)', v) is not None and confidence > .5:
        good_transcript = True
        
    return good_transcript

def valid_transcript_ravdess(v, confidence=1):
    # RAVDESS
    return re.match(r'(dogs are sitting{0,1}|kids are talking{0,1}) by the door', v) is not None and confidence > 0.5

def extract_text(files):
    base_temp_file_dir = f'/tmp/bpt_{str(uuid.uuid4())}'
    Path().mkdir(parents=True, exist_ok=True)
    
    if text_extractor == 'GOOGLE':
        return extract_text_google(files)
    else:
        return extract_text_vosk(files)

# Define function to get "ground truth" according to VOSK/KALDI 
# Input: Dict of files (key is filename, value is full path)
# Returns: A dictionary of transcripts and a dictionary of confidence levels; keys are the same as the input
vosk_model = vosk.Model(vosk_model_path)
def extract_text_vosk(files):
    base_temp_file_dir = f'/tmp/bpt_{str(uuid.uuid4())}'
    Path(base_temp_file_dir).mkdir(parents=True, exist_ok=True)

    transcripts = {}
    confidences = {}
    
    for key, file in files.items():

        # Hack to ensure that we have mono data
        x, sr = librosa.load(file, mono=True, sr=48000)
        sf.write(f'{base_temp_file_dir}/foo.wav', x, sr, )
                
        with wave.open(f'{base_temp_file_dir}/foo.wav', "rb") as in_file:
#        with wave.open(file, "rb") as in_file:
            rec = vosk.KaldiRecognizer(vosk_model, in_file.getframerate())

            while True:
                data = in_file.readframes(50000)
                if len(data) == 0:
                    break
                rec.AcceptWaveform(data)
            
            result = rec.FinalResult()
            result = json.loads(result)
            try:
                conf = 0.0
                for res in result['result']:
                    conf += res['conf']
                conf = conf / len(result['result'])
                transcripts[key] = result['text']
                confidences[key] = conf
            except KeyError:
                transcripts[key] = "Ooops"
                confidences[key] = 0.0
                
    return transcripts, confidences

    

# Define function to get "ground truth" according to Google's spech API 
# Input: Dict of files (key is filename, value is full path)
# Returns: A dictionary of transcripts and a dictionary of confidence levels; keys are the same as the input

# Google credentials for call to 
#os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=google_app_credentialspad

def extract_text_google(files):
    base_temp_file_dir = f'/tmp/bpt_{str(uuid.uuid4())}'
    Path(base_temp_file_dir).mkdir(parents=True, exist_ok=True)
    
    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code="en-US",
    )
    transcripts = {}
    confidences = {}
    for key, file in files.items():
        
        # Hack to ensure that we have mono data for the Google speech API
        x, sr = librosa.load(file, mono=True)
        sf.write(f'{base_temp_file_dir}/foo.wav', x, sr, )
        
        # Now call the API to extract text from the audio files
        with open(f'{base_temp_file_dir}/foo.wav', "rb") as in_file: # opening for [r]eading as [b]inary
            data = in_file.read() # if you only wanted to read 512 bytes, do .read(512)
            in_file.close()
            audio = speech.RecognitionAudio(content=data)
            response = client.recognize(config=config, audio=audio)

            for result in response.results:
                for alt in result.alternatives:
                    transcripts[key] = alt.transcript
                    confidences[key] = alt.confidence
    return transcripts, confidences

def remove_transcription_errors(ravdess_files, transcripts, confidences):
    # Performing some quick cleanup...we don't want to proceed on files that Google's Speech to Text misunderstood
    # This will exclude missed sentences (though I will accept "talkin" and "sittin") or extractions with < 50% confidence
    removeme = []
    for k, v in transcripts.items():
        if not valid_transcript(v, confidences[k]):
            removeme.append(k)

    # Duplicationg dictionary b/c we can still use incorrectly transcribed audio to train the upcoming classifier
    ravdess_files_correct_transcripts = copy.deepcopy(ravdess_files)
    fixed_transcripts = copy.deepcopy(transcripts)
    fixed_confidences = copy.deepcopy(confidences)
    
    for k in removeme:
        del ravdess_files_correct_transcripts[k]
        del fixed_transcripts[k]
        del fixed_confidences[k]
    return ravdess_files_correct_transcripts, fixed_transcripts, fixed_confidences
