import os
import re
import random

# Method to get all the audio data filenames
# IMPORTANT NOTE: This method sets the data management strategy for the rest of the code.
# All data passed around will be in dictionaries with the filename (no path) as the key
def list_audio_files(datadir):
    filelist = {}
    
    for dir_path, subdir_list, file_list in os.walk(datadir):
        for filename in file_list:
            if filename.endswith('wav') and filename.startswith('003-'):
                filelist[filename] = f'{dir_path}/{filename}'

    return filelist

def get_field_n(filename, n):
    parts = re.findall(r'[0-9][0-9][0-9]', filename)
    if len(parts) != 7 or n > 6:
        return None
    else:
        return int(parts[n])
    
def get_emotion_id(filename):
    return get_field_n(filename, 2)

def get_sentence_id(filename):
    return get_field_n(filename, 4)

def get_speaker_id(filename):
    return get_field_n(filename, -1)

def is_variant(filename):
    return re.search(r'.*(_.*).wav', filename) is not None

def startified_sample(filelist, sample_size):
    by_sentence = {}
    for key in filelist:
        sid = get_sentence_id(key)
        if not sid in by_sentence:
            by_sentence[sid] = set()
        by_sentence[sid].add(key)
    
    randomized_keys = list(by_sentence.keys())
    random.shuffle(randomized_keys)
    
    # TODO - Finish this
    sample = {}

    while len(sample) < sample_size and len(randomized_keys) > 0:
        for sid in randomized_keys:
            choices = by_sentence[sid]
            choice = random.sample(list(choices), 1)[0]
            sample[choice] = filelist[choice]
            by_sentence[sid].remove(choice)

            if len(by_sentence[sid]) == 0:
                del by_sentence[sid]
            if len(sample) >= sample_size:
                break
        randomized_keys = list(by_sentence.keys())
        random.shuffle(randomized_keys)  

    return sample
    
    