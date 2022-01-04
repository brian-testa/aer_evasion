import os
import re

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