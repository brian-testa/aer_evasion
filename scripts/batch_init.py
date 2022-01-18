################################################
#
# Purpose: Prep work for GP attacks. Should only need to do this once and then 
#          everyone else can just use it read only 
# Outputs: pickles, including
#              - dict of all data files 
#              - dicts of speaker ans sentence ids for our datasets
#              - dict with original softmax values 
#
################################################


# Don't stress about the future...
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)

# Basic imports
import sys
import os
import imp
import re
import pickle
import math
import copy
import random
from pathlib import Path
import numpy as np
import json

# Audio libraries
import librosa
import soundfile as sf
import sounddevice as sd

# ML Imports
import tensorflow as tf
import tensorflow.keras as keras

# Setup to import custom modules
if not f'{os.path.dirname(os.path.realpath(__file__))}/../scripts' in sys.path:
    sys.path.append(f'{os.path.dirname(os.path.realpath(__file__))}/../scripts')

# Get text transcripts from audio files
import transcription

# Load audio datasets and get info based upon filenames
import audiodatasets

# Ease of use with my dictionaries
import dictmgmt

# Other miscellaneous utilities (like the timer)
import misc

# Load classifier utilities
import classifiers

# Genetic programming stuff
import gp

##################################
# Do some setup and prep the datasets
#####################################

# Define some basic variables for this runtime
jar = f'{os.path.dirname(os.path.realpath(__file__))}/../pickles'
model_dir = f'{os.path.dirname(os.path.realpath(__file__))}/../models'
data_dir = f'{os.path.dirname(os.path.realpath(__file__))}/../data'

# Load the datasets; "pure" is digital data, "RECORDED" was played and recorded in our target environment E 
ravdess_files = audiodatasets.list_audio_files(os.path.join(data_dir, 'fixed_ravdess'))
ravdess_files_acoustic = audiodatasets.list_audio_files(os.path.join(data_dir, 'fixed_ravdess_acoustic'))
ravdess_speaker_ids = {audiodatasets.get_speaker_id(key) for key in ravdess_files.keys()}
ravdess_sentence_ids = {audiodatasets.get_sentence_id(key) for key in ravdess_files.keys()}

tess_files = audiodatasets.list_audio_files(os.path.join(data_dir, 'fixed_tess'))
tess_files_acoustic = audiodatasets.list_audio_files(os.path.join(data_dir, 'fixed_tess_acoustic'))
tess_speaker_ids = {audiodatasets.get_speaker_id(key) for key in tess_files.keys()}
tess_sentence_ids = {audiodatasets.get_sentence_id(key) for key in tess_files.keys()}

all_data_files = {"PURE": {**ravdess_files, **tess_files}, "RECORDED": {**ravdess_files_acoustic, **tess_files_acoustic}} 

# save the data file structure
if not os.path.exists(os.path.join(jar, 'all_data_files.pickle')):
    pickle.dump(all_data_files, open(os.path.join(jar, 'all_data_files.pickle'), 'wb'))

if not os.path.exists(os.path.join(jar, 'ravdess_speaker_ids.pickle')):
    pickle.dump(ravdess_speaker_ids, open(os.path.join(jar, 'ravdess_speaker_ids.pickle'), 'wb'))

if not os.path.exists(os.path.join(jar, 'ravdess_sentence_ids.pickle')):
    pickle.dump(ravdess_sentence_ids, open(os.path.join(jar, 'ravdess_sentence_ids.pickle'), 'wb'))

if not os.path.exists(os.path.join(jar, 'tess_speaker_ids.pickle')):
    pickle.dump(tess_speaker_ids, open(os.path.join(jar, 'tess_speaker_ids.pickle'), 'wb'))

if not os.path.exists(os.path.join(jar, 'tess_sentence_ids.pickle')):
    pickle.dump(tess_sentence_ids, open(os.path.join(jar, 'tess_sentence_ids.pickle'), 'wb'))


########################################
# Run all data through the classifiers; 
# cull anything already misclassified
########################################
emo_classifiers = {"PURE": classifiers.models["SIMPLE_PURE"], "RECORDED": classifiers.models["SIMPLE_RECORDED"]}
feature_extractors = {"PURE": classifiers.feature_extractors["SIMPLE_PURE"], "RECORDED": classifiers.feature_extractors["SIMPLE_RECORDED"]}
label_getters = {"PURE": classifiers.label_getters["SIMPLE_PURE"], "RECORDED": classifiers.label_getters["SIMPLE_RECORDED"]}

valid_classification = {}
original_softmax_values = {}

if os.path.exists(os.path.join(jar, 'valid_classification.pickle')) and os.path.exists(os.path.join(jar, 'original_softmax_values.pickle')):
    valid_classification = pickle.load(open(os.path.join(jar, 'valid_classification.pickle'), 'rb'))
    original_softmax_values = pickle.load(open(os.path.join(jar, 'original_softmax_values.pickle'), 'rb'))
else:
    for key in all_data_files.keys():
        dataset = all_data_files[key]
        emo_model = emo_classifiers[key]
        feature_extractor = feature_extractors[key]
        label_getter = label_getters[key]

        X_vals, y_vals, featureindices = feature_extractor(dataset)
        predictions = emo_model.predict(X_vals) 

        for sample_key in featureindices.keys():
            actual, predicted = label_getter(y_vals, predictions, featureindices[sample_key])
            dictmgmt.dictset(original_softmax_values, [key, sample_key], copy.deepcopy(predictions[featureindices[sample_key]]))

            if actual == predicted:
                dictmgmt.dictset(valid_classification, [key, sample_key], all_data_files[key][sample_key])
    pickle.dump(valid_classification, open(os.path.join(jar, 'valid_classification.pickle'), 'wb'))
    pickle.dump(original_softmax_values, open(os.path.join(jar, 'original_softmax_values.pickle'), 'wb'))

###############################################################
# Run all data through the transcription library; 
# cull anything with a bad transcript or really low confidence
###############################################################
all_transcripts = {}
all_confidences = {}

if os.path.exists(os.path.join(jar, 'all_transcripts.pickle')) and os.path.exists(os.path.join(jar, 'all_confidences.pickle')):
    all_transcripts = pickle.load(open(os.path.join(jar, 'all_transcripts.pickle'), 'rb'))
    all_confidences = pickle.load(open(os.path.join(jar, 'all_confidences.pickle'), 'rb'))
else:
    for key in all_data_files.keys():
        transcripts, confidences = transcription.extract_text(all_data_files[key])
        all_transcripts[key] = transcripts
        all_confidences[key] = confidences

    pickle.dump(all_transcripts, open(os.path.join(jar, 'all_transcripts.pickle'), 'wb'))
    pickle.dump(all_confidences, open(os.path.join(jar, 'all_confidences.pickle'), 'wb'))

# Next, validate the transcripts and drop everything else
valid_transcripts = {}
valid_confidences = {}

if os.path.exists(os.path.join(jar, 'valid_transcripts.pickle')) and os.path.exists(os.path.join(jar, 'valid_confidences.pickle')):
    valid_transcripts = pickle.load(open(os.path.join(jar, 'valid_transcripts.pickle'), 'rb'))
    valid_confidences = pickle.load(open(os.path.join(jar, 'valid_confidences.pickle'), 'rb'))
else:
    for key in all_data_files.keys():
        _, transcripts, confidences = transcription.remove_transcription_errors(all_data_files[key], all_transcripts[key], all_confidences[key])
        valid_transcripts[key] = transcripts
        valid_confidences[key] = confidences

    pickle.dump(valid_transcripts, open(os.path.join(jar, 'valid_transcripts.pickle'), 'wb'))
    pickle.dump(valid_confidences, open(os.path.join(jar, 'valid_confidences.pickle'), 'wb'))
    
###########################################################
#
# Finally, take the intersection of the correctly transcribed
# and correctly classified. This is our TO DO list...
#
###########################################################

if os.path.exists(os.path.join(jar, 'evasive_todo.pickle')):
    evasive_todo = pickle.load(open(os.path.join(jar, 'evasive_todo.pickle'), 'rb'))
else:
    evasive_todo = {}

    for key in all_data_files.keys():
        for sample_key in all_data_files[key].keys():
            if sample_key in valid_transcripts[key] and sample_key in valid_classification[key].keys():
                dictmgmt.dictset(evasive_todo, [key, sample_key], all_data_files[key][sample_key])

    pickle.dump(evasive_todo, open(os.path.join(jar, 'evasive_todo.pickle'), 'wb'))

    a = len(all_data_files[key])
    b = len(valid_classification[key])
    c = len(valid_transcripts[key])
    d = len(evasive_todo[key])
    print(f"Original {key} Dataset: {a} | Correctly Classified: {b} | Correctly Transcribed: {c} | Intersection: {d}")

