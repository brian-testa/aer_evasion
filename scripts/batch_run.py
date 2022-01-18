################################################
#
# Purpose: Run GP attacks variants. Methods vary
# Inputs:  
# Outputs: pickles, including
#              - dict of all data files 
#              - dict with original softmax values 
#
################################################


# Don't stress about the future...
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

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

jar = f'{os.path.dirname(os.path.realpath(__file__))}/../pickles'
model_dir = f'{os.path.dirname(os.path.realpath(__file__))}/../models'
data_dir = f'{os.path.dirname(os.path.realpath(__file__))}/../data'


############################################################
#
# Load all of the necessary pickles dumped from batch_init.py
#
############################################################

# Classifier data
original_softmax_values = pickle.load(open(os.path.join(jar, 'original_softmax_values.pickle'), 'rb'))

# Transcription data
valid_transcripts = pickle.load(open(os.path.join(jar, 'valid_transcripts.pickle'), 'rb'))
valid_confidences = pickle.load(open(os.path.join(jar, 'valid_confidences.pickle'), 'rb'))

# Speaker and sentence IDs by datasets
speaker_ids = { "RAVDESS": pickle.load(open(os.path.join(jar, 'ravdess_speaker_ids.pickle'), 'rb')),
                "TESS": pickle.load(open(os.path.join(jar, 'tess_speaker_ids.pickle'), 'rb'))
}

sentence_ids = { "RAVDESS": pickle.load(open(os.path.join(jar, 'ravdess_sentence_ids.pickle'), 'rb')),
                 "TESS": pickle.load(open(os.path.join(jar, 'tess_sentence_ids.pickle'), 'rb'))
}

# Evasive TODO (correctly classified and transcribed)
evasive_todo = pickle.load(open(os.path.join(jar, 'evasive_todo.pickle'), 'rb'))

########################################################
#
# Load some other important datastructures and objects
#
########################################################

# Classifier pointers
emo_classifiers = {"PURE": classifiers.models["SIMPLE_PURE"], "RECORDED": classifiers.models["SIMPLE_RECORDED"]}
feature_extractors = {"PURE": classifiers.feature_extractors["SIMPLE_PURE"], "RECORDED": classifiers.feature_extractors["SIMPLE_RECORDED"]}
label_getters = {"PURE": classifiers.label_getters["SIMPLE_PURE"], "RECORDED": classifiers.label_getters["SIMPLE_RECORDED"]}

# Autoencoders
def custom_loss(y_true, y_pred, sample_weight=None, **kwargs):
    foo = tf.real(tf.signal.fft(tf.complex(y_true, 0.0)))
    foo1 = tf.real(tf.signal.fft(tf.complex(y_pred, 0.0)))
    return keras.losses.mean_squared_error(foo, foo1)

autoencoders = { "PURE": None, 
                 "MSE": keras.models.load_model(os.path.join(model_dir, 'autoencoder.h5')),
                 "MSE-FFT": keras.models.load_model(os.path.join(model_dir, 'autoencoder_customloss.h5'), custom_objects={"custom_loss":custom_loss})
               }

##########################################################################
#
# Setup the GP framework and prepare to run the experiment
#
#########################################################################

# Define some baseline perturbation characteristics; in general, should not change these between runs
min_freq=100
max_freq=8000 
min_duration=2.5
max_duration=4.0
min_offset=0.0 
max_offset=0.5
min_muzzle=25
max_muzzle=50

# Applies a single perturbation to a single file
def apply_perturbation(individual, audio_key, filelist, temp_dir='/tmp/bpt', max_duration=4.0, sample_rate=48000, autoencoder_key="PURE"):

    # Everything is file based...make sure that the temp directory is there
    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    
    x, sr = librosa.load(filelist[audio_key], duration=max_duration, sr=sample_rate)

    new_x = x
    for freq,duration,offset,muzzle in zip(individual[::4], individual[1::4], individual[2::4], individual[3::4]):

        # Adjust normalized values
        freq = freq * (max_freq - min_freq) + min_freq
        duration = duration * (max_duration - min_duration) + min_duration
        offset = offset * (max_offset - min_offset) + min_offset
        muzzle = muzzle * (max_muzzle - min_muzzle) + min_muzzle
        
        tone = librosa.tone(freq, duration=duration, sr=sample_rate)
        tone = tone/muzzle
        int_off = int(sr * offset)
        
        # Called if we are running acoustically (i.e. - using the autoencoder to simulate the room's acoustics)
        if not autoencoder_key == "PURE":
            tone = librosa.util.fix_length(tone, int(max_duration*sample_rate))
            
            # Need to resample down and up for the autoencoder (ae hardcoded to sr=10000)
            tone = librosa.resample(tone, orig_sr=sample_rate, target_sr=10000)
            acoustic_tone = autoencoders[autoencoder_key].predict(np.array([np.reshape(tone, (np.shape(tone)[0], 1))]))[0]
            acoustic_tone = librosa.resample(tone, orig_sr=10000, target_sr=sample_rate)
            tone=acoustic_tone
        
        for indx in range(int_off, min(len(tone)+int_off, len(new_x))):
            new_x[indx] += tone[indx-int_off]

    sf.write(f'{temp_dir}/{audio_key}', new_x, sample_rate, )
    return(f'{temp_dir}/{audio_key}')

# Scores transcriptions
def measure_text_extract_goodness(new_transcripts, new_confidences, orig_confidences):
    retval = 1.0
    num_samples = len(new_transcripts)
    
    for key, text in new_transcripts.items():
        if not transcription.valid_transcript(text):
            retval = 0.0
            break
        
        try:
            if new_confidences[key] < orig_confidences[key]:
                retval -= (orig_confidences[key] - new_confidences[key]) / num_samples
        except KeyError:
            # Skip this one if there is a KeyError (shouldn't happen)
            print(f'Key Error: {key}')
            
    return retval

# Scores misclassification
def measure_classification_degradation( 
                                        classifier, # classifier being used
                                        feature_extractor, # feature extractor for this classifier
                                        label_getter, # knows how to parse the results
                                        original_softmax, # dict containing softmax values for original files
                                        modified_files, # dict containing files that we have modified to fool the classifier 
                                        misclassification_bonus=50, # Bonus given to score if we have an actual misclassification 
                                        target_class=None
                                      ):

    score = 0.0
    
    X_vals, y_vals, featureindices = feature_extractor(modified_files)
    predictions = classifier.predict(X_vals) 

    for sample_key in featureindices.keys():
        actual, predicted = label_getter(y_vals, predictions, featureindices[sample_key])

        
        
        # Bonus for misclassifications
        if target_class is None: # untargeted evason

            score += max(original_softmax[sample_key][actual]-predictions[featureindices[sample_key]][actual], 0)
            
            # Bonus
            if actual != predicted:
                score += misclassification_bonus
        else: # targeted evason

            score += max(predictions[featureindices[sample_key]][target_class] - original_softmax[sample_key][target_class], 0)

            # Bonus
            if predicted == target_class:
                score += misclassification_bonus
    return score

# Finally, we pull together all of these measures into a single fitness measure
# DEAP does support multiple, independent fitness measures but they were not working for me
def evaluation(individual):
    
    # The globals used by this function are stored in a dictionary
    global evaluation_dictionary
    
    evaluation_file_list = evaluation_dictionary["Evaluation File List"]
    evaluation_sample_size = evaluation_dictionary["Evaluation Sample Size"]
    original_transcription_confidences = evaluation_dictionary["Original Transcription Confidences"]
    evaluation_autoencoder_key = evaluation_dictionary["Evaluation Autoencoder Key"]
    evaluation_classifier_modality = evaluation_dictionary["Evaluation Classifier Modality"] # PURE|RECORDED
    target_class = evaluation_dictionary["Target Class"]
    original_softmax = evaluation_dictionary["Original Softmax Values"]
    temp_dir = evaluation_dictionary["Temporary Directory"]
    
    # Derive a few from the classifier modality
    emo_classifier = emo_classifiers[evaluation_classifier_modality]
    feature_extractor = feature_extractors[evaluation_classifier_modality]
    label_getter = label_getters[evaluation_classifier_modality]

    # Take a random sample and generate modified versions of these files using individual's perturbation
    sample_keys = random.sample(list(evaluation_file_list), min(evaluation_sample_size, len(evaluation_file_list)))

    modified_files = { key:apply_perturbation(individual=individual, audio_key=key, temp_dir=temp_dir, filelist=evaluation_file_list, autoencoder_key=evaluation_autoencoder_key) \
                      for key in sample_keys 
                     }

    # Now, assess the fitness of this individual
    fitness_score = 0.0

    if len(modified_files) > 0:
        classifier_fool_score = measure_classification_degradation( classifier=emo_classifier, 
                                                                    feature_extractor=feature_extractor,
                                                                    label_getter=label_getter,
                                                                    original_softmax=original_softmax,
                                                                    modified_files=modified_files,
                                                                    target_class=target_class
                                                                  )
        text_extract_score = 0.0

        # Only waste time on the text extraction if we are making some progress with the classifier
        if classifier_fool_score > 0.0:

            # Adding some retires in here becaue the VOSK stuff is a little buggy (with the model that I'm using)
            eval_transcripts = {}
            eval_confidences = {}

            # 10 retries should be enough
            for _ in range(10):
                if len(modified_files) == 0:
                    break

                new_transcripts, new_confidences = transcription.extract_text(modified_files)

                remove_me = []
                for key in new_transcripts.keys():
                    if new_confidences[key] != 0.0:
                        eval_transcripts[key] = new_transcripts[key]
                        eval_confidences[key] = new_confidences[key]
                        remove_me.append(key)

                for key in remove_me:
                    del modified_files[key]

            text_extract_score = measure_text_extract_goodness(new_transcripts, new_confidences, original_transcription_confidences)

        fitness_score = text_extract_score*classifier_fool_score

    return fitness_score,

gp.initialize_gp_environment(fitness_function=evaluation)


#######################################################################
#
# That's all the setup. Here are the experiment options 
# (which our JSON config will select from and configure)
#
#######################################################################
evaluation_dictionary = {}

def kfold_sample_run(runtime_str, num_trials, sample_size, modality="PURE", dataset_str="ALL", autoencoder_key=None, target_class=None, starting_population=None, ngen=40):
    pops = []
    samples = []

    for k in range(num_trials):
        rstr = f"{runtime_str}_TRIAL_{k:02}"
        p, s = sample_run( rstr, sample_size, modality=modality, dataset_str=dataset_str, autoencoder_key=autoencoder_key, 
                           target_class=target_class, starting_population=starting_population, ngen=ngen )
        pops.append(p)
        samples.append(s)

    return pops, samples

# One and done - sample based
def sample_run(runtime_str, sample_size, modality="PURE", dataset_str="ALL", autoencoder_key=None, target_class=None, starting_population=None, ngen=40):
    global evaluation_dictionary

    # Initialize the dictionary used by the GP fitness evaluation method
    evaluation_dictionary["Evaluation Sample Size"] = 40
    evaluation_dictionary["Original Transcription Confidences"] = valid_confidences[modality]
    evaluation_dictionary["Evaluation Autoencoder Key"] = autoencoder_key
    evaluation_dictionary["Evaluation Classifier Modality"] = modality
    evaluation_dictionary["Target Class"] = target_class
    evaluation_dictionary["Original Softmax Values"] = original_softmax_values[modality]
    evaluation_dictionary["Temporary Directory"] = f'/tmp/bpt_{runtime_str}'

#    gp.initialize_gp_environment(fitness_function=evaluation)

    # Make sure that we need to do this run
    pop_pickle_fn = f"{jar}/populations_{runtime_str}_samplesize_{sample_size}.pickle"
    sample_pickle_fn = f"{jar}/samples_{runtime_str}_samplesize_{sample_size}.pickle"

    if os.path.exists(pop_pickle_fn) and os.path.exists(sample_pickle_fn):
        populations = pickle.load(open(pop_pickle_fn, 'rb'))
        sample_dict = pickle.load(open(sample_pickle_fn, 'rb'))
    else:
        sample = random.sample(list(sentence_ids[dataset_str]), sample_size)
        train_test_size = sample_size // 2

        train_sample = { sample_key:evasive_todo[modality][sample_key] for sample_key in evasive_todo[modality] \
                                                         if audiodatasets.get_sentence_id(sample_key) in sample[:train_test_size] } 
        test_sample = { sample_key:evasive_todo[modality][sample_key] for sample_key in evasive_todo[modality] \
                                                         if audiodatasets.get_sentence_id(sample_key) in sample[train_test_size:] } 
        eval_sample = { sample_key:evasive_todo[modality][sample_key] for sample_key in evasive_todo[modality] \
                                                         if not audiodatasets.get_sentence_id(sample_key) in sample } 

        sample_dict = {"TRAIN": train_sample, "TEST": test_sample, "EVAL": eval_sample}
        pickle.dump(sample_dict, open(sample_pickle_fn, 'wb'))

        evaluation_dictionary["Evaluation File List"] = train_sample

        populations, _ = gp.run_gp(starting_population=starting_population, number_of_generations=ngen)
        pickle.dump(list(populations), open(pop_pickle_fn, 'wb'))

    return populations, sample_dict


# One and done
def simple_run(runtime_str, modality="PURE", dataset_str="ALL", autoencoder_key=None, target_class=None, starting_population=None, ngen=40):
    global evaluation_dictionary

    # Initialize the dictionary used by the GP fitness evaluation method

    if dataset_str == "ALL":
        evaluation_dictionary["Evaluation File List"] = evasive_todo[modality]
    else:

        evaluation_dictionary["Evaluation File List"] = {key:evasive_todo[modality][key] for key in evasive_todo[modality].keys() \
                                                            if audiodatasets.get_sentence_id(key) in sentence_ids[dataset_str]}
    evaluation_dictionary["Evaluation Sample Size"] = 40
    evaluation_dictionary["Original Transcription Confidences"] = valid_confidences[modality]
    evaluation_dictionary["Evaluation Autoencoder Key"] = autoencoder_key
    evaluation_dictionary["Evaluation Classifier Modality"] = modality
    evaluation_dictionary["Target Class"] = target_class
    evaluation_dictionary["Original Softmax Values"] = original_softmax_values[modality]
    evaluation_dictionary["Temporary Directory"] = f'/tmp/bpt_{runtime_str}'

#    gp.initialize_gp_environment(fitness_function=evaluation)

    if os.path.exists(f"{jar}/populations_{runtime_str}.pickle"):
        populations = pickle.load(open(f"{jar}/populations_{runtime_str}.pickle", 'rb'))
    else:
        populations, _ = gp.run_gp(starting_population=starting_population, number_of_generations=ngen)
        pickle.dump(list(populations), open(f"{jar}/populations_{runtime_str}.pickle", 'wb'))

    return populations    

# Run, then call another, then call another with the results of the previous, etc.
def cascaded_run(runtime_str, modalities, dataset_strs, autoencoder_keys, target_classes, starting_population=None, ngens=40):
    start_pop = starting_population

    print(f"WTF: {modalities}")

    for run_indx in range(len(modalities)):

        if os.path.exists(f"{jar}/populations_{runtime_str}_CASCADEINDX_{run_indx}.pickle"):
            populations = pickle.load(open(f"{jar}/populations_{runtime_str}_CASCADEINDX_{run_indx}.pickle", 'rb'))
        else:
            populations = simple_run(runtime_str, modality=modalities[run_indx], dataset_str=dataset_strs[run_indx], 
                                        autoencoder_key=autoencoder_keys[run_indx], target_class=target_classes[run_indx], 
                                        starting_population=start_pop, ngen=ngens[run_indx])
            pickle.dump(list(populations), open(f"{jar}/populations_{runtime_str}_CASCADEINDX_{run_indx}.pickle", 'wb'))
        start_pop = populations

    return populations


#######################################################
#
# Driver that pulls it all together
#
#######################################################

def get_start_pop(fname, gen):
    if isinstance(fname, list):
        if len(fname) == 0:
            return None
    elif isinstance(fname, str) and os.path.exists(fname):
        pops = pickle.load(open(fname, 'rb'))
        return pops[gen]
    else:
        return None

# NEED A UNIQUE ID FOREACH RUN
def main():

    if not (len(sys.argv) > 1 and os.path.exists(sys.argv[1])):
        print(f"usage: {argv[0]} <path to config file>") 

    config = json.load(open(sys.argv[1], 'r'))

    # Each key in config is a "job"
    for job in config.values():
        misc.resetTimer()
        print("=============================================================")
        print("Starting job:")
        print(job)
        print("=============================================================")


        # Load the config
        runtime_str = job['runtime_str']
        method_name = job['method']
        parameters = job['params']
        starting_population = get_start_pop(parameters['starting_population'], parameters['starting_generation'])

        # PURE or RECORDED
        modality = parameters['modality'] if 'modality' in parameters else None
        modalities = parameters['modalities'] if 'modalities' in parameters else None

        # Number of generations to run
        ngen = parameters['ngen'] if 'ngen' in parameters else None
        ngens = parameters['ngens'] if 'ngens' in parameters else None

        # Target class (if targeted evasion)
        target_class = parameters['target_class'] if 'target_class' in parameters and parameters['target_class'] >= 0 else None
        target_classes = parameters['target_classes'] if 'target_classes' in parameters else None

        # Dataset(s) used to train
        dataset_str = parameters['dataset_str'] if 'dataset_str' in parameters else None
        dataset_strs = parameters['dataset_strs'] if 'dataset_strs' in parameters else None

        # Autoencoder(s) used
        autoencoder_key = parameters['autoencoder_key'] if 'autoencoder_key' in parameters else None
        autoencoder_keys = parameters['autoencoder_keys'] if 'autoencoder_keys' in parameters else None

        # Method for this run
        method = globals()[method_name] if method_name in globals() else None


        if method_name == 'simple_run':
            populations = method( runtime_str, modality=modality, dataset_str=dataset_str, autoencoder_key=autoencoder_key, 
                                  target_class=target_class, starting_population=starting_population, ngen=ngen)

        elif method_name == 'sample_run':
            sample_size = parameters['sample_size']
            populations = method( runtime_str, sample_size=sample_size, modality=modality, dataset_str=dataset_str, autoencoder_key=autoencoder_key, 
                                  target_class=target_class, starting_population=starting_population, ngen=ngen)

        elif method_name == 'kfold_sample_run':
            sample_size = parameters['sample_size']
            num_trials = parameters['num_trials']

            populations = method( runtime_str, num_trials=num_trials, sample_size=sample_size, modality=modality, dataset_str=dataset_str, 
                                  autoencoder_key=autoencoder_key, target_class=target_class, starting_population=starting_population, ngen=ngen)
        elif method_name == 'cascaded_run':
            populations = method( runtime_str, modalities=modalities, dataset_strs=dataset_strs, autoencoder_keys=autoencoder_keys, 
                                  target_classes=target_classes, starting_population=starting_population, ngens=ngens)
        else:
            print (f"Bad method name: {method_name}")

        misc.elapsedTime()

main()
