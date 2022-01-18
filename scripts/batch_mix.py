################################################
#
# Purpose: Take previously recorded perturbations and digitally mix with the target dataset
# Inputs:  JSON file containing:
#              - pickle file with perturbation datastructure 
#              - dataset details
# Outputs: pickle of:
#              - TODO: pickle with results of digitally mixing every perturbation with the target dataset (needs lots of RAM)
#              - pickle containing the eval results: SUCCESSFUL_EVASION or FAILED_MISCLASSIFICATION or BAD_TRANSCRIPT
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
import uuid
import shutil
import gc

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

# Premix
def dig_mix_to_file(filedict, key, perturbation, tempdir='/tmp/letstest4'):
    Path(tempdir).mkdir(parents=True, exist_ok=True)
    
    # Perform the mix
    result = np.zeros(40000)
    
    a, _ = librosa.load(filedict[key], sr=10000)
    
    for x in range(min(40000, len(a))):
        result[x] += a[x]

    for x in range(min(40000, len(perturbation))):
        result[x] += perturbation[x]
    
    # Write to file and return the filename
    path = os.path.join(tempdir, key)
    sf.write(path, result, 10000)
    
    return path

# Main function
def main():

    if not (len(sys.argv) > 1 and os.path.exists(sys.argv[1])):
        print(sys.argv[1])
        print(f"usage: {sys.argv[0]} <path to config file>") 
        sys.exit(1)
        
    config = json.load(open(sys.argv[1], 'r'))

#    {
#    "job0": {
#        "input_pickle": "<path to pickle>",
#        "dataset_str": "TESS",
#        "mix_eval_filename": "<path to output>",
#        "params": {
#            "sample_sz_strs": [],
#            "autoencoder_keys": [],
#            "trial_numbers": [],
#            "population_indices": [5,10],
#            "perturbation_indices": [], 
#            "loudnesses": [17,16,14,12,9] 
#        }
#    }
#    }

    # Each key in config is a "job"
    for job in config.values():
        misc.resetTimer()
        print("=============================================================")
        print("Starting job:")
        print(job)
        print("=============================================================")


        # Load the config
        input_pickle = job['input_pickle']
        dataset_str = job['dataset_str']
        mix_eval_filename = job['mix_eval_filename']
        
        parameters = job['params']
        sample_sz_strs = parameters['sample_sz_strs']
        autoencoder_keys = parameters['autoencoder_keys']
        trial_numbers = parameters['trial_numbers']
        population_indices = parameters['population_indices']
        perturbation_indices = parameters['perturbation_indices']
        loudnesses = parameters['loudnesses']

        # Load the input pickle, Format:
        # dictionary[sample_sz_strings][autoencoder string][trial_num][index_from_orig_populations_ds][pert_indx][loudness]
        perturbations = pickle.load(open(input_pickle, 'rb'))
        
        # Load the dataset 
        dataset_files = audiodatasets.list_audio_files(f'{data_dir}/fixed_{dataset_str.lower()}_acoustic')
#        fornow = random.sample(list(dataset_files), 10)
#        dataset_files = { x: dataset_files[x] for x in fornow}
        
        
        # Now, define a directory for temporary files to use for mixing our data
        temp_dirs = []

        # Digitally mix all of the dataset files with the perturbations
        dig_mix = {}
 
        for sample_sz in [x for x in perturbations.keys() if len(sample_sz_strs) == 0 or x in sample_sz_strs]:
            for ae_str in [x for x in perturbations[sample_sz].keys() if len(autoencoder_keys) == 0 or x in autoencoder_keys]:
                for trial_num in [x for x in perturbations[sample_sz][ae_str].keys() if len(trial_numbers) == 0 or x in trial_numbers]:
                    print("++++++++++++++++++++++++++++++++++TRIAL #", trial_num)
                    for pop_indx in [x for x in perturbations[sample_sz][ae_str][trial_num].keys() if len(population_indices) == 0 or x in population_indices]:
                        for pert_indx in [x for x in perturbations[sample_sz][ae_str][trial_num][pop_indx].keys() if len(perturbation_indices) == 0 or x in perturbation_indices]:
                            for loudness in [x for x in perturbations[sample_sz][ae_str][trial_num][pop_indx][pert_indx].keys() if len(loudnesses) == 0 or x in loudnesses]:
                                temp_dirs.append(f'/tmp/bpt_{uuid.uuid1()}')
                                Path(temp_dirs[-1]).mkdir(parents=True, exist_ok=True)

                                for audio_key in dataset_files.keys():
                                    dictmgmt.dictset(dig_mix, [sample_sz, ae_str, trial_num, pop_indx, pert_indx, loudness, audio_key], 
                                                     dig_mix_to_file(dataset_files, audio_key, perturbations[sample_sz][ae_str][trial_num][pop_indx][pert_indx][loudness], 
                                                                    tempdir=temp_dirs[-1]))

        # Evaluate the perturbed audio files
        misc.resetTimer()
        dig_premix_results = {}
        if os.path.exists(mix_eval_filename):
            dig_premix_results = pickle.load(open(mix_eval_filename, 'rb'))
        
        classifier = classifiers.models["SIMPLE_RECORDED"]
        feature_extractor = classifiers.feature_extractors["SIMPLE_RECORDED"]
        label_getter = classifiers.label_getters["SIMPLE_RECORDED"]

        SUCCESSFUL_EVASION = 0
        FAILED_MISCLASSIFICATION = 1
        BAD_TRANSCRIPT = 2        

        for sample_sz in [x for x in perturbations.keys() if len(sample_sz_strs) == 0 or x in sample_sz_strs]:
            for ae_str in [x for x in perturbations[sample_sz].keys() if len(autoencoder_keys) == 0 or x in autoencoder_keys]:
                for trial_num in [x for x in perturbations[sample_sz][ae_str].keys() if len(trial_numbers) == 0 or x in trial_numbers]:
                    for pop_indx in [x for x in perturbations[sample_sz][ae_str][trial_num].keys() if len(population_indices) == 0 or x in population_indices]:
                        for pert_indx in [x for x in perturbations[sample_sz][ae_str][trial_num][pop_indx].keys() if len(perturbation_indices) == 0 or x in perturbation_indices]:
                            for loudness in [x for x in perturbations[sample_sz][ae_str][trial_num][pop_indx][pert_indx].keys() if len(loudnesses) == 0 or x in loudnesses]:

                                X, y, fi = feature_extractor({audio_key: dig_mix[sample_sz][ae_str][trial_num][pop_indx][pert_indx][loudness][audio_key] \
                                                      for audio_key in dig_mix[sample_sz][ae_str][trial_num][pop_indx][pert_indx][loudness]})
                                pred = classifier.predict(X)
                                for audio_key in fi.keys():
                                    a, b = label_getter(y, pred, fi[audio_key])
                                    if a == b:
                                        dictmgmt.dictset(dig_premix_results, [sample_sz, ae_str, trial_num, pop_indx, pert_indx, loudness, audio_key], FAILED_MISCLASSIFICATION)
                                    else:
                                        t, c = transcription.extract_text({audio_key: dig_mix[sample_sz][ae_str][trial_num][pop_indx][pert_indx][loudness][audio_key]})
                                        if transcription.valid_transcript(t[audio_key], c[audio_key]):
                                            dictmgmt.dictset(dig_premix_results, [sample_sz, ae_str, trial_num, pop_indx, pert_indx, loudness, audio_key], SUCCESSFUL_EVASION)
                                        else:
                                            dictmgmt.dictset(dig_premix_results, [sample_sz, ae_str, trial_num, pop_indx, pert_indx, loudness, audio_key], BAD_TRANSCRIPT)


                                # Now just export the results
                                print(f"Finished {sample_sz} {ae_str} {trial_num} {pop_indx} {pert_indx} {loudness}")
                                misc.elapsedTime()
                                pickle.dump(dig_premix_results, open(mix_eval_filename, 'wb'))
                                            
        # Cleanup                           
        for tdir in temp_dirs:
            shutil.rmtree(tdir)
main()
