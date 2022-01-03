# Imports and utility functions
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)

import pandas as pd
import librosa
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
import os
import sys
from pathlib import Path
import json
import soundfile as sf
import wave
import ntpath
import sounddevice as sd
import pickle
import glob
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, mutual_info_score
from sklearn.metrics.cluster import normalized_mutual_info_score

# Setup to import custom modules
if not f'{os.path.dirname(os.path.realpath(__file__))}/../scripts' in sys.path:
    sys.path.append(f'{os.path.dirname(os.path.realpath(__file__))}/../scripts')
    

# Load audio datasets and get info based upon filenames
import audiodatasets

# Ease of use with my dictionaries
import dictmgmt

# Other miscellaneous utilities (like the timer)
import misc

# Some important directory paths
jar = f'{os.path.dirname(os.path.realpath(__file__))}/../pickles'
model_dir = f'{os.path.dirname(os.path.realpath(__file__))}/../models'
data_dir = f'{os.path.dirname(os.path.realpath(__file__))}/../data'

#### We expose 4 models: SIMPLE_PURE, SIMPLE_RECORDED, SEA and FLAIR

models = {"SIMPLE_PURE": None, "SIMPLE_RECORDED": None, "SEA": None, "FLAIR": None}

#######################
#SIMPLE CLASSIFIERS
#######################

# Simple feature extraction

def wrapped_simple(filelist):
    X_vals, _, _, y_vals, _, featureindices = extract_features_simple(filelist)
    return X_vals, y_vals, featureindices

def extract_features_simple(filelist, max_duration=4.0, sample_rate=10000):
    
    max_sample_length = sample_rate * max_duration
                 
    y_values_emo = []
    y_values_speaker = []
    X_mfcc = []
    X_zc = []
    X_mf = []

    index = 0
    feature_indices = {}
    for key in sorted(filelist.keys()):
        file = filelist[key]
        x, sr = librosa.load(file, duration=max_duration, sr=sample_rate)
        
        # Pad with zeroes, if necessary
        x = librosa.util.pad_center(x, max_sample_length, mode='constant')
        
        mfcc = librosa.feature.mfcc(x, sr=sr)
        zc = librosa.feature.zero_crossing_rate(x)
        mf = librosa.feature.melspectrogram(x)
        X_mfcc.append(mfcc)
        X_zc.append(zc)
        X_mf.append(mf)

        # Infer the class from the filename
        head, tail = ntpath.split(file)
        y_values_emo.append(audiodatasets.get_emotion_id(tail) - 1)
        y_values_speaker.append(audiodatasets.get_speaker_id(tail) - 1)
        
        # Finally, update the feature indices
        feature_indices[key] = index
        index += 1
        
    X_mfcc = np.stack(X_mfcc)
    X_mfcc = np.reshape(X_mfcc, (np.shape(X_mfcc)[0], np.shape(X_mfcc)[1]*np.shape(X_mfcc)[2]))

    X_zc = np.stack(X_zc)
    X_zc = np.reshape(X_zc, (np.shape(X_zc)[0], np.shape(X_zc)[1]*np.shape(X_zc)[2]))

    X_mf = np.stack(X_mf)
    X_mf = np.reshape(X_mf, (np.shape(X_mf)[0], np.shape(X_mf)[1]*np.shape(X_mf)[2]))

    return X_mfcc, X_zc, X_mf, y_values_emo, y_values_speaker, feature_indices

# SIMPLE MODELS...load or train
ravdess_files = audiodatasets.list_audio_files('{data_dir}/fixed_ravdess')
ravdess_files_acoustic = audiodatasets.list_audio_files('{data_dir}/fixed_ravdess_acoustic')

tess_files = audiodatasets.list_audio_files('{data_dir}/fixed_tess')
tess_files_acoustic = audiodatasets.list_audio_files('{data_dir}/fixed_tess_acoustic')

training_data = { "PURE": {**ravdess_files, **tess_files}, "RECORDED": {**ravdess_files_acoustic, **tess_files_acoustic} }

for key in training_data:

    if os.path.exists(f'{model_dir}/SIMPLE_{key}_EMOTION_CLASSIFIER.h5'):
        models[f'SIMPLE_{key}'] = keras.models.load_model(f'{model_dir}/SIMPLE_{key}_EMOTION_CLASSIFIER.h5')
    else:
    
        X_values, y_values, _ = wrapped_simple(training_data[key])
        num_classes = len(set(y_values))

        models[f'SIMPLE_{key}'] = tf.keras.models.Sequential([tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                        tf.keras.layers.Dense(256, activation=tf.nn.relu), 
                                        tf.keras.layers.Dense(256, activation=tf.nn.relu), 
                                        tf.keras.layers.Dense(256, activation=tf.nn.relu), 
                                        tf.keras.layers.Dense(256, activation=tf.nn.relu), 
                                        tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                        tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)])

        models[f'SIMPLE_{key}'].compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        models[f'SIMPLE_{key}'].fit(X_values, np.stack(y_values), epochs=20)
        models[f'SIMPLE_{key}'].save(f'{model_dir}/SIMPLE_{key}_EMOTION_CLASSIFIER.h5')

# SEA...load
models[f'SEA'] = keras.models.load_model(f'{model_dir}/Speech-Emotion-Analyzer/saved_models/Emotion_Voice_Detection_Model.h5')

def wrapped_sea(filelist):
    X_vals, y_vals, featureindices = extract_features_sea(filelist)

    X_vals = np.expand_dims(X_vals, axis=2)
    
    return X_vals, y_vals, featureindices

# External classifier feature extraction
def extract_features_sea(file_dict):

    features = []
    labels = []
    featureindices = {}
    
    # Integrate 3rd-party feature extraction logic with my data management scheme
    # 3rd Party Source: https://github.com/MITESHPUTHRANNEU/Speech-Emotion-Analyzer/blob/master/final_results_gender_test.ipynb 
    indx = 0
    for key in sorted(file_dict.keys()):
        file = file_dict[key]
       
        # First we will see if this is a valid label for the classifier
        emotion = audiodatasets.get_emotion_id (key)
        if emotion == 2:
            emotion = 'calm'
        elif emotion == 3:
            emotion = 'happy'
        elif emotion == 4:
            emotion = 'sad'
        elif emotion == 5:
            emotion = 'angry'
        elif emotion == 6:
            emotion = 'fearful'
        else:
            emotion = 'INVALID'

        gender = 'female' if audiodatasets.get_speaker_id (key) % 2 == 0 else 'male'

        # If this is a valid emotion for the classifier, continue
        if emotion != 'INVALID':
            labels.append(f'{gender}_{emotion}')
        
            X, sample_rate = librosa.load(file, res_type='kaiser_fast', duration=2.5,sr=22050*2,offset=0.5)
            X = librosa.util.pad_center(X, 2.5*22050*2, mode='constant')
            
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
            
            features.append(mfccs)
            
            # Record the index and increment before the next iteration
            featureindices[key] = indx
            indx += 1
    
    # Finally, convert the lists generated above into np.ndarrays and one-hot encode the labels
    features = np.stack(features)

    lb = LabelEncoder()
    labels_onehot = np.zeros(shape=(np.shape(labels)[0], 10))

    # Match the labels as documented here:
    #       https://github.com/MITESHPUTHRANNEU/Speech-Emotion-Analyzer/blob/master/README.md
    indx = 0
    for label in labels:
        if label == 'female_angry':
            labels_onehot[indx][0] = 1
        elif label == 'female_calm':
            labels_onehot[indx][1] = 1
        elif label == 'female_fearful':
            labels_onehot[indx][2] = 1
        elif label == 'female_happy':
            labels_onehot[indx][3] = 1
        elif label == 'female_sad':
            labels_onehot[indx][4] = 1
        elif label == 'male_angry':
            labels_onehot[indx][5] = 1
        elif label == 'male_calm':
            labels_onehot[indx][6] = 1
        elif label == 'male_fearful':
            labels_onehot[indx][7] = 1
        elif label == 'male_happy':
            labels_onehot[indx][8] = 1
        elif label == 'male_sad':
            labels_onehot[indx][9] = 1
        else:
            print(f'Oops: {label}')

        indx += 1
    
    return features, labels_onehot, featureindices

############################
# Flair
############################
# Train FLAIR
emotions={
  1:'neutral',
  2:'calm',
  3:'happy',
  4:'sad',
  5:'angry',
  6:'fearful',
  7:'disgust',
  8:'surprised'
}
#DataFlair - Emotions to observe
observed_emotions=['calm', 'happy', 'fearful', 'disgust']

def extract_features_flair(file_dict):
    features = []
    labels = []
    featureindices = {}
    
    # Integrate 3rd-party feature extraction logic with my data management scheme
    # 3rd Party Source: https://data-flair.training/blogs/python-mini-project-speech-emotion-recognition/
    indx = 0
    for key in sorted(file_dict.keys()):
        file = file_dict[key]
       
        # First we will see if this is a valid label for the classifier
        emotion = audiodatasets.get_emotion_id(key)

        if emotions[emotion] in observed_emotions:
            
            labels.append(emotion)
            
            # Hack for monotone; not in the original paper, but won't work without this
            Path('/tmp/evaluate').mkdir(parents=True, exist_ok=True)
            x, sr = librosa.load(file_dict[key], mono=True)
            sf.write(f'/tmp/evaluate/foo.wav', x, sr, )
            
            # Reference did not specify if their results were achieved with MFCC, Mel Spectrogram or chroma
            # Tried all 3: MFCC came closest to the reported accuracy
            # Using the same extraction setings as the original code

            with sf.SoundFile('/tmp/evaluate/foo.wav') as sound_file:
                X = sound_file.read(dtype="float32")
                sample_rate=sound_file.samplerate

                mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
                features.append(list(mfccs))
                
#                stft=np.abs(librosa.stft(X))
#                chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
#                for val in chroma:
#                    features[-1].append(val)

#                mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
#                for val in mel:
#                    features[-1].append(val)
    
            # Update our dictionary to tie keys to the indices for the features and labels, increment the index and continue
            featureindices[key] = indx
            indx += 1
    
    # Before we return, convert the lists generated above into np.ndarrays
    features = np.stack(features)
    labels = np.stack(labels)
    
    return features, labels, featureindices

# Wrapped FLAIR feature extraction
def wrapped_flair(filelist):
    X_vals, y_vals, featureindices = extract_features_flair(filelist)
    return X_vals, y_vals, featureindices

# Finally, train Flair if we have not done so already
if os.path.exists(f'{model_dir}/FLAIR_EMOTION_CLASSIFIER.pickle'):
    models["FLAIR"] = pickle.load(open(f'{model_dir}/FLAIR_EMOTION_CLASSIFIER.pickle', 'rb'))
else:
    flair_training_filelist = audiodatasets.list_audio_files('{data_dir}/fixed_ravdess_flair')
    X, y, _ = extract_features_flair(flair_training_filelist)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)
    flair_model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500, random_state=101)
    flair_model.fit(X_train,y_train)
    y_pred = flair_model.predict(X_test)
    accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
    
    models["FLAIR"] = flair_model
    pickle.dump(models["FLAIR"], open(f'{model_dir}/FLAIR_EMOTION_CLASSIFIER.pickle', 'wb'))

#########################################
# DEFINE "label getters"; typically good enough to find misclassifications
#########################################
def get_class_labels_simple(actuals, predicted, indx):
    actual_label = actuals[indx]
    predicted_label = np.argmax(predicted[indx])
    
    return actual_label, predicted_label

def get_class_labels_sea(actuals, predicted, indx):
    actual_label = np.argmax(actuals[indx])
    predicted_label = np.argmax(predicted[indx])
    
    return actual_label, predicted_label

def get_class_labels_flair(actuals, predicted, indx):
    actual_label = actuals[indx]
    predicted_label = predicted[indx]
    
    return actual_label, predicted_label

# Add them to a list


#########################################
# Simple getters for the calling program
##########################################
classifier_names = ["SIMPLE_PURE", "SIMPLE_RECORDED", "SEA", "FLAIR"]

class_names = {
                    "SIMPLE_PURE": ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised'],

                    "SIMPLE_RECORDED": ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised'],

                    "SEA": ['female_angry', 'female_calm', 'female_fearful', 'female_happy', 'female_sad', 
                            'male_angry', 'male_calm', 'male_fearful', 'male_happy', 'male_sad'],
    
                    "FLAIR": ['', 'neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
}

feature_extractors = { "SIMPLE_PURE": wrapped_simple, 
                       "SIMPLE_RECORDED": wrapped_simple, 
                       "SEA": wrapped_sea, 
                       "FLAIR": wrapped_flair
                     }

label_getters = {"SIMPLE_PURE": get_class_labels_simple, 
                 "SIMPLE_RECORDED": get_class_labels_simple, 
                 "SEA": get_class_labels_sea, 
                 "FLAIR": get_class_labels_flair
                }     


        
