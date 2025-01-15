#%% import libraries    
# OpenSoundscape imports
from opensoundscape import Audio, Spectrogram
from opensoundscape.annotations import BoxedAnnotations

# General-purpose packages
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path

from matplotlib import pyplot as plt
plt.rcParams['figure.figsize']=[15,5] #for big visuals

#%%
# Load the data
dataset_path_audio_filesCSV= '/mnt/class_data/group1_bioacoustics/Guy/130_annotations/annotations_130_list_soundFiles.csv'
audio_csv = pd.read_csv(dataset_path_audio_filesCSV)
audio_csv['Sound_files'] = audio_csv['Sound_files'].str.replace('WAV', 'wav')
audio_files_path = '/mnt/class_data/group1_bioacoustics/Guy/annotated_files'
txt_files_path = '/mnt/class_data/group1_bioacoustics/Guy/130_annotations'

annotation_files = txt_files_path+'/'+audio_csv['text_files']
sound_files = audio_files_path+'/'+audio_csv['Sound_files']

#%% warp the annotations and audio
annot_list = annotation_files.to_list()
audio_list = sound_files.to_list()
# %%
# Create an object from Raven file
all_annotations = BoxedAnnotations.from_raven_files(
    annot_list,annotation_column="Annotation",audio_files=audio_list)
all_annotations.df.head(2)


# Inspect the object's .df attribute
# which contains the table of annotations
all_annotations.df.head()
all_annotations.df.annotation.value_counts()
# %%