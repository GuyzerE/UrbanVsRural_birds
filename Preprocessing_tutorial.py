#%% import necessary packages
from opensoundscape.preprocess.preprocessors import SpectrogramPreprocessor
from opensoundscape.ml.datasets import AudioFileDataset, AudioSplittingDataset
from opensoundscape import preprocess
from opensoundscape.annotations import BoxedAnnotations
from sklearn.model_selection import train_test_split
from opensoundscape.preprocess.utils import show_tensor, show_tensor_grid
from opensoundscape.preprocess.utils import show_tensor_grid



#other utilities and packages
import torch
import pandas as pd
from pathlib import Path
import numpy as np
import random
import subprocess
import IPython.display as ipd
from matplotlib import pyplot as plt

#%% seeding
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
#%% load data
# Load the data
dataset_path_audio_filesCSV= '/mnt/class_data/group1_bioacoustics/Guy/130_annotations/annotations_130_list_soundFiles.csv'
audio_csv = pd.read_csv(dataset_path_audio_filesCSV)
audio_csv['Sound_files'] = audio_csv['Sound_files'].str.replace('WAV', 'wav')
audio_files_path = '/mnt/class_data/group1_bioacoustics/Guy/annotated_files'
txt_files_path = '/mnt/class_data/group1_bioacoustics/Guy/130_annotations'

annotation_files = txt_files_path+'/'+audio_csv['text_files']
sound_files = audio_files_path+'/'+audio_csv['Sound_files']

#%% warp the annotations and audio as bounding boxes
annot_list = annotation_files.to_list()
audio_list = sound_files.to_list()

annotations = BoxedAnnotations.from_raven_files(
    annot_list,annotation_column="Annotation",audio_files=audio_list)
annotations.df.head(2)

#%% clip the audio files

# Parameters to use for label creation
clip_duration = 30
clip_overlap = 0
min_label_overlap = 0.25
# species_of_interest = ["Sparrow", "Bulbul", "Unknown"]

# Create dataframe of one-hot labels
labels = annotations.clip_labels(
    clip_duration = clip_duration,
    clip_overlap = clip_overlap,
    min_label_overlap = min_label_overlap,
#    class_subset = species_of_interest # You can comment this line out if you want to include all species.
)


#%% applying the preprocessor on the clipped data
preprocessor = SpectrogramPreprocessor(sample_duration=2.0)
dataset = AudioFileDataset(labels,preprocessor)

dataset.bypass_augmentations = True

tensors = [dataset[i].data for i in range(33)]
sample_labels = [list(dataset[i].labels[dataset[i].labels>0].index) for i in range(33)]

_ = show_tensor_grid(tensors,3,labels=sample_labels)


#%% applying the preprocessing on the raw data +splitting the data
preprocessor = SpectrogramPreprocessor(sample_duration=2.0)
prediction_df = pd.DataFrame(index=audio_list)
splitting_dataset = AudioSplittingDataset(prediction_df,preprocessor,clip_overlap_fraction=0.2)
splitting_dataset.bypass_augmentations = True

#get the first 9 samples and plot them
tensors = [splitting_dataset[i].data for i in range(9)]

_ = show_tensor_grid(tensors,3)

# %% inspect the current pipeline (ordered sequence of Actions to take)
preprocessor = SpectrogramPreprocessor(sample_duration=2)
preprocessor.pipeline
preprocessor.pipeline.to_spec.params
# %% changing the "pipeline"
preprocessor.pipeline.to_spec.set(dB_scale=False)
preprocessor.pipeline.to_spec.params.window_samples = 512
preprocessor.pipeline.to_spec.params['overlap_fraction'] = 0.25

preprocessor.pipeline.to_spec.params
preprocessor.pipeline.add_noise.bypass=True
preprocessor.pipeline.time_mask.bypass=True
preprocessor.pipeline.frequency_mask.bypass=True
# %%
preprocessor = SpectrogramPreprocessor(sample_duration=2)
preprocessor.pipeline.load_audio.bypass
preprocessor.pipeline.frequency_mask.bypass
dataset = AudioFileDataset(labels,preprocessor)

print('random affine off')
preprocessor.pipeline.random_affine.bypass = True
show_tensor(dataset[0].data,transform_from_zero_centered=True)
plt.show()

print('random affine on')
preprocessor.pipeline.random_affine.bypass = False
show_tensor(dataset[0].data,transform_from_zero_centered=True)
# %% rewrite the preprocessing 
#initialize a preprocessor 
preprocessor = SpectrogramPreprocessor(2.0)

#overwrite the pipeline with a slice of the original pipeline
print('\nnew pipeline:')
preprocessor.pipeline = preprocessor.pipeline[0:4]
[print(p) for p in preprocessor.pipeline]

print('\nWe now have a preprocessor that returns Spectrograms instead of Tensors:')
dataset = AudioFileDataset(labels,preprocessor)
print(f"Type of returned sample: {type(dataset[0].data)}")
dataset[0].data.plot()




# %%
pre = SpectrogramPreprocessor(sample_duration=2)
# pre.pipeline.load_audio.set(sample_rate=24000) # change sample rate


#load data set
dataset = AudioFileDataset(labels,pre)
dataset.bypass_augmentations=True

print('default parameters:')
show_tensor(dataset[21].data)
plt.show()

# change window size
print('chaning params:')
dataset.preprocessor.pipeline.to_spec.set(window_samples=512) # change window size

# show_tensor(dataset[14].data)

# show_tensor(dataset[14].data,invert=True,transform_from_zero_centered=True)

# dataset.preprocessor.pipeline.bandpass.set(min_f=100,max_f=4000) # change bandpass filter
# dataset.preprocessor.height = 100
# dataset.preprocessor.width = 200
# dataset.preprocessor.channels = 3

# dataset.preprocessor.pipeline.to_tensor.params.range = (-70,-50) # change range
# show_tensor(dataset[21].data)

# tensors = [dataset[i].data for i in range(33)]
# sample_labels = [list(dataset[i].labels[dataset[i].labels>0].index) for i in range(33)]

# _ = show_tensor_grid(tensors,3,labels=sample_labels)
dataset.preprocessor.pipeline.to_tensor.params.range = (-70,-50) # change range

dataset.preprocessor.pipeline.overlay.set(
    overlay_class='present',
    overlay_weight=0.1
)
show_tensor(dataset[21].data)

# %%
# %%
# normaliz the data
test = (dataset[21].data)/dataset[21].data.max()
show_tensor(test)
