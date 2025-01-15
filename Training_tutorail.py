#%% import the relevant libraries
#from opensoundscape 
from opensoundscape import CNN
from opensoundscape.annotations import BoxedAnnotations
from sklearn.model_selection import train_test_split
from opensoundscape.data_selection import resample

#other utilities and packages
import torch
import pandas as pd
from pathlib import Path
import numpy as np
import pandas as pd
import random
import subprocess
from glob import glob
import sklearn
from matplotlib import pyplot as plt
import wandb # weight and bias
#%%
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
#%%
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
clip_labels = annotations.clip_labels(
    clip_duration = clip_duration,
    clip_overlap = clip_overlap,
    min_label_overlap = min_label_overlap,
#    class_subset = species_of_interest # You can comment this line out if you want to include all species.
)

# %% 
mask = np.ones((33,1), dtype=bool)
indices_to_set_true = [11, 14,3, 31] 
mask[ indices_to_set_true,0] = False
test_set = clip_labels[mask]
# All other files will be used as a training set
train_and_val_set = clip_labels.drop(test_set.index)

train_df, valid_df = sklearn.model_selection.train_test_split(train_and_val_set, test_size=0.00001, random_state=0)

#%% Save .csv tables of the training and validation sets to keep a record of them
# train_and_val_set.to_csv("./annotated_data/train_and_val_set.csv")
# test_set.to_csv("./annotated_data/test_set.csv")

# %% creating the model

balanced_train_df = resample(train_df,n_samples_per_class=800,random_state=0)
# Use resnet34 architecture
architecture = 'resnet34'

# Can use this code to get your classes, if needed
class_list = list(train_df.columns)

model = CNN(
    architecture = architecture,
    classes = class_list,
    sample_duration = clip_duration #3s, selected above
)

model.preprocessor.pipeline.to_tensor.params.range = (-70,-50)
# %% weights and biases
# wandb.login()
wandb_session = wandb.init(
    entity='guyzer1123-tel-aviv-university',
    project='First training',
    name='Train CNN',
    settings=wandb.Settings(init_timeout=120)  
)


# %% Training the model
checkpoint_folder = Path("model_training_checkpoints_tutorial")
checkpoint_folder.mkdir(exist_ok=True)

model.train(
    balanced_train_df, 
    valid_df, 
    epochs = 10, 
    batch_size = 64, 
    log_interval = 100, #log progress every 100 batches
    num_workers = 32, #parallelized cpu tasks for preprocessing
    wandb_session = wandb_session,
    save_interval = 10, #save checkpoint every 10 epochs
    save_path = checkpoint_folder #location to save checkpoints
)

#%% post training 
# scores_df = model.predict(valid_df.head(),activation_layer='sigmoid')