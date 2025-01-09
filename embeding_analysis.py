#%% import necessary libraries
import pandas as pd
import umap
import seaborn as sns
import matplotlib.pyplot as plt
from tkinter import filedialog
import numpy as np
#%% functions
# function to filter the birds
def bird_filtering(file,birds):
    if len(birds)==1:
        specific_bird=np.repeat(file[birds], 10)
    else:
        specific_bird_num=sum(file[birds].astype(int).values.T)
        specific_bird_single= specific_bird_num.astype(bool)
        specific_bird=np.repeat(specific_bird_single, 10)

    return specific_bird

# visuallize the embeddings function
def visualize_embedings(embedding_df,file,birds):
    # picking a bird
    spcies=birds[0]   
    bird=bird_filtering(file,birds)
    
    plt.figure(figsize=(10, 8))

    # Plot ' no birds' in blue
    sns.scatterplot(
        data=embedding_df[~bird],  # Only birds
        x="UMAP1",
        y="UMAP2",
        color="red",
        label=f"no {spcies}",
        alpha=0.7
    )

    # Plot 'birds' in red
    sns.scatterplot(
        data=embedding_df[bird],  # Only no birds
        x="UMAP1",
        y="UMAP2",
        color="blue",
        label=f"{spcies}",
        alpha=0.7
    )

    plt.title(f"UMAP Visualization of Embeddings for {spcies} (No {spcies} Highlighted)")
    plt.legend(title="Category")
    plt.show()



#%% load the data from the annoations file
input_dir = "/mnt/d/Katarina_annotations/big/" 
file = pd.read_excel("/mnt/c/Users/guyze/OneDrive/Documents/CV4ecology/urbanization_birds/Bird identification_Golan_forGuy.xlsx",
                        sheet_name='10big') 


#%% Creating df with the species names
species_cols = ['species 1', 'species 2', 'species 3', 'species 4', 'species 5', 'species 6', 'species 7']

# Collect all unique species names
unique_species = pd.unique(file[species_cols].values.ravel('K'))
unique_species = [s for s in unique_species if pd.notna(s)]  # Remove NaN values

for species in unique_species:
    file[species] = file[species_cols].apply(lambda row: species in row.values, axis=1)


#%% Load embeddings from CSV

data = pd.read_csv('/home/guyzer/embeddings.csv')

# Apply UMAP for dimensionality reduction
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
embedding = reducer.fit_transform(data)

# Create a DataFrame for visualization
embedding_df = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"])
embedding_df = embedding_df.drop_duplicates().reset_index(drop=True)
#%% Visualization
visualize_embedings(embedding_df,file,['blackbird','Blackbird','blacbird','blackbird','Blackbird','blacbird','blackbird '])
visualize_embedings(embedding_df,file,['chukar ','Chukar ','chukar'])
visualize_embedings(embedding_df,file,['greenfinch'])
visualize_embedings(embedding_df,file,['great tit'])
visualize_embedings(embedding_df,file,['Crested lark','crested lark'])




# %%
