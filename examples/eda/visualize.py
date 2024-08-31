import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist

def parse_data(path):
    npy_path_latents = glob.glob(os.path.join(path, "latents") + "/*.npy")
    npy_path_noise_preds = glob.glob(os.path.join(path, "noise_preds") + "/*.npy")
    
    data_latents = []
    for npy in npy_path_latents:
        data_latents.append(np.load(npy))
    
    data_noise_preds = []
    for npy in npy_path_noise_preds:
        data_noise_preds.append(np.load(npy))
        
    print(len(data_latents), len(data_noise_preds))
    
    #flatten the data, keep first dimension
    data_latents = np.array(data_latents)
    data_noise_preds = np.array(data_noise_preds)

    data_latents = data_latents.reshape(data_latents.shape[0], data_latents.shape[1], -1)
    data_noise_preds = data_noise_preds.reshape(data_noise_preds.shape[0], data_noise_preds.shape[1], -1)
    
    return data_latents, data_noise_preds

church_path = "/scratch/mp5847/eda/church_sd1.4"
church_latents, church_noise_preds = parse_data(church_path)

van_gogh_path = "/scratch/mp5847/eda/van_gogh_sd1.4"
van_gogh_latents, van_gogh_noise_preds = parse_data(van_gogh_path)

thomas_kinkade_path = "/scratch/mp5847/eda/thomas_kinkade_sd1.4"
thomas_kinkade_latents, thomas_kinkade_noise_preds = parse_data(thomas_kinkade_path)

monet_path = "/scratch/mp5847/eda/monet_sd1.4"
monet_latents, monet_noise_preds = parse_data(monet_path)

pos = 10
church_latents_selected = church_latents[:,pos,:]
van_gogh_latents_selected = van_gogh_latents[:,pos,:]
thomas_kinkade_latents_selected = thomas_kinkade_latents[:,pos,:]
monet_latents_selected = monet_latents[:,pos,:]

church_noise_preds_selected = church_noise_preds[:,pos,:]
van_gogh_noise_preds_selected = van_gogh_noise_preds[:,pos,:]
thomas_kinkade_noise_preds_selected = thomas_kinkade_noise_preds[:,pos,:]
monet_noise_preds_selected = monet_noise_preds[:,pos,:]

church_latents_selected = church_noise_preds_selected
van_gogh_latents_selected = van_gogh_noise_preds_selected
thomas_kinkade_latents_selected = thomas_kinkade_noise_preds_selected
monet_latents_selected = monet_noise_preds_selected

'''
latents: (50, 51, 16384)
noise_pred = (50, 51, 16384)
'''
# Compute pairwise distances within each category
church_distances_within = cdist(church_latents_selected, church_latents_selected, metric='euclidean')
van_gogh_distances_within = cdist(van_gogh_latents_selected, van_gogh_latents_selected, metric='euclidean')
thomas_kinkade_distances_within = cdist(thomas_kinkade_latents_selected, thomas_kinkade_latents_selected, metric='euclidean')
monet_latents_distances_within = cdist(monet_latents_selected, monet_latents_selected, metric='euclidean')

# Compute pairwise distances across categories
church_vs_van_gogh_distances = cdist(church_latents_selected, van_gogh_latents_selected, metric='euclidean')
church_vs_thomas_kinkade_distances = cdist(church_latents_selected, thomas_kinkade_latents_selected, metric='euclidean')
church_vs_monet_distances = cdist(church_latents_selected, monet_latents_selected, metric='euclidean')
monet_vs_van_gogh_distances = cdist(monet_latents_selected, van_gogh_latents_selected, metric='euclidean')
monet_vs_thomas_kinkade_distances = cdist(monet_latents_selected, thomas_kinkade_latents_selected, metric='euclidean')
van_gogh_vs_thomas_kinkade_distances = cdist(van_gogh_latents_selected, thomas_kinkade_latents_selected, metric='euclidean')


# Flatten distance matrices to get all pairwise distances as a 1D array
church_distances_within_flat = church_distances_within[np.triu_indices_from(church_distances_within, k=1)]
van_gogh_distances_within_flat = van_gogh_distances_within[np.triu_indices_from(van_gogh_distances_within, k=1)]
thomas_kinkade_distances_within_flat = thomas_kinkade_distances_within[np.triu_indices_from(thomas_kinkade_distances_within, k=1)]
monet_latents_distances_within_flat = monet_latents_distances_within[np.triu_indices_from(monet_latents_distances_within, k=1)]

church_vs_van_gogh_distances_flat = church_vs_van_gogh_distances.flatten()[:len(church_distances_within_flat)]
church_vs_thomas_kinkade_distances_flat = church_vs_thomas_kinkade_distances.flatten()[:len(church_distances_within_flat)]
church_vs_monet_distances_flat = church_vs_monet_distances.flatten()[:len(church_distances_within_flat)]
monet_vs_van_gogh_distances_flat = monet_vs_van_gogh_distances.flatten()[:len(church_distances_within_flat)]
monet_vs_thomas_kinkade_distances_flat = monet_vs_thomas_kinkade_distances.flatten()[:len(church_distances_within_flat)]
van_gogh_vs_thomas_kinkade_distances_flat = van_gogh_vs_thomas_kinkade_distances.flatten()[:len(church_distances_within_flat)]


print("church_distances_within_flat mean and std", np.mean(church_distances_within_flat), np.std(church_distances_within_flat))
print("van_gogh_distances_within_flat mean and std", np.mean(van_gogh_distances_within_flat), np.std(van_gogh_distances_within_flat))
print("thomas_kinkade_distances_within_flat mean and std", np.mean(thomas_kinkade_distances_within_flat), np.std(thomas_kinkade_distances_within_flat))
print("monet_latents_distances_within_flat mean and std", np.mean(monet_latents_distances_within_flat), np.std(monet_latents_distances_within_flat))

print("church_vs_van_gogh_distances_flat mean and std", np.mean(church_vs_van_gogh_distances_flat), np.std(church_vs_van_gogh_distances_flat))
print("church_vs_thomas_kinkade_distances_flat mean and std", np.mean(church_vs_thomas_kinkade_distances_flat), np.std(church_vs_thomas_kinkade_distances_flat))
print("church_vs_monet_distances_flat mean and std", np.mean(church_vs_monet_distances_flat), np.std(church_vs_monet_distances_flat))
print("monet_vs_van_gogh_distances_flat mean and std", np.mean(monet_vs_van_gogh_distances_flat), np.std(monet_vs_van_gogh_distances_flat))
print("monet_vs_thomas_kinkade_distances_flat mean and std", np.mean(monet_vs_thomas_kinkade_distances_flat), np.std(monet_vs_thomas_kinkade_distances_flat))
print("van_gogh_vs_thomas_kinkade_distances_flat mean and std", np.mean(van_gogh_vs_thomas_kinkade_distances_flat), np.std(van_gogh_vs_thomas_kinkade_distances_flat))

#print shape of the distances
print("church_distances_within_flat shape", church_distances_within_flat.shape)
print("van_gogh_distances_within_flat shape", van_gogh_distances_within_flat.shape)
print("thomas_kinkade_distances_within_flat shape", thomas_kinkade_distances_within_flat.shape)
print("monet_latents_distances_within_flat shape", monet_latents_distances_within_flat.shape)

print("church_vs_van_gogh_distances_flat shape", church_vs_van_gogh_distances_flat.shape)
print("church_vs_thomas_kinkade_distances_flat shape", church_vs_thomas_kinkade_distances_flat.shape)
print("church_vs_monet_distances_flat shape", church_vs_monet_distances_flat.shape)
print("monet_vs_van_gogh_distances_flat shape", monet_vs_van_gogh_distances_flat.shape)
