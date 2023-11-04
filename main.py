# Authors: Simone Maravigna, Francesco Marotta
# Date: 2023-11
# Project: 3D Object Detection on nuScenes Dataset
# File: main.py
# Description: This file contains the main code and for now it is just a test.

# --- Imports --- #
import torch
from nuscenes.nuscenes import NuScenes
from nuscenes.nuscenes import NuScenesExplorer
from loader import NuScenesDataset, collate_fn, get_id_dict, get_id_dict_rev
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.axes as Axes
# --- Functions --- #

# --- Main --- #
nusc = NuScenes(version='v1.0-mini', dataroot='data/sets/nuscenes', verbose=False)
#defining the dataset
dataset = NuScenesDataset(root='data/sets/nuscenes', id_dict=get_id_dict(), version='mini')
#defining the dataloader
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
#iterating over the dataset
explorer = NuScenesExplorer(nusc)
for i, data in enumerate(dataloader):
    #projecting the pointcloud onto the image, the result is a tuple of 3 elements: points, coloring and image 
    points, coloring, im = NuScenesExplorer.map_pointcloud_to_image(explorer, data[-1][0][1], data[-1][0][0], render_intensity=True)
    print(i)
    print(points, coloring, im)  #this is just to check the output, to be removed
    #getting the sample token
    sample_token = nusc.get('sample_data', data[-1][0][1])['sample_token']

    #plotting the image with the points
    #comment the following lines if you don't want to see the images
    ax: Axes = None
    fig, ax = plt.subplots(1, 1, figsize=(9, 16))
    fig.canvas.set_window_title(sample_token)
    ax.imshow(im)
    ax.scatter(points[0, :], points[1, :], c=coloring, s=5)
    ax.axis('off')
    plt.show() 

    #stopping the loop after 10 iterations, just to test
    if i==10:
        break