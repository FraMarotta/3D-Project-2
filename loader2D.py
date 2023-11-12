# Authors: Simone Maravigna, Francesco Marotta
# Date: 2023-11
# Project: 3D Object Detection on nuScenes Dataset
# File: loader2D.py
# Description: This file contains the code to load the nuscene dataset for 2D object detection.

# --- Imports --- #
import torch
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
import numpy as np
# --- Functions --- #

#defining the collate function
def collate_fn(batch):
    return tuple(zip(*batch))

#function to get a dictionary class->number
def get_id_dict():
    id_dict = {}
    #eg, id_dict['animal']=1, id_dict['human.pedestrian.adult']=2, etc 0 is background
    for i, line in enumerate(open('classes.txt', 'r')):
        id_dict[line.replace('\n', '')] = i+1 #creating matches class->number
    return id_dict

#function to get a dictionary number->class
def get_id_dict_rev():
    id_dict_rev = {}
    #eg, id_dict_rev[1]='animal', id_dict_rev[2]='human.pedestrian.adult', etc 0 is background
    for i, line in enumerate(open('classes.txt', 'r')):
        id_dict_rev[i+1] = line.replace('\n', '') #creating matches number->class
    return id_dict_rev

#--- Classes --- #

#defining the dataset class
class NuScenesDataset(Dataset):
    #HP: camera front plus lidar
    def __init__(self, root, id_dict, version='mini'):
        self.root = root
        #self.filenames = []
        self.front_tokens = []
        self.lidar_tokens = []
        self.id_dict = id_dict
        self.nusc = NuScenes(version='v1.0-'+version, dataroot=root, verbose=True)
        #now we need to build the lists of filenames and tokens
        #iterate over the samples
        for i, sample in  enumerate(self.nusc.sample):
            #pick the token of the sample in analysis
            sample_token = sample['token']
            #get the list of sample_data tokens that have sample_token = sample_token
            sample_data_tokens = []
            for sample_data in self.nusc.sample_data:
                if sample_data['sample_token'] == sample_token:
                    sample_data_tokens.append(sample_data['token'])
            #iterate over the sample_data tokens searching for keyframes
            keyframes = []
            for tk in sample_data_tokens:
                #get the instance of sample_data that has token = tk
                sample_data_instance = self.nusc.get('sample_data', tk)
                #check of it is a keyframe
                if sample_data_instance['is_key_frame'] :
                    keyframes.append(tk)
            #now we need to iterate over the keyframes
            for tk in keyframes:
                sample_data_instance = self.nusc.get('sample_data', tk)
                #get the sensor modality navigating the tables
                sensor_channel = self.nusc.get('sensor', self.nusc.get('calibrated_sensor', sample_data_instance['calibrated_sensor_token'])['sensor_token'])['channel']
                #check if the sensor is a front camera
                if sensor_channel == 'CAM_FRONT':
                    #self.filenames.append(sample_data_instance['filename'])
                    self.front_tokens.append(tk)
                #check if the sensor is a lidar
                elif sensor_channel == 'LIDAR_TOP':
                    self.lidar_tokens.append(tk)



    def __getitem__(self, idx):
        img_token = self.front_tokens[idx]
        lidar_token = self.lidar_tokens[idx]
        #get the filename of the image
        img_filename = self.nusc.get_sample_data_path(img_token)
        #get the filename of the lidar
        lidar_filename = self.nusc.get_sample_data_path(lidar_token)
        #read the image as tensor and normalize it
        img = read_image(img_filename, ImageReadMode.RGB)/255
        #read the lidar 
        lidar = LidarPointCloud.from_file(lidar_filename)
        #get the sample token of the image
        sample_token = self.nusc.get('sample_data', img_token)['sample_token']
        #get the annotations tokens of the image
        annotations = []
        data = []
        for ann in self.nusc.sample_annotation:
            if ann['sample_token'] == sample_token:
                annotations.append(ann['token'])
        #in annotations there are all the sample_annotations tokens of the image
        for ann in annotations:
            #get the label of the annotation navigating the tables
            label = self.nusc.get('category', self.nusc.get('instance', self.nusc.get('sample_annotation', ann)['instance_token'])['category_token'])['name']
            #get the bbox of the annotation
            bbox = self.nusc.get('image_annotation', ann)['bbox_corners']
            #get the class of the annotation
            cl = self.id_dict[label]
            #append the data to the list
            data.append({
            'bbox': bbox,
            'category_id': cl,
            'category_name': label
            })

        if len(data) == 0:
            data.append({
            'bbox': [0,0,0,0,0.1,0.1,0.1,0.1],
            'category_id': 0,
            'category_name': 'void'
            })
        
        # put boxes and labels into tensors
        boxes = torch.Tensor(np.array([d['bbox'] for d in data]))
        labels = torch.Tensor(np.array([d['category_id'] for d in data]))
        tokens = [img_token, lidar_token]
        #return img, lidar, boxes, labels, tokens
        return img, boxes, labels

    def __len__(self):
        return len(self.front_tokens)
