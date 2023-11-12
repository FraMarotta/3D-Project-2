# Authors: Simone Maravigna, Francesco Marotta
# Date: 2023-11
# Project: 3D Object Detection on nuScenes Dataset
# File: eval.py
# Description: This file contains the code to evaluate the model (2D predictions on camera images) on the validation set and to save the results in a json file.

#-------------------------------------------
# libraries
#-------------------------------------------
import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import os
#from PIL import Image, ImageDraw, ImageFont
from loader2D import NuScenesDataset, collate_fn, get_id_dict, get_id_dict_rev
import json
#-------------------------------------------
# hyperparameters
#-------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#-------------------------------------------
# dataset and dataloader
# ------------------------------------------

id_dict = get_id_dict()
id_dict_rev = get_id_dict_rev()
val_dataset = NuScenesDataset('./data/sets/nuscenes', id_dict=id_dict, version='mini')
val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_fn)

#-------------------------------------------
# model
# ------------------------------------------

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
num_classes = 24 # 23 classes + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

model.to(device)

model.load_state_dict(torch.load("./checkpoints/checkpoint_epoch9_2023-10-23_01-46-43.pth"))

mAP = MeanAveragePrecision(class_metrics=True).to(device)

#-------------------------------------------
# evaluation
# ------------------------------------------

print('Start evaluation...')

valaccuracy = 0

model.eval()
with torch.no_grad():
    for i, data in enumerate(val_loader):
        images, boxes, labels, token = data

        images = list((image/255.0).to(device) for image in images)

        targets = []
        for im in range(len(images)):
            d = {}
            d['boxes'] = boxes[im].to(device)
            d['labels'] = labels[im].to(device)
            targets.append(d)
        output = model(images)

        mAP.update(output, targets)
        # Check if the file exists
        if not os.path.isfile('results_fasterRCNN.json'):
            with open('results_fasterRCNN.json', 'w') as f:
                json.dump({}, f)
        with open('results_fasterRCNN.json', 'r') as f:
            try:
                data = json.load(f)
            except ValueError:
                data = {}
        # Write the results to the file
        for j in range(len(output)):
            output[j]['boxes'] = output[j]['boxes'].tolist()
            output[j]['labels'] = output[j]['labels'].tolist()
            output[j]['scores'] = output[j]['scores'].tolist()
            data[token[0]] = output[j]
        # Save the results to the file
        with open('results_fasterRCNN.json', 'w') as f:
            json.dump(data, f)

mean_ap = mAP.compute()
print(mean_ap)
# print accuracy per class
i=0
print("{:<25} {:<25}".format('Class', 'AP'))  # Header
for j in range(1,24):
    if i < len(mean_ap['classes']) and j == mean_ap['classes'][i].item():
        print("{:<25} {:<25}".format(id_dict_rev[j].split('.')[-1], str(round(mean_ap['map_per_class'][i].item(), 3))))
        i += 1
    else:
        print("{:<25} {:<25}".format(id_dict_rev[j].split('.')[-1], '0.0'))
   
print('Val accuracy: {}'.format(mean_ap['map'].item()))
