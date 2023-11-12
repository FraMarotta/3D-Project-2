# Authors: Simone Maravigna, Francesco Marotta
# Date: 2023-11
# Project: 3D Object Detection on nuScenes Dataset
# File: annotations_fusion.py
# Description: This file contains the code to fuse the annotations from the 3D object detection taken by Lidar and the 2D object detection taken by Camera.

#-------------------------------------------
# imports
#-------------------------------------------
import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import rcParams
import json
from nuscenes import NuScenes
from pyquaternion import Quaternion
from loader2D import get_id_dict_rev
import os
#-------------------------------------------
# helper functions
#-------------------------------------------

def create_box(sample_data_token, ann_rec):
    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    data_path = nusc.get_sample_data_path(sample_data_token)

    if sensor_record['modality'] == 'camera':
            cam_intrinsic = np.array(cs_record['camera_intrinsic'])
            imsize = (sd_record['width'], sd_record['height'])
    else:
        cam_intrinsic = None
        imsize = None
    
    box = Box(ann_rec['translation'], ann_rec['size'], Quaternion(ann_rec['rotation']), name=ann_rec['detection_name']) 
    # Move box to ego vehicle coord system.
    box.translate(-np.array(pose_record['translation']))
    box.rotate(Quaternion(pose_record['rotation']).inverse)

    #  Move box to sensor coord system.
    box.translate(-np.array(cs_record['translation']))
    box.rotate(Quaternion(cs_record['rotation']).inverse)
    
    if sensor_record['modality'] == 'camera' and not \
            box_in_image(box, cam_intrinsic, imsize, vis_level=BoxVisibility.ALL):
        return None, None, None
 
    return data_path, box, cam_intrinsic

#-------------------------------------------
# main
#-------------------------------------------

# open dataset and json files
nusc = NuScenes(version='v1.0-mini', dataroot='data/sets/nuscenes', verbose=False)
lidar_pred_file = open('results_nusc.json', 'r')
lidar_pred_data = json.load(lidar_pred_file)
camera_pred_file = open('results_fasterRCNN.json', 'r')
camera_pred_data = json.load(camera_pred_file)
margin = 50
id_dict_rev = get_id_dict_rev()
# iterate over all samples
for i, d in enumerate(lidar_pred_data['results']):
    scene = nusc.get('scene', nusc.get('sample', str(d))['scene_token'])['name']
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    # plot lidar predictions
    # iterate over all predictions for the current sample
    for ann_rec in lidar_pred_data['results'][str(d)]:
        sample_record = nusc.get('sample', str(d))
        if(ann_rec['detection_score'] < 0.4):
            continue
        _, box, _ = create_box(sample_record['data']['CAM_FRONT'], ann_rec)
        if box is None:
            continue
        cam = sample_record['data']['CAM_FRONT']

        # Plot LIDAR view.
        lidar = sample_record['data']['LIDAR_TOP']
        data_path, box, camera_intrinsic = create_box(lidar, ann_rec)
        LidarPointCloud.from_file(data_path).render_height(axes[0], view=np.eye(4))
        box.render(axes[0], view=np.eye(4), colors=('b', 'b', 'b'), linewidth=2)
        corners = view_points(box.corners(), np.eye(4), False)[:2, :]
        axes[0].set_xlim([np.min(corners[0, :]) - margin, np.max(corners[0, :]) + margin])
        axes[0].set_ylim([np.min(corners[1, :]) - margin, np.max(corners[1, :]) + margin])
        axes[0].axis('off')
        axes[0].set_aspect('equal')
        axes[0].set_title('LIDAR view')
        axes[0].text(corners[0, 0], corners[1, 0], ann_rec['detection_name'], color='b', fontsize=8)
        
        # Plot CAMERA view.
        data_path, box, camera_intrinsic = create_box(cam, ann_rec)
        im = Image.open(data_path)
        axes[1].imshow(im)
        axes[1].set_title(nusc.get('sample_data', cam)['channel'])
        axes[1].axis('off')
        axes[1].set_aspect('equal')
        box.render(axes[1], view=camera_intrinsic, normalize=True, colors=('b', 'b', 'b'), linewidth= 1)
        corners = view_points(box.corners(), camera_intrinsic, True)[:2, :]
        axes[1].text(corners[0, 0], corners[1, 0], ann_rec['detection_name'], color='b', fontsize=8)
    
    # plot 2D camera predictions
    sample_prediction = camera_pred_data[str(d)]
    for j, box in enumerate(sample_prediction['boxes']):
        if sample_prediction['scores'][j] < 0.6:
            break #break because scores are in descending order
        axes[1].add_patch(plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], fill=False, edgecolor='r', linewidth=1))
        axes[1].text(box[0], box[1], id_dict_rev[sample_prediction['labels'][j]].split('.')[-1], color='r', fontsize=8)
    
    # save figure
    output_dir = 'output/'+ str(scene)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_dir +'/eval_' + str(i)+ '_' + str(d) +'.png')


# close json file
lidar_pred_file.close()
camera_pred_file.close()