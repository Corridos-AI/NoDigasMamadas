import numpy as np 
import pandas as pd 
import os
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import random
import tensorflow_hub as hub

import pandas as pd
import numpy as np
import random

import matplotlib.pyplot as plt

import tensorflow as tf
#from tensorflow.keras.applications import InceptionV3


# from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Input, GlobalAveragePooling2D
from keras import backend as K
from keras import applications
from keras.utils import plot_model


import keras_tuner as kt

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from tqdm.auto import tqdm
import shutil as sh

import matplotlib.pyplot as plt
from IPython.display import Image, clear_output

import os
import shutil
from xml.etree import ElementTree as ET
import cv2
import xml.etree.ElementTree as ET
from PIL import Image
import glob

import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torch.optim import SGD
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

print('Tensorflow version:', tf.__version__)

import yaml
with open('config2.yaml', 'r') as file:
    config = yaml.safe_load(file)

train_path = config['train']
train_labels_path = config['train_labels']
val_path = config['val']
test_path = config['test']
num_classes = config['nc']
class_names = config['names']

transform = transforms.Compose([
    transforms.ToTensor()
])

train_data_path = train_path
train_label_file = train_labels_path

train_dataset = CocoDetection(root=train_path, annFile=train_labels_path, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)

backbone = torchvision.models.mobilenet_v2(pretrained=True).features
backbone.out_channels = 1280  # Set the number of output channels
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                output_size=7,
                                                sampling_ratio=2)
model = FasterRCNN(backbone,
                   num_classes=91,  # COCO dataset has 91 classes
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)

# Optionally, modify the model's head
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=91)

# Define the optimizer and learning rate scheduler
params = [p for p in model.parameters() if p.requires_grad]
optimizer = SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for images, targets in train_loader:
        images = list(image for image in images)
        targets = [{k: v for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    # Update the learning rate
    lr_scheduler.step()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {losses}")

# Save the trained model
torch.save(model.state_dict(), 'faster_rcnn_model.pth')