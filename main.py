#Importing vital frameworks
import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

from ultralytics import YOLO, hub

#Defining YOLO model
model = YOLO('yolov8s')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

