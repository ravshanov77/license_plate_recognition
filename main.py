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


#Preparing the video
video = 'single_car.mp4'

cap = cv2.VideoCapture(video)

if not cap.isOpened():
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow('Video', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#Detecting the cars
car_detect = model(cap)