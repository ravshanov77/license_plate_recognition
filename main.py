#Importing vital frameworks
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from drawing_car_bbox import *
from ultralytics import YOLO

#Defining YOLO model
model = YOLO('yolov8s')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


#Preparing the video
video = 'test_resources/single_car.mp4'

cap = cv2.VideoCapture(video)

if not cap.isOpened():
    exit()

while True:
    ret, frame = cap.read()
    frame = np.asarray(frame)
    print(type(frame))
    car_detect = model(frame)

    # separating only cars from the predictions
    cars = car_detect[0].boxes[car_detect[0].boxes.cls==2.]

    # drawing bbox for a car
    vid1 = draw_bbox(frame, labels=cars.xyxy)

    # Detecting the licese plate
    model2 = YOLO('models\licence.pt')

    license_detect = model2(frame)

    img2 = draw_bbox_lp(vid1, license_detect[0].boxes.xyxy)


    if not ret:
        break

    cv2.imshow('Video', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

