import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from drawing_car_bbox import *
from ultralytics import YOLO


#Defining YOLO model
model = YOLO('models/yolov8s.pt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

#Preparing the video
video = 'test_resources/single_car.mp4'

output_video_path = "output"

cap = cv2.VideoCapture(video)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use appropriate codec based on file extension
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))


if not cap.isOpened():
    exit()

while True:
    ret, frame = cap.read()
    frame = np.asarray(frame)

    car_detect = model(frame)

    # separating only cars from the predictions
    cars = car_detect[0].boxes[car_detect[0].boxes.cls==2.]

    # drawing bbox for a car
    vid1 = draw_bbox(frame, labels=cars.xyxy)

    # Detecting the license plate
    model2 = YOLO('models\licence.pt')

    license_detect = model2(frame)

    img2 = draw_bbox_lp(frame, license_detect[0].boxes.xyxy)

    # Setting up the points of the number
    if license_detect[0].boxes.cls != None:
        print(license_detect[0].boxes)
        x1, y1, x2, y2 = license_detect[0].boxes.xyxy[0].numpy()
    else:
        print("License detection result is empty.")


    # licence_plate = img2[int(x1):int(x2), int(y1):int(x2)]

    # x1 = int(x1)
    # y1 = int(y1)
    # x2 = int(x2)
    # y2 = int(y2)

    # w = int(x2 - x1)
    # h = int(y2 - y1)

    # crop_img = frame[y1:y1+h, x1:x1+w]
    # cv2.imshow('Crop', mat=crop_img)


    out.write(frame)
    if not ret:
        break

    cv2.imshow('Video', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
