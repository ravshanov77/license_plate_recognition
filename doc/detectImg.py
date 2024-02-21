"""
This file is built in order to recognise license plate (number plate) only on images.

"""

# needs
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from ultralytics import YOLO


# defining the models
model_yolo = YOLO('models/yolov8s.pt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_yolo = model_yolo.to(device)

model_license = YOLO('models/license.pt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_license = model_license.to(device)

model_alpha = YOLO('models/alpha_num.pt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_alpha = model_alpha.to(device)

def getImg(path):
    """
    Read the input image and return image with converting it into numpy array

    Args:
        image path: input image containing car

    Returns:
        array: actual image
    """
    img = cv2.imread(path)
    img = np.array(img)
    return img

def getCars(img):
    """
    Predict only cars within the image after using the YOLOv8s model on the image 

    Args:
        image: numpy array

    Returns:

    """
    car_detect = model_yolo(img)

    # separating only cars from the prediction
    cars = car_detect[0].boxes[car_detect[0].boxes.cls==2.]
    return cars

def drawBbox(img, labels):
    """
    Draw bounding box of cars

    Args:
        img: numpy array
        labels (array): labels (points) of bbox that was detected by YOLO model and is in the result it returned (car_detect)

    Returns:
        array: image with detected cars
    """
    x = 0
    while x < len(labels):
        for i in labels:
            img = cv2.rectangle(img, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), color=[0,255,0], thickness=2)

            img = cv2.putText(img, 'car', org=(int(i[0]), int(i[1]) - 5),
                            fontFace=int(img.shape[0] ** .15),
                            fontScale=int(img.shape[0] ** .1),
                            color=[255,0,0],
                            thickness=int(img.shape[0] ** .1))

            font = cv2.FONT_HERSHEY_SIMPLEX
        x += 1

    return img

licenseDetect = lambda img: model_license(img) # detecting the license
alphaDetect = lambda img: model_alpha(img) # detecting the characters and numbers

def drawLisence(img, labels):
    """
    Draw bbox for license

    Args:
        img: numpy array
        lables (array): labels of license plate that was found by 'model_license'

    Returns:
        array: img with the detected license(s)
    """
    x = 0
    while x < len(labels):
        for i in labels:
            img = cv2.rectangle(img, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), color=[0,255,0], thickness=int(img.shape[0] ** .2))

            img = cv2.putText(img, 'number', org=(int(i[0]), int(i[1]) - 5),
                       fontFace=int(img.shape[0] ** .11),
                       fontScale=int(img.shape[0] ** .1),
                       color=[255,0,0],
                       thickness=int(img.shape[0] ** .15))

            font = cv2.FONT_HERSHEY_SIMPLEX

        x += 1
    return img

def cropImg(img):
    """
    Crop the license plate

    Args:
        img: numpy array

    Returns:
        img: cropped license plate
    """
    x1, y1, x2, y2 = licenseDetect(img)[0].boxes.xyxy[0].numpy()

    licence_plate = img[int(x1):int(x2), int(y1):int(x2)]
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)

    w = int(x2 - x1)
    h = int(y2 - y1)
    crop_img = img[y1:y1+h, x1:x1+w]
    return crop_img

def getActualNum(results):
    """
    Get the an actual number plate value with the results got by alpha_num model

    Args:
        results (array): labels of license plate values that was found by 'alpha_num' model
    
    Returns:
        Ordered value of a license plate
    """
    mixed_result = [results[0].names.get(int(x)) for x in results[0].boxes.cls]
    values = list(zip(results[0].boxes.cls.numpy(), results[0].boxes.xyxy.numpy()))

    x = sorted(values, key=lambda x: x[1][0])
    list_x = []
    for cl, box in x:
        list_x.append(int(cl))
    
    true_seq = [results[0].names.get(y) for y in list_x]
    # Mapping of string numbers to their integer representations
    number_mapping = {
        'zero': '0',
        'one': '1',
        'two': '2',
        'three': '3',
        'four': '4',
        'five': '5',
        'six': '6',
        'seven': '7',
        'eight': '8',
        'nine': '9',
        # Add more numbers as needed
    }

    # Your original list
    original_list = true_seq

    # Convert string numbers to integers while keeping other items unchanged
    converted_list = [number_mapping[item] if item in number_mapping else item for item in original_list]

    actual_num = "".join(converted_list)
    return actual_num[:2] + ' ' + actual_num[2:]

# sample usage
img = getImg('resources/car2.jpg')
cars = getCars(img)
car = drawBbox(img, labels=cars.xyxy)
license_detect = licenseDetect(car)
license = drawLisence(car, labels=license_detect[0].boxes.xyxy)
crop_img = cropImg(car)
alpha = alphaDetect(crop_img)
print(alpha[0].boxes.cls)
actual_num = getActualNum(alpha)
print(actual_num)

cv2.imshow('License Image', license)
cv2.imshow('Crop Image', crop_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


