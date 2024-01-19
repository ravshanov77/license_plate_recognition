import torch
import cv2 as cv
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

from ultralytics import YOLO, hub