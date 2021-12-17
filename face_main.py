import cv2
import numpy as np
from utils import *
import time


file_batch_dict = {
    "frame_1" : "images/1.jpg",
    "frame_2" : "images/1_1.jpg",
    "frame_3" : "images/11.png",
    "frame_4" : "images/10.png",
    "frame_5" : "images/9.png"
}


def face_infer_batch(batch_dict):
    results = {}

    for image_path in batch_dict:
        results[image_path] = infer_image(batch_dict[image_path])

    print(results)
    return results


if __name__ == '__main__':
    face_infer_batch(file_batch_dict)