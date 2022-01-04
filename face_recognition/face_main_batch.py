import cv2
import numpy as np
from utils_batch import *
import time
from mtcnn_ort_batch import MTCNN
import onnxruntime

# Initializing the detector
detector = MTCNN()

# Initializing the recognizer
model_path = "models/arcface.onnx"
session = onnxruntime.InferenceSession(model_path, None)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


file_batch_dict = {
    "frame_0" : "images/20.jpg",
    "frame_1" : "images/21.jpg",
    "frame_2" : "images/22.jpg",
    "frame_3" : "images/23.jpg",
    "frame_4" : "images/24.jpg",
    "frame_5" : "images/25.jpg",
    "frame_6" : "images/26.jpg",
    "frame_7" : "images/27.jpg",
    "frame_8" : "images/28.jpg",
    "frame_9" : "images/29.jpg",
    "frame_10" : "images/30.jpg",
    "frame_11" : "images/31.jpg",
    "frame_12" : "images/32.jpg",
    "frame_13" : "images/33.jpg",
    "frame_14" : "images/34.jpg",
    "frame_15" : "images/35.jpg",
    "frame_16" : "images/36.jpg",
    "frame_17" : "images/37.jpg",
    "frame_18" : "images/38.jpg",
    "frame_19" : "images/39.jpg",
    "frame_20" : "images/40.jpg",
    "frame_21" : "images/41.jpg",
    "frame_22" : "images/42.jpg",
    "frame_23" : "images/43.jpg"
}


def face_infer_batch(batch_dict):
    results = {}
    start = time.time()
    results = infer_image(batch_dict, detector, session, input_name, output_name)
    end = time.time()
    #print("Total time : ", end - start)

    print(results)
    return results


if __name__ == '__main__':
    face_infer_batch(file_batch_dict)