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
    "frame_1" : "images/1.jpg",
    "frame_2" : "images/1_1.jpg",
    "frame_3" : "images/11.png",
    "frame_4" : "images/10.png",
    "frame_5" : "images/9.png",
    "frame_2" : "images/1_1.jpg",
    "frame_3" : "images/11.png",
    "frame_4" : "images/10.png",
    "frame_5" : "images/9.png",
    "frame_2" : "images/1_1.jpg"
}


def face_infer_batch(batch_dict):
    results = {}
    start = time.time()
    results = infer_image(batch_dict, detector, session, input_name, output_name)
    end = time.time()
    print("Total time : ", end - start)

    print(results)
    return results


if __name__ == '__main__':
    face_infer_batch(file_batch_dict)