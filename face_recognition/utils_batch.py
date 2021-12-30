import cv2
import numpy as np
from skimage import transform as trans
import onnxruntime
import json
import os
from mtcnn_ort_batch import MTCNN
import time

src = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041]], dtype=np.float32)
src[:, 0] += 8.0

input_dim = (1280, 720)

tform = trans.SimilarityTransform()
distance_threshold = 0.55



# Reading dictionary
f = open('embeddings.json')
data = json.load(f)
#print("Dict length : ", len(data["embeddings"]))

def detection_results_show(image, faces):

    #image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    for i in range(len(faces[0])):
        cv2.rectangle(image,
                    (int(faces[0][i][0]), int(faces[0][i][1])),
                    (int(faces[0][i][2]), int(faces[0][i][3])),
                    (0,155,255), 2)

        cv2.circle(image,(int(faces[1][0][i]), int(faces[1][5][i])), 2, (0,155,255), 2)
        cv2.circle(image,(int(faces[1][1][i]), int(faces[1][6][i])), 2, (0,155,255), 2)
        cv2.circle(image,(int(faces[1][2][i]), int(faces[1][7][i])), 2, (0,155,255), 2)
        cv2.circle(image,(int(faces[1][3][i]), int(faces[1][8][i])), 2, (0,155,255), 2)
        cv2.circle(image,(int(faces[1][4][i]), int(faces[1][9][i])), 2, (0,155,255), 2)

    cv2.imshow("Input", image)
    cv2.waitKey(0)

def cropped_results_show(image, faces):

    cv2.circle(image,(int(faces[0]), int(faces[5])), 2, (0,155,255), 2)
    cv2.circle(image,(int(faces[1]), int(faces[6])), 2, (0,155,255), 2)
    cv2.circle(image,(int(faces[2]), int(faces[7])), 2, (0,155,255), 2)
    cv2.circle(image,(int(faces[3]), int(faces[8])), 2, (0,155,255), 2)
    cv2.circle(image,(int(faces[4]), int(faces[9])), 2, (0,155,255), 2)

    cv2.imshow("cropped Image", image)
    cv2.waitKey(0)

def cropped_results(image):

    cv2.imshow("Warped Image", image)
    cv2.waitKey(0)

def find_roll(pts):
    return pts[6] - pts[5]

def find_yaw(pts):
    le2n = pts[2] - pts[0]
    re2n = pts[1] - pts[2]
    if le2n < 0 or re2n < 0:
        return 100
    return le2n - re2n

def find_pitch(pts):
    eye_y = (pts[5] + pts[6]) / 2
    mou_y = (pts[8] + pts[9]) / 2
    e2n = eye_y - pts[7]
    n2m = pts[7] - mou_y
    return e2n/n2m

def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def infer_image(batch_dict, detector, session, input_name, output_name):

    images_batch = []
    for image_path in batch_dict:
        image = cv2.cvtColor(cv2.imread(batch_dict[image_path]), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, input_dim, interpolation = cv2.INTER_AREA)
        images_batch.append(image)
    
    faces = detector.detect_faces_raw(images_batch)
    
    recog_dict = {}
    recog_imgs = []
    for j in range(len(images_batch)): # loop over batch
        img_frame_counter = 0
        for i in range(len(faces[str(j)][0])): # loop over faces in frame
            image = images_batch[j]

            # Size of Image is less than 100 pixel,scrap it
            if ((int(faces[str(j)][0][i][3]) - int(faces[str(j)][0][i][1])) > 100 or
             (int(faces[str(j)][0][i][2]) - int(faces[str(j)][0][i][0])) > 100):
                crop_image = image[int(faces[str(j)][0][i][1]): int(faces[str(j)][0][i][3]),
                                int(faces[str(j)][0][i][0]):int(faces[str(j)][0][i][2])]
                landmarks = [int(faces[str(j)][1][0][i]), int(faces[str(j)][1][1][i]), 
                            int(faces[str(j)][1][2][i]), int(faces[str(j)][1][3][i]), 
                            int(faces[str(j)][1][4][i]), int(faces[str(j)][1][5][i]), 
                            int(faces[str(j)][1][6][i]), int(faces[str(j)][1][7][i]), 
                            int(faces[str(j)][1][8][i]), int(faces[str(j)][1][9][i])]
                
                #cropped_results_show(crop_image, landmarks)
                if find_roll(landmarks) > - 20 and  find_roll(landmarks) < 20 and find_yaw(landmarks) > -35 and find_yaw(landmarks) < 35 and find_pitch(landmarks) < 2 and find_pitch(landmarks) > 0.5:
                    # Face Warping
                    facial5points = np.reshape(landmarks, (2, 5)).T
                    tform.estimate(facial5points, src)
                    M = tform.params[0:2, :]
                    img = cv2.warpAffine(image, M, (112, 112), borderValue=0.0)
                    #cropped_results(img)
                    
                    recog_imgs.append(img)
                    img_frame_counter += 1
                else:
                    pass
                    #print("Invalid Pose")
            else :
                pass
                #print("Image size is small")

        recog_dict.update({str(j) : img_frame_counter})

    # Recognition Preprocessing
    blob = cv2.dnn.blobFromImages(recog_imgs, 1, (112, 112), (0, 0, 0))
    blob -= 127.5
    blob /= 128

    # Recognition
    result = session.run([output_name], {input_name: blob})

    counter = 0
    result_dict = {}
    for i in range (len(images_batch)):
        celeb = []
        for j in range(recog_dict[str(i)]):
            # Matching with the existing embeddings
            distance = 1000
            best_distance = 1
            best_index = -1
            emb = result[0][counter]
            #start = time.time()
            for j in range(len(data["embeddings"])):
                distance = findCosineDistance(emb, data["embeddings"][j])
                if (distance < distance_threshold) and (distance < best_distance):
                    best_index = j
                    best_distance = distance
            #end = time.time()
            #print("Embedding time : ", end - start)

            if (best_index != -1):
                celeb += [data["names"][best_index]]
            counter += 1
        keys_list = list(batch_dict)
        result_dict.update({keys_list[i] : celeb})

    return result_dict