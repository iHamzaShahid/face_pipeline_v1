import os
import imghdr

import cv2
import numpy as np

from onnx_runner import load_model

from trt_inference import trt_infer

class StageStatus(object):
    """
    Keeps status between MTCNN stages
    """
    def __init__(self, pad_result: tuple = None, width=0, height=0):
        self.width = width
        self.height = height
        self.dy = self.edy = self.dx = self.edx = self.y = self.ey = self.x = self.ex = self.tmpw = self.tmph = []
        if pad_result is not None:
            self.update(pad_result)

    def update(self, pad_result: tuple):
        s = self
        s.dy, s.edy, s.dx, s.edx, s.y, s.ey, s.x, s.ex, s.tmpw, s.tmph = pad_result


class MTCNN(object):
    """
    Allows to perform MTCNN Detection ->
        a) Detection of faces (with the confidence probability)
        b) Detection of keypoints (left eye, right eye, nose, mouth_left, mouth_right)
    """
    def __init__(self, min_face_size: int = 20, steps_threshold: list = None,
                 scale_factor: float = 0.709, runner_cls=None):
        """
        Initializes the MTCNN.
        :param min_face_size: minimum size of the face to detect
        :param steps_threshold: step's thresholds values
        :param scale_factor: scale factor
        """
        if steps_threshold is None:
            steps_threshold = [0.6, 0.7, 0.7]

        self._min_face_size = min_face_size
        self._steps_threshold = steps_threshold
        self._scale_factor = scale_factor

        pnet_path = os.path.join(os.path.dirname(__file__), "models/pnet.onnx")
        rnet_path = os.path.join(os.path.dirname(__file__), "models/rnet.onnx")
        onet_path = os.path.join(os.path.dirname(__file__), "models/onet.onnx")

        self._pnet = load_model(pnet_path, runner_cls)
        self._rnet = load_model(rnet_path, runner_cls)
        self._onet = load_model(onet_path, runner_cls)


    @property
    def min_face_size(self):
        return self._min_face_size

    @min_face_size.setter
    def min_face_size(self, mfc=20):
        try:
            self._min_face_size = int(mfc)
        except ValueError:
            self._min_face_size = 20

    def __compute_scale_pyramid(self, m, min_layer):
        scales = []
        factor_count = 0

        while min_layer >= 12:
            scales += [m * np.power(self._scale_factor, factor_count)]
            min_layer = min_layer * self._scale_factor
            factor_count += 1

        return scales

    @staticmethod
    def __scale_image(image, scale: float):
        """
        Scales the image to a given scale.
        :param image:
        :param scale:
        :return:
        """
        height, width, _ = image.shape

        width_scaled = int(np.ceil(width * scale))
        height_scaled = int(np.ceil(height * scale))

        im_data = cv2.resize(image, (width_scaled, height_scaled), interpolation=cv2.INTER_AREA)

        # Normalize the image's pixels
        im_data_normalized = (im_data - 127.5) * 0.0078125

        return im_data_normalized

    @staticmethod
    def __generate_bounding_box(imap, reg, scale, t):

        # use heatmap to generate bounding boxes
        stride = 2
        cellsize = 12

        imap = np.transpose(imap)
        dx1 = np.transpose(reg[:, :, 0])
        dy1 = np.transpose(reg[:, :, 1])
        dx2 = np.transpose(reg[:, :, 2])
        dy2 = np.transpose(reg[:, :, 3])

        y, x = np.where(imap >= t)

        if y.shape[0] == 1:
            dx1 = np.flipud(dx1)
            dy1 = np.flipud(dy1)
            dx2 = np.flipud(dx2)
            dy2 = np.flipud(dy2)

        score = imap[(y, x)]
        reg = np.transpose(np.vstack([dx1[(y, x)], dy1[(y, x)], dx2[(y, x)], dy2[(y, x)]]))

        if reg.size == 0:
            reg = np.empty(shape=(0, 3))

        bb = np.transpose(np.vstack([y, x]))

        q1 = np.fix((stride * bb + 1) / scale)
        q2 = np.fix((stride * bb + cellsize) / scale)
        boundingbox = np.hstack([q1, q2, np.expand_dims(score, 1), reg])

        return boundingbox, reg

    @staticmethod
    def __nms(boxes, threshold, method):
        """
        Non Maximum Suppression.
        :param boxes: np array with bounding boxes.
        :param threshold:
        :param method: NMS method to apply. Available values ('Min', 'Union')
        :return:
        """
        if boxes.size == 0:
            return np.empty((0, 3))

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        s = boxes[:, 4]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        sorted_s = np.argsort(s)

        pick = np.zeros_like(s, dtype=np.int16)
        counter = 0
        while sorted_s.size > 0:
            i = sorted_s[-1]
            pick[counter] = i
            counter += 1
            idx = sorted_s[0:-1]

            xx1 = np.maximum(x1[i], x1[idx])
            yy1 = np.maximum(y1[i], y1[idx])
            xx2 = np.minimum(x2[i], x2[idx])
            yy2 = np.minimum(y2[i], y2[idx])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)

            inter = w * h

            # if method is 'Min':
            if method == 'Min':
                o = inter / np.minimum(area[i], area[idx])
            else:
                o = inter / (area[i] + area[idx] - inter)

            sorted_s = sorted_s[np.where(o <= threshold)]

        pick = pick[0:counter]

        return pick

    @staticmethod
    def __pad(total_boxes, w, h):
        # compute the padding coordinates (pad the bounding boxes to square)
        tmpw = (total_boxes[:, 2] - total_boxes[:, 0] + 1).astype(np.int32)
        tmph = (total_boxes[:, 3] - total_boxes[:, 1] + 1).astype(np.int32)
        numbox = total_boxes.shape[0]

        dx = np.ones(numbox, dtype=np.int32)
        dy = np.ones(numbox, dtype=np.int32)
        edx = tmpw.copy().astype(np.int32)
        edy = tmph.copy().astype(np.int32)

        x = total_boxes[:, 0].copy().astype(np.int32)
        y = total_boxes[:, 1].copy().astype(np.int32)
        ex = total_boxes[:, 2].copy().astype(np.int32)
        ey = total_boxes[:, 3].copy().astype(np.int32)

        tmp = np.where(ex > w)
        edx.flat[tmp] = np.expand_dims(-ex[tmp] + w + tmpw[tmp], 1)
        ex[tmp] = w

        tmp = np.where(ey > h)
        edy.flat[tmp] = np.expand_dims(-ey[tmp] + h + tmph[tmp], 1)
        ey[tmp] = h

        tmp = np.where(x < 1)
        dx.flat[tmp] = np.expand_dims(2 - x[tmp], 1)
        x[tmp] = 1

        tmp = np.where(y < 1)
        dy.flat[tmp] = np.expand_dims(2 - y[tmp], 1)
        y[tmp] = 1

        return dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph

    @staticmethod
    def __rerec(bbox):
        # convert bbox to square
        height = bbox[:, 3] - bbox[:, 1]
        width = bbox[:, 2] - bbox[:, 0]
        max_side_length = np.maximum(width, height)
        bbox[:, 0] = bbox[:, 0] + width * 0.5 - max_side_length * 0.5
        bbox[:, 1] = bbox[:, 1] + height * 0.5 - max_side_length * 0.5
        bbox[:, 2:4] = bbox[:, 0:2] + np.transpose(np.tile(max_side_length, (2, 1)))
        return bbox

    @staticmethod
    def __bbreg(boundingbox, reg):
        # calibrate bounding boxes
        if reg.shape[1] == 1:
            reg = np.reshape(reg, (reg.shape[2], reg.shape[3]))

        w = boundingbox[:, 2] - boundingbox[:, 0] + 1
        h = boundingbox[:, 3] - boundingbox[:, 1] + 1
        b1 = boundingbox[:, 0] + reg[:, 0] * w
        b2 = boundingbox[:, 1] + reg[:, 1] * h
        b3 = boundingbox[:, 2] + reg[:, 2] * w
        b4 = boundingbox[:, 3] + reg[:, 3] * h
        boundingbox[:, 0:4] = np.transpose(np.vstack([b1, b2, b3, b4]))
        return boundingbox

    def detect_faces_raw(self, batch):
        
        batch_size, height, width, _ = np.array(batch).shape

        stage_status = []
        for i in range(batch_size):
            stage_status.append(StageStatus(width=width, height=height))

        m = 12 / self._min_face_size
        min_layer = np.amin([height, width]) * m

        scales = self.__compute_scale_pyramid(m, min_layer)

        stages = [self.__stage1, self.__stage2, self.__stage3]
        result = [scales, stage_status]

        # We pipe here each of the stages
        for stage in stages:
            result = stage(batch, result[0], result[1])

        return result  # total_boxes, points



    def __stage1(self, batch, scales: list, stage_status: StageStatus):

        total_boxes = {}

        for i in range((np.array(batch).shape)[0]):
            key_str = str(i)
            total_boxes[key_str] = np.empty((0, 9))

        status = stage_status

        for scale in scales:
            scaled_image_batch = []
            for i in range((np.array(batch).shape)[0]):
                scaled_image = self.__scale_image(batch[i], scale)
                scaled_image_batch.append(scaled_image)

            img_y = np.transpose(scaled_image_batch, (0, 2, 1, 3))

            #out = self._pnet(img_y)
            out = trt_infer(img_y, model_name='pnet_onnx')

            out0 = np.transpose(out[0], (0, 2, 1, 3))
            out1 = np.transpose(out[1], (0, 2, 1, 3))

            for i in range((np.array(batch).shape)[0]):
                boxes, _ = self.__generate_bounding_box(out1[i, :, :, 1].copy(),
                                                        out0[i, :, :, :].copy(), scale, self._steps_threshold[0])

            # inter-scale nms
                pick = self.__nms(boxes.copy(), 0.5, 'Union')
                if boxes.size > 0 and pick.size > 0:
                    boxes = boxes[pick, :]
                    total_boxes[str(i)] = np.append(total_boxes[str(i)], boxes, axis=0)
        
        for i in range((np.array(batch).shape)[0]):

            numboxes = total_boxes[str(i)].shape[0]

            if numboxes > 0:
                pick = self.__nms(total_boxes[str(i)].copy(), 0.7, 'Union')
                total_boxes[str(i)] = total_boxes[str(i)][pick, :]

                regw = total_boxes[str(i)][:, 2] - total_boxes[str(i)][:, 0]
                regh = total_boxes[str(i)][:, 3] - total_boxes[str(i)][:, 1]

                qq1 = total_boxes[str(i)][:, 0] + total_boxes[str(i)][:, 5] * regw
                qq2 = total_boxes[str(i)][:, 1] + total_boxes[str(i)][:, 6] * regh
                qq3 = total_boxes[str(i)][:, 2] + total_boxes[str(i)][:, 7] * regw
                qq4 = total_boxes[str(i)][:, 3] + total_boxes[str(i)][:, 8] * regh

                total_boxes[str(i)] = np.transpose(np.vstack([qq1, qq2, qq3, qq4, total_boxes[str(i)][:, 4]]))
                total_boxes[str(i)] = self.__rerec(total_boxes[str(i)].copy())

                total_boxes[str(i)][:, 0:4] = np.fix(total_boxes[str(i)][:, 0:4]).astype(np.int32)
                status[i] = StageStatus(self.__pad(total_boxes[str(i)].copy(), stage_status[i].width, stage_status[i].height),
                                    width=stage_status[i].width, height=stage_status[i].height)

        return total_boxes, status

    def __stage2(self, batch, total_boxeses, stage_status):

        input_img = np.empty([0, 24, 24, 3])
        for i in range((np.array(batch).shape)[0]):
            
            total_boxes = total_boxeses[str(i)]
            img = batch[i]

            num_boxes = total_boxes.shape[0]
            if num_boxes == 0:
                continue

            # second stage
            tempimg = np.zeros(shape=(24, 24, 3, num_boxes))

            for k in range(0, num_boxes):
                tmp = np.zeros((int(stage_status[i].tmph[k]), int(stage_status[i].tmpw[k]), 3))

                tmp[stage_status[i].dy[k] - 1:stage_status[i].edy[k], stage_status[i].dx[k] - 1:stage_status[i].edx[k], :] = \
                    img[stage_status[i].y[k] - 1:stage_status[i].ey[k], stage_status[i].x[k] - 1:stage_status[i].ex[k], :]

                if tmp.shape[0] > 0 and tmp.shape[1] > 0 or tmp.shape[0] == 0 and tmp.shape[1] == 0:
                    tempimg[:, :, :, k] = cv2.resize(tmp, (24, 24), interpolation=cv2.INTER_AREA)

                else:
                    return np.empty(shape=(0,)), stage_status[i]

            tempimg = (tempimg - 127.5) * 0.0078125
            tempimg1 = np.transpose(tempimg, (3, 1, 0, 2))
            input_img = np.concatenate((input_img, tempimg1), axis=0)

        #print("Input image shape : ", np.array(input_img).shape)
        #out = self._rnet(input_img)
        out = trt_infer(input_img, model_name='rnet_onnx')


        batch_output_0 = np.array(out[0])
        batch_output_1 = np.array(out[1])
        
        step = 0
        for i in range((np.array(batch).shape)[0]):
            total_boxes = total_boxeses[str(i)]
            output_0 = batch_output_0[step:step + total_boxes.shape[0], :]
            output_1 = batch_output_1[step:step + total_boxes.shape[0], :]
            step += total_boxes.shape[0]

            out0 = np.transpose(output_0)
            out1 = np.transpose(output_1)

            score = out1[1, :]

            ipass = np.where(score > self._steps_threshold[1])

            total_boxeses[str(i)] = np.hstack([total_boxeses[str(i)][ipass[0], 0:4].copy(), np.expand_dims(score[ipass].copy(), 1)])

            mv = out0[:, ipass[0]]

            if total_boxeses[str(i)].shape[0] > 0:
                pick = self.__nms(total_boxeses[str(i)], 0.7, 'Union')
                total_boxeses[str(i)] = total_boxeses[str(i)][pick, :]
                total_boxeses[str(i)] = self.__bbreg(total_boxeses[str(i)].copy(), np.transpose(mv[:, pick]))
                total_boxeses[str(i)] = self.__rerec(total_boxeses[str(i)].copy())
            
        return total_boxeses, stage_status

    def __stage3(self, batch, total_boxeses, stage_status):

        faces = {}
        input_img = np.empty([0, 48, 48, 3])
        for i in range((np.array(batch).shape)[0]):
            key_str = str(i)
            #faces[key_str] = np.empty((0, 5))
            #faces[key_str] = np.empty((0, 10))
            total_boxes = total_boxeses[str(i)]
            img = batch[i]

            num_boxes = total_boxes.shape[0]
            #print("num_boxes : ", num_boxes)
            if num_boxes == 0:
                continue

            total_boxes = np.fix(total_boxes).astype(np.int32)

            status = StageStatus(self.__pad(total_boxes.copy(), stage_status[i].width, stage_status[i].height),
                                width=stage_status[i].width, height=stage_status[i].height)

            tempimg = np.zeros((48, 48, 3, num_boxes))

            for k in range(0, num_boxes):

                tmp = np.zeros((int(status.tmph[k]), int(status.tmpw[k]), 3))

                tmp[status.dy[k] - 1:status.edy[k], status.dx[k] - 1:status.edx[k], :] = \
                    img[status.y[k] - 1:status.ey[k], status.x[k] - 1:status.ex[k], :]

                if tmp.shape[0] > 0 and tmp.shape[1] > 0 or tmp.shape[0] == 0 and tmp.shape[1] == 0:
                    tempimg[:, :, :, k] = cv2.resize(tmp, (48, 48), interpolation=cv2.INTER_AREA)
                else:
                    return np.empty(shape=(0,)), np.empty(shape=(0,))

            tempimg = (tempimg - 127.5) * 0.0078125
            tempimg1 = np.transpose(tempimg, (3, 1, 0, 2))
            #print("tempimg1 shape : ", tempimg1.shape)
            input_img = np.concatenate((input_img, tempimg1), axis=0)
        #print("input_img shape : ", np.array(input_img).shape)
        """
        self._onet.setInput(tempimg1)
        out = self._onet.forward(['dense_5', 'dense_6', 'softmax_2'])
        """
        #out = self._onet(input_img)
        out = trt_infer(input_img, model_name='onet_onnx')

        #print("Out_0 shape : ", out[0].shape)
        #print("Out_1 shape : ", out[1].shape)
        #print("Out_2 shape : ", out[2].shape)
        batch_output_0 = np.array(out[0])
        batch_output_1 = np.array(out[1])
        batch_output_2 = np.array(out[2])

        step = 0
        for i in range((np.array(batch).shape)[0]):
            total_boxes = total_boxeses[str(i)]
            output_0 = batch_output_0[step:step + total_boxes.shape[0], :]
            output_1 = batch_output_1[step:step + total_boxes.shape[0], :]
            output_2 = batch_output_2[step:step + total_boxes.shape[0], :]
            step += total_boxes.shape[0]

            out0 = np.transpose(output_0)
            out1 = np.transpose(output_1)
            out2 = np.transpose(output_2)

            score = out2[1, :]

            points = out1

            ipass = np.where(score > self._steps_threshold[2])

            points = points[:, ipass[0]]

            total_boxes = np.hstack([total_boxes[ipass[0], 0:4].copy(), np.expand_dims(score[ipass].copy(), 1)])

            mv = out0[:, ipass[0]]

            w = total_boxes[:, 2] - total_boxes[:, 0] + 1
            h = total_boxes[:, 3] - total_boxes[:, 1] + 1

            points[0:5, :] = np.tile(w, (5, 1)) * points[0:5, :] + np.tile(total_boxes[:, 0], (5, 1)) - 1
            points[5:10, :] = np.tile(h, (5, 1)) * points[5:10, :] + np.tile(total_boxes[:, 1], (5, 1)) - 1

            if total_boxes.shape[0] > 0:
                total_boxes = self.__bbreg(total_boxes.copy(), np.transpose(mv))
                pick = self.__nms(total_boxes.copy(), 0.7, 'Min')
                total_boxes = total_boxes[pick, :]
                points = points[:, pick]
            
            faces.update({str(i) : (total_boxes,points)})
        #print("faces : ", faces)
        return faces
