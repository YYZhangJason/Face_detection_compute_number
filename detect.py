#!/usr/bin/env python3
# encoding:utf-8
'''
@author: YYZhang
@file: detect.py
@time: 2020/5/31 21:20
@purpose:compute detection face number
'''
#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import division
import time
import sys
import cv2
def classify_gray_hist(image1,image2,size = (256,256)):
 image1 = cv2.resize(image1,size)
 image2 = cv2.resize(image2,size)
 hist1 = cv2.calcHist([image1],[0],None,[256],[0.0,255.0])
 hist2 = cv2.calcHist([image2],[0],None,[256],[0.0,255.0])

 degree = 0
 for i in range(len(hist1)):
  if hist1[i] != hist2[i]:
   degree = degree + (1 - abs(hist1[i]-hist2[i])/max(hist1[i],hist2[i]))
  else:
   degree = degree + 1
 degree = degree/len(hist1)
 return degree

def calculate(image1,image2):
 hist1 = cv2.calcHist([image1],[0],None,[256],[0.0,255.0])
 hist2 = cv2.calcHist([image2],[0],None,[256],[0.0,255.0])

 degree = 0
 for i in range(len(hist1)):
  if hist1[i] != hist2[i]:
   degree = degree + (1 - abs(hist1[i]-hist2[i])/max(hist1[i],hist2[i]))
  else:
   degree = degree + 1
 degree = degree/len(hist1)
 return degree

def classify_hist_with_split(image1,image2,size = (256,256)):
 image1 = cv2.resize(image1,size)
 image2 = cv2.resize(image2,size)
 sub_image1 = cv2.split(image1)
 sub_image2 = cv2.split(image2)
 sub_data = 0
 for im1,im2 in zip(sub_image1,sub_image2):
  sub_data += calculate(im1,im2)
 sub_data = sub_data/3
 return sub_data

def classify_aHash(image1,image2):
 image1 = cv2.resize(image1,(8,8))
 image2 = cv2.resize(image2,(8,8))
 gray1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
 gray2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
 hash1 = getHash(gray1)
 hash2 = getHash(gray2)
 return Hamming_distance(hash1,hash2)

def classify_pHash(image1,image2):
 image1 = cv2.resize(image1,(32,32))
 image2 = cv2.resize(image2,(32,32))
 gray1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
 gray2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
 dct1 = cv2.dct(np.float32(gray1))
 dct2 = cv2.dct(np.float32(gray2))
 dct1_roi = dct1[0:8,0:8]
 dct2_roi = dct2[0:8,0:8]
 hash1 = getHash(dct1_roi)
 hash2 = getHash(dct2_roi)
 return Hamming_distance(hash1,hash2)

# 输入灰度图，返回hash
def getHash(image):
 avreage = np.mean(image)
 hash = []
 for i in range(image.shape[0]):
  for j in range(image.shape[1]):
   if image[i,j] > avreage:
    hash.append(1)
   else:
    hash.append(0)
 return hash


# 计算汉明距离
def Hamming_distance(hash1,hash2):
 num = 0
 for index in range(len(hash1)):
  if hash1[index] != hash2[index]:
   num += 1
 return num

def detectFaceOpenCVDnn(net, frame):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    '''检测'''
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], False, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    number = len(bboxes)
    string_out = "Face Detection number is :" +str(number)
    cv2.putText(frameOpencvDnn, string_out, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3,cv2.LINE_AA)

    print(len(bboxes))
    return frameOpencvDnn, bboxes
import numpy as np
if __name__ == "__main__":

    # OpenCV DNN supports 2 networks.
    # 1. FP16 version of the original caffe implementation ( 5.4 MB )
    # 2. 8 bit Quantized version using Tensorflow ( 2.7 MB )
    '''模型加载'''
    DNN = "CAFFE"
    if DNN == "CAFFE":
        modelFile = r".\models\face_detection.caffemodel"
        configFile = r".\models\deploy.prototxt"
        net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    else:
        modelFile = "./model/opencv_face_detector_uint8.pb"
        configFile = "./model/opencv_face_detector.pbtxt"
        net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

    conf_threshold = 0.7

    source = 0
    if len(sys.argv) > 1:
        source = sys.argv[1]

    cap = cv2.VideoCapture(source)
    hasFrame, frame = cap.read()

    vid_writer = cv2.VideoWriter('output-dnn-{}.avi'.format(str(source).split(".")[0]),
                                 cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15, (frame.shape[1], frame.shape[0]))

    frame_count = 0
    tt_opencvDnn = 0



    while (1):
        hasFrame, frame = cap.read()
        if not hasFrame:
            break
        frame_count += 1

        t = time.time()
        outOpencvDnn, bboxes = detectFaceOpenCVDnn(net, frame)


        tt_opencvDnn += time.time() - t
        fpsOpencvDnn = frame_count / tt_opencvDnn
        label = "Face Detection Result FPS : {:.2f}".format(fpsOpencvDnn)
        print("Face Detection Result:")


        cv2.putText(outOpencvDnn, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)

        cv2.imshow("Face Detection", outOpencvDnn)

        vid_writer.write(outOpencvDnn)
        if frame_count == 1:
            tt_opencvDnn = 0

        k = cv2.waitKey(10)
        if k == 27:
            break
    cv2.destroyAllWindows()
    vid_writer.release()