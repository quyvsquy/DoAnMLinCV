from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from os import listdir                  #use for get dir file in folder
from os.path import isfile, join, isdir #use for get dir file in folder
import sys #use for arvg
import pickle   #use for load file pkl
import tensorflow as tf     #use tensorflow
import facenet              #use for 
import align.detect_face    #use for detect face
import numpy as np
from sklearn.metrics import classification_report,f1_score, accuracy_score
import pandas as pd # use for save predict Y
import os   #use create file
from scipy import misc #use readimg
import matplotlib.pyplot as plt
import cv2

def main(path):
    MINSIZE = 20
    THRESHOLD = [0.6, 0.7, 0.7]
    FACTOR = 0.709
    INPUT_IMAGE_SIZE = 160
    FACENET_MODEL_PATH = '../ModelsPD/20180402-114759.pb'
    CLASSIFIER_PATH = '../Models/facePKL1.pkl'

    with tf.Graph().as_default():
        # Set up GPU
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

        with sess.as_default():
            #read file pkl
            with open(CLASSIFIER_PATH, 'rb') as file:
                model, listNameofDataSet,_,_  = pickle.load(file)

            # Load model MTCNN
            # print('Loading feature extraction model') as .pb
            facenet.load_model(FACENET_MODEL_PATH)

            # Get tensor input and output
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            # Set up NN
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "align")


            IMG_READ = misc.imread(path)
            bounding_boxes, _ = align.detect_face.detect_face(IMG_READ, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

    
            det = bounding_boxes[0, 0:4]
            bb = np.zeros( 4, dtype=np.int32)
            bb[0] = det[0]
            bb[1] = det[1]
            bb[2] = det[2]
            bb[3] = det[3]

            # crop face detected
            cropped = IMG_READ[bb[1]:bb[3], bb[0]:bb[2], :]
            # scaled = facenet.crop(cropped,False,160)
            # scaled = facenet.prewhiten(scaled)
            scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
            scaled = facenet.prewhiten(scaled)
            scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
            
            feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
            emb_array = sess.run(embeddings, feed_dict=feed_dict)
                                
            indexName = model.predict(emb_array)            
            fullName = listNameofDataSet[indexName[0]].split('_')[-1]
            print(fullName)
            return fullName
# main(sys.argv[1])
# print(main(sys.argv[1]))