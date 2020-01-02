from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import math
import pickle
from sklearn.svm import SVC, LinearSVC
import tachTestTrain as t
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier

def main(args):
  
    with tf.Graph().as_default():
      
        with tf.Session() as sess:
            paths, labels, testX, testY = t.loadDuLieu(args.data_dir, args.checkCreateNewDataSet)
            listNameFolderImage = t.timFolderName(args.data_dir)
            print('Number of classes: %d' % len(listNameFolderImage))
            
            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(args.model)
            
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            
            for ia in range(10): #k-cross
                print("create " + str(ia + 1) + "/10")
                print('Number of images: %d' % len(paths[ia])) 
                # Run forward pass to calculate embeddings
                print('Calculating features for images')
                nrof_images = len(paths[ia]) 
                emb_array = np.zeros((nrof_images, embedding_size))
                images = facenet.load_data(paths[ia], False, False, args.image_size)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                emb_array = sess.run(embeddings, feed_dict=feed_dict)
                
                classifier_filename_exp = os.path.expanduser(args.classifier_filename)

                # Train classifier
                print('Training classifier')
                # model = SVC(kernel='linear', probability=True)
                # model = LinearSVC()
                # model = GaussianNB()
                model = KNeighborsClassifier(n_neighbors=100, n_jobs=-1)
                model.fit(emb_array, labels[ia])
            
                # Saving classifier model
                with open(classifier_filename_exp + str(ia) + ".pkl", 'wb') as outfile:
                    pickle.dump((model, listNameFolderImage, testX[ia], testY[ia]), outfile)
                print('Saved classifier model to file %s' % classifier_filename_exp+ str(ia) + ".pkl")
                print("------------------------------------------------------")
                    

            
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('data_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument("--c", dest="checkCreateNewDataSet", action="store_true",
        help="if has --c is create new random dataset", default=False)
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('classifier_filename', 
        help='Save Classifier model file name as a pickle (.pkl) file. ' + 
        'For training this is the output')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
