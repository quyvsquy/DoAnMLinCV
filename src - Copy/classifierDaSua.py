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
            # np.random.seed(seed=args.seed)
            
            # if args.use_split_dataset:
            #     dataset_tmp = facenet.get_dataset(args.data_dir)
            #     train_set, test_set = split_dataset(dataset_tmp, args.min_nrof_images_per_class, args.nrof_train_images_per_class)
            #     if (args.mode=='TRAIN'):
            #         dataset = train_set
            #     elif (args.mode=='CLASSIFY'):
            #         dataset = test_set
            # else:
            #     dataset = facenet.get_dataset(args.data_dir)
            # # print(dataset)    
            # # Check that there are at least one training image per class
            # for cls in dataset:
            #     assert len(cls.image_paths)>0, 'There must be at least one image for each class in the dataset'

                 
            # paths, labels = facenet.get_image_paths_and_labels(dataset)
            paths, labels, testX, testY = t.loadDuLieu(args.data_dir, args.checkCreateNewDataSet)
            listNameFolderImage = t.timFolderName(args.data_dir)
            # print(labels)
            # print(paths)

            # print(ia)
            # print(paths[0])
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
                print('Number of images: %d' % len(paths[ia])) #########################################
                # Run forward pass to calculate embeddings
                print('Calculating features for images')
                nrof_images = len(paths[ia]) ######################################################
                # nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / args.batch_size))
                emb_array = np.zeros((nrof_images, embedding_size))
                # for i in range(nrof_batches_per_epoch):
                    # start_index = i*args.batch_size
                    # end_index = min((i+1)*args.batch_size, nrof_images)
                    # paths_batch = paths[ia][start_index:end_index]   #############################
                images = facenet.load_data(paths[ia], False, False, args.image_size)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                emb_array = sess.run(embeddings, feed_dict=feed_dict)
                
                classifier_filename_exp = os.path.expanduser(args.classifier_filename)

                # Train classifier
                print('Training classifier')
                # model = SVC(kernel='linear', probability=True)
                # model = LinearSVC()
                model = GaussianNB()
                # model = KNeighborsClassifier(n_neighbors=100, n_jobs=-1)
                model.fit(emb_array, labels[ia])
            
                # Create a list of class names
                # class_names = [ cls.name.replace('_', ' ') for cls in dataset]
                # class_names = listNameFolderImage  
                # print(class_names)
                # Saving classifier model
                with open(classifier_filename_exp + str(ia) + ".pkl", 'wb') as outfile:
                    pickle.dump((model, listNameFolderImage, testX[ia], testY[ia]), outfile)
                print('Saved classifier model to file %s' % classifier_filename_exp+ str(ia) + ".pkl")
                print("------------------------------------------------------")
                    
            # elif (args.mode=='CLASSIFY'):
            #     # Classify images
            #     print('Testing classifier')
            #     with open(classifier_filename_exp, 'rb') as infile:
            #         (model, class_names) = pickle.load(infile)

            #     print('Loaded classifier model from file "%s"' % classifier_filename_exp)

            #     predictions = model.predict_proba(emb_array)
            #     best_class_indices = np.argmax(predictions, axis=1)
            #     best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                
            #     for i in range(len(best_class_indices)):
            #         print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))
                    
            #     accuracy = np.mean(np.equal(best_class_indices, labels))
            #     print('Accuracy: %.3f' % accuracy)
                
            
# def split_dataset(dataset, min_nrof_images_per_class, nrof_train_images_per_class):
#     train_set = []
#     test_set = []
#     for cls in dataset:
#         paths = cls.image_paths
#         # Remove classes with less than min_nrof_images_per_class
#         if len(paths)>=min_nrof_images_per_class:
#             np.random.shuffle(paths)
#             train_set.append(facenet.ImageClass(cls.name, paths[:nrof_train_images_per_class]))
#             test_set.append(facenet.ImageClass(cls.name, paths[nrof_train_images_per_class:]))
#     return train_set, test_set

            
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    # parser.add_argument('mode', type=str, choices=['TRAIN', 'CLASSIFY'],
    #     help='Indicates if a new classifier should be trained or a classification ' + 
    #     'model should be used for classification', default='CLASSIFY')
    parser.add_argument('data_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.')
    # parser.add_argument('checkCreateNewDataSet', type=bool,
    #     help='if True is create new random dataset', default=False)
    parser.add_argument("--c", dest="checkCreateNewDataSet", action="store_true",
        help="if has --c is create new random dataset", default=False)
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('classifier_filename', 
        help='Save Classifier model file name as a pickle (.pkl) file. ' + 
        'For training this is the output')
    # parser.add_argument('--batch_size', type=int,
    #     help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    # parser.add_argument('--seed', type=int,
    #     help='Random seed.', default=12)
    
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
