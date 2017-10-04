# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 17:55:36 2017

@author: elmon
"""

import tensorflow as tf
import os, sys
import numpy as np
from sklearn.metrics import accuracy_score
import csv


def get_labels(labels_file):
    """Get the labels our retraining created."""
    with open(labels_file, 'r') as fin:
        labels = [line.rstrip('/n') for line in fin]
        return labels


def predict_on_image(image, labels, cnn_file):

    # Unpersists graph from file
    with tf.gfile.FastGFile(cnn_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        # Read in the image_data
        image_data = tf.gfile.FastGFile(image, 'rb').read()

        try:
            predictions = sess.run(softmax_tensor, \
                 {'DecodeJpeg/contents:0': image_data})
            prediction = predictions[0]
        except:
            print("Error making prediction.")
            sys.exit()

        # Return the label of the top classification.
        prediction = prediction.tolist()
        print(prediction)        
        return prediction



def predict_on_folder_images(path, labels, cnn_file):

    hf=os.listdir(path)
    predicted_label=np.zeros(len(hf))
    # Unpersists graph from file
    with tf.gfile.FastGFile(cnn_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        for j in range(len(hf)):
            image=path+hf[j]
            # Read in the image_data
            image_data = tf.gfile.FastGFile(image, 'rb').read()
            
            try:
                predictions = sess.run(softmax_tensor, \
                     {'DecodeJpeg/contents:0': image_data})
                prediction = predictions[0]
            except:
                print("Error making prediction.")
                sys.exit()

            # Return the label of the top classification.
            prediction = prediction.tolist()
            max_value = max(prediction)
            max_index = prediction.index(max_value)
            predicted_label[j] = labels[max_index][-2]
            print(hf[j], predicted_label[j])
        return predicted_label

"""
if __name__ == '__main__':
    
    cnn_file='C:/Users/elmon/Desktop/work/HandwritingCNN/re_trainedinception200_150/retrained_graph.pb'
    real_labels='C:/Users/elmon/Desktop/work/HandwritingCNN/laball.txt'
    
    file_result='C:/Users/elmon/Desktop/work/HandwritingCNN/results/results200_150all.csv'
    list_images0='C:/Users/elmon/Desktop/work/HandwritingCNN/data/images/all/200_150jpg/200_150_0/'
    list_images1='C:/Users/elmon/Desktop/work/HandwritingCNN/data/images/all/200_150jpg/200_150_1/'
    
    labelsf=get_labels(real_labels)
    print(labelsf)
    labels=[]
    
    hf0=os.listdir(list_images0)
    hf1=os.listdir(list_images1)
    
    predicted_labels0=predict_on_folder_images(list_images0, labelsf, cnn_file)
    
    labels0=np.ones(len(hf0))
    
    acc0=accuracy_score(labels0, predicted_labels0)
    print('Accuracy class 0:',acc0)
    
    

    predicted_labels1=predict_on_folder_images(list_images1, labelsf, cnn_file)
    
    labels1=np.zeros(len(hf1))
    
    acc1=accuracy_score(labels1, predicted_labels1)
    print('Accuracy class 1:',acc1)
        
    data=[]
    

        
        

with open(file_result, 'w') as csvfile:
    fieldnames = ['name_file', 'predicted_class', 'real_class']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for j in range(len(hf0)):        
        writer.writerow({'name_file': hf0[j], 'predicted_class': predicted_labels0[j], 'real_class':labels0[j]})
    for j in range(len(hf1)):        
        writer.writerow({'name_file': hf1[j], 'predicted_class': predicted_labels1[j], 'real_class':labels1[j]})
        
    #data=np.vstack(data)
    #print(data)
    #np.savetxt('C:/Users/elmon/Desktop/work/HandwritingCNN/re_trainedinception200_150/Results.txt', data)
"""