from keras.models import load_model
from os import listdir
from os.path import isdir
from random import choice

import numpy as np
from numpy import savez_compressed
from numpy import load
from numpy import expand_dims
from PIL import Image
from matplotlib import pyplot

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

import mtcnn
import cv2


# extract faces from given files(used during verification)
def resize_face(img, size=(160, 160)):
    image = Image.fromarray(img)
    image = image.resize(size)
    face_array = np.asarray(image)
    return face_array


# extract faces from given files(used during training)
def extract_facefile(filename, size=(160, 160)):
    # read image to detect
    img = Image.open(filename)
    img_arr = np.asarray(img)

    # create a MTCNN detector and catch faces
    detector = mtcnn.MTCNN()
    results = detector.detect_faces(img_arr)
    x1, y1, w, h = results[0].get('box')
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + w, y1 + h

    # face extracted
    face = img_arr[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(size)
    face_array = np.asarray(image)
    return face_array


# load images and extract faces for all images in a directory
def load_faces(directory):
    faces = list()
    # enumerate files
    for filename in listdir(directory):
        # path
        path = directory + filename
        print("Image path obtained: ", path)
        # get face
        face = extract_facefile(path)
        print(path, " has been extracted!")
        # store
        faces.append(face)
    return faces


# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory):
    X, y = list(), list()
    # enumerate folders, on per class
    for subdir in listdir(directory):
        # path
        path = directory + subdir + '/'
        # skip any files that might be in the dir
        if not isdir(path):
            continue
        # load all faces in the subdirectory
        print("Path loaded:", path)
        faces = load_faces(path)
        # create labels
        labels = [subdir for _ in range(len(faces))]
        # summarize progress
        print('>loaded %d examples for class: %s' % (len(faces), subdir))
        # store
        X.extend(faces)
        y.extend(labels)
    return np.asarray(X), np.asarray(y)


# get the face embedding for one face
def get_embedding(model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0]


'''load all the data and prepare to train the FaceNet model'''
# load train dataset
trainX, trainy = load_dataset('dataset/train/')
print(trainX.shape, trainy.shape)
# load test dataset
testX, testy = load_dataset('dataset/val/')
# save arrays to one file compressed format
savez_compressed('dataset.npz', trainX, trainy, testX, testy)

'''using FaceNet to do face embedding and preprocessing dataset(normalization and etc...)'''
# load the face dataset
data = load('dataset.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)
# load the FaceNet model
Face_net_model = load_model('facenet_keras.h5')
print('Loaded Model')
# convert each face in the train set to an embedding
newTrainX = list()
for face_pixels in trainX:
    embedding = get_embedding(Face_net_model, face_pixels)
    newTrainX.append(embedding)
newTrainX = np.asarray(newTrainX)
print(newTrainX.shape)
# convert each face in the test set to an embedding
newTestX = list()
for face_pixels in testX:
    embedding = get_embedding(Face_net_model, face_pixels)
    newTestX.append(embedding)
newTestX = np.asarray(newTestX)
print(newTestX.shape)
# save arrays to one file in compressed format
savez_compressed('dataset_embeddings.npz', newTrainX, trainy, newTestX, testy)


'''apply SVM to do classification beca、use SVM is suitable when face embedding is done'''
# load dataset
Face_net_model = load_model('facenet_keras.h5')
data = load('dataset_embeddings.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
# normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
textX = in_encoder.transform(testX)
# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)
# fit model
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)
# predict
yhat_train = model.predict(trainX)
yhat_test = model.predict(testX)
# score
score_train = accuracy_score(trainy,  yhat_train)
score_test = accuracy_score(testy,  yhat_test)
# summarize
print('Accuracy: train=%.3f, test=%.3f' % (score_train*100,  score_test*100))




