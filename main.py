import PIL.Image
from keras.models import load_model
from os import listdir
from os.path import isdir
from random import choice

import numpy as np
from numpy import savez_compressed
from numpy import load
from numpy import expand_dims
from PIL import Image, ImageTk
from matplotlib import pyplot
import tkinter as tk

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
        # get face
        face = extract_facefile(path)
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


def load_model_and_data():
    """apply SVM to do classification because SVM is suitable when face embedding is done"""
    # load model
    FaceNet = load_model('facenet_keras.h5')
    # load dataset
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
    return FaceNet, model, out_encoder


def show_img_canvas(imgCV_in, canva, layout='null'):
    global imgTK
    global imgCV
    canvawidth = int(canva.winfo_reqwidth())
    canvaheight = int(canva.winfo_reqheight())
    sp = imgCV_in.shape
    cvheight = sp[0]  # height(rows) of image
    cvwidth = sp[1]  # width(columns) of image
    if layout == "fill":
        imgCV = cv2.resize(imgCV_in, (canvawidth, canvaheight), interpolation=cv2.INTER_AREA)
    elif layout == "fit":
        if float(cvwidth / cvheight) > float(canvawidth / canvaheight):
            imgCV = cv2.resize(imgCV_in, (canvawidth, int(canvawidth * cvheight / cvwidth)),
                               interpolation=cv2.INTER_AREA)
        else:
            imgCV = cv2.resize(imgCV_in, (int(canvaheight * cvwidth / cvheight), canvaheight),
                               interpolation=cv2.INTER_AREA)
    else:
        imgCV = imgCV_in
    imgCV2 = cv2.cvtColor(imgCV, cv2.COLOR_BGR2RGBA)
    current_image = Image.fromarray(imgCV2)
    imgTK = ImageTk.PhotoImage(image=current_image)
    canva.create_image(0, 0, anchor='nw', image=imgTK)


def init_window():
    """using tkinter to create UI interface"""
    window = tk.Tk()
    window.title("人脸识别系统")
    label_1 = tk.Label(text="南京邮电大学人脸识别系统", font=('Arial', 20), width=50, height=10)
    label_1.pack()
    width = 1080
    height = 900
    screenwidth = window.winfo_screenwidth()
    screenheight = window.winfo_screenheight()
    alighstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
    window.geometry(alighstr)
    canva = tk.Canvas(window, width=600, height=335)
    return window, canva


FaceNet, model, out_encoder = load_model_and_data()

'''input an face captured from camera and predict this unlabeled face'''
# initiate camera
cap = cv2.VideoCapture(0)
stop = False
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
count = 1

window, canvas = init_window()

# start face detection
while not stop:
    success, img = cap.read()

    gary = None
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5,
                                          minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    # show detection rectangle
    for (x, y, w, h) in faces:
        # show image on the window
        # canvas.create_rectangle((x-270, y-150), (x-130, y), outline='red')
        cv2.rectangle(img, (x, y), (x + w, y + w), (0, 255, 0), 2)

        # transform origin image to the form that fits SVM
        face = resize_face(img[y:y + w, x:x + w, :])
        face_emb = get_embedding(FaceNet, face)
        sample = expand_dims(face_emb, axis=0)
        # predict
        yhat_class = model.predict(sample)
        yhat_prob = model.predict_proba(sample)
        # get name
        class_index = yhat_class[0]
        class_probability = yhat_prob[0, class_index] * 100
        predict_names = out_encoder.inverse_transform(yhat_class)
        print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))

    show_img_canvas(img, canvas, 'fit')
    # show label info
    label_1 = "Result:Detected " + str(len(faces)) + " faces!"
    label_2 = "Result:Saved " + str(count) + " faces!"
    label_3 = predict_names[0]
    label_4 = "Probability: " + str(class_probability)

    '''
    cv2.putText(img, label_1, (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 1)
    cv2.putText(img, label_2, (10, 40), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 1)
    cv2.putText(img, label_3, (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1)
    cv2.putText(img, label_4, (x, y - 40), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1)
    '''

    canvas.create_text(x-200, y-170, text=label_3, fill='green')
    canvas.create_text(x-200, y-190, text=label_4, fill='green')
    canvas.pack()
    window.update_idletasks()
    window.update()




    '''
    # show image
    cv2.imshow('img', img)
    c = cv2.waitKey(1)

    # way to quit programme
    if c & 0xFF == ord('q'):
        stop = True
        cv2.destroyAllWindows()
    '''