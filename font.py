# -*- coding: utf-8 -*-

# imports:
import cv2
import numpy as np
import h5py
import math
import os
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow
from tensorflow.keras.utils import to_categorical


# uploading the training, validation and test sets:
x_train = []
y_train = []
db = h5py.File('data/extra_train.h5', 'r')
im_names = list(db['data'].keys())
for k in range (len(im_names)):
        im = im_names[k]
        imgs = db['data'][im][:]
        font = db['data'][im].attrs['font']
        txt = db['data'][im].attrs['txt']
        charBB = db['data'][im].attrs['charBB']
        wordBB = db['data'][im].attrs['wordBB']


        font_name = ['Skylark', 'Ubuntu Mono', 'Sweet Puppy']
        nC = charBB.shape[-1]
        for b_inx in range(nC):
            if(font[b_inx].decode('UTF-8')==font_name[0]):
                color = 'r'
            elif(font[b_inx].decode('UTF-8')==font_name[1]):
                color = 'b'
            else:
                color = 'g'
        bb = charBB[:,:,b_inx]
        x = np.append(bb[0,:], bb[0,0])
        y = np.append(bb[1,:], bb[1,0])
        nW = wordBB.shape[-1]
        for i in range(len(charBB[0,0])):
            pts1 = np.float32([charBB[:,:,i].T[0],charBB[:,:,i].T[1],charBB[:,:,i].T[3],charBB[:,:,i].T[2]])
            pts2 = np.float32([[0,0],[400,0],[0,400],[400,400]])
            M = cv2.getPerspectiveTransform(pts1,pts2)
            dst = cv2.warpPerspective(imgs,M,(400,400))
            dst = cv2.pyrDown(cv2.pyrDown(dst))
            dst = (dst/255.).astype(np.float16)
            x_train.append(dst)
            y_train.append(font[i].decode('UTF-8'))

x_val = []
y_val = []
db = h5py.File('data/SynthText.h5', 'r')
im_names = list(db['data'].keys())
for k in range(len(im_names)):
        im = im_names[k]
        imgs = db['data'][im][:]
        font = db['data'][im].attrs['font']
        txt = db['data'][im].attrs['txt']
        charBB = db['data'][im].attrs['charBB']
        wordBB = db['data'][im].attrs['wordBB']


        font_name = ['Skylark', 'Ubuntu Mono', 'Sweet Puppy']
        nC = charBB.shape[-1]
        for b_inx in range(nC):
            if(font[b_inx].decode('UTF-8')==font_name[0]):
                color = 'r'
            elif(font[b_inx].decode('UTF-8')==font_name[1]):
                color = 'b'
            else:
                color = 'g'
        bb = charBB[:,:,b_inx]
        x = np.append(bb[0,:], bb[0,0])
        y = np.append(bb[1,:], bb[1,0])
        nW = wordBB.shape[-1]
        for i in range(len(charBB[0,0])):
            pts1 = np.float32([charBB[:,:,i].T[0],charBB[:,:,i].T[1],charBB[:,:,i].T[3],charBB[:,:,i].T[2]])
            pts2 = np.float32([[0,0],[400,0],[0,400],[400,400]])
            M = cv2.getPerspectiveTransform(pts1,pts2)
            dst = cv2.warpPerspective(imgs,M,(400,400))
            dst = cv2.pyrDown(cv2.pyrDown(dst))
            dst = (dst/255.).astype(np.float16)
            x_val.append(dst)
            y_val.append(font[i].decode('UTF-8'))

x_test = []
y_test = []
db = h5py.File('data/SynthText_val.h5', 'r')
im_names = list(db['data'].keys())
for k in range (len(im_names)):
        im = im_names[k]
        imgs = db['data'][im][:]
        font = db['data'][im].attrs['font']
        txt = db['data'][im].attrs['txt']
        charBB = db['data'][im].attrs['charBB']
        wordBB = db['data'][im].attrs['wordBB']


        font_name = ['Skylark', 'Ubuntu Mono', 'Sweet Puppy']
        nC = charBB.shape[-1]
        for b_inx in range(nC):
            if(font[b_inx].decode('UTF-8')==font_name[0]):
                color = 'r'
            elif(font[b_inx].decode('UTF-8')==font_name[1]):
                color = 'b'
            else:
                color = 'g'
        bb = charBB[:,:,b_inx]
        x = np.append(bb[0,:], bb[0,0])
        y = np.append(bb[1,:], bb[1,0])
        nW = wordBB.shape[-1]
        for i in range(len(charBB[0,0])):
            pts1 = np.float32([charBB[:,:,i].T[0],charBB[:,:,i].T[1],charBB[:,:,i].T[3],charBB[:,:,i].T[2]])
            pts2 = np.float32([[0,0],[400,0],[0,400],[400,400]])
            M = cv2.getPerspectiveTransform(pts1,pts2)
            dst = cv2.warpPerspective(imgs,M,(400,400))
            dst = cv2.pyrDown(cv2.pyrDown(dst))
            dst = (dst/255.).astype(np.float16)
            x_test.append(dst)
            y_test.append(font[i].decode('UTF-8'))

# preprocessing:
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_val = np.asarray(x_val)
y_val = np.asarray(y_val)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

def to_numeric(x):
    return [0 if i=='Ubuntu Mono' else 1 if i=='Skylark' else 2 for i in x]
y_train = to_numeric(y_train)
y_val = to_numeric(y_val)
y_test = to_numeric(y_test)


y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes=3)
y_val = tensorflow.keras.utils.to_categorical(y_val, num_classes=3)
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes=3)

y_train = tensorflow.constant(y_train, shape=[26070,3])
y_val = tensorflow.constant(y_val, shape=[12238, 3])
y_test = tensorflow.constant(y_test, shape=[8198, 3])

tensorflow.dtypes.cast(y_train, tensorflow.uint8)
tensorflow.dtypes.cast(y_val, tensorflow.uint8)
tensorflow.dtypes.cast(y_test, tensorflow.uint8)


# the model

def initialize_model():
    model = tensorflow.keras.models.Sequential([
    # Note the input shape is the desired size of the image 100x100 with 3 bytes color
    # This is the first convolution
    tensorflow.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu', input_shape=(100, 100,3)),
    tensorflow.keras.layers.MaxPooling2D(2, 2),
    tensorflow.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu'),
    tensorflow.keras.layers.MaxPooling2D(2,2),
    tensorflow.keras.layers.Conv2D(256, (3,3), padding='same', activation='relu'),
    tensorflow.keras.layers.MaxPooling2D(2,2),
    tensorflow.keras.layers.Flatten(),
    tensorflow.keras.layers.Dropout(0.5),
    tensorflow.keras.layers.Dense(512, activation='relu'),
    tensorflow.keras.layers.Dense(3, activation='softmax')
])
    return model

def compile_model(model):
    model.compile(loss='categorical_crossentropy',
                    optimizer=opt,
                    metrics=['accuracy'])
    return model

model = initialize_model()
opt = tensorflow.optimizers.Nadam(learning_rate=0.0016)

model = compile_model(model)

es = EarlyStopping(patience=7,restore_best_weights=True, verbose=1)

history = model.fit(x_train, y_train,
                      validation_data = (x_val, y_val),
                      callbacks=[es],
                      epochs=250,
                      batch_size=8)
score = model.evaluate(x_test, y_test, verbose=False)
print('Loss: {}'.format(score[0]))
print('Accuracy: {}%'.format(np.round(10000*score[1])/100))


