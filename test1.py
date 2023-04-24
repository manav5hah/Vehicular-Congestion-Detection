import glob
import cv2
import array
import matplotlib.pyplot as plt
import numpy
import os
from scipy.io import loadmat
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.applications import imagenet_utils
from keras import regularizers
from scipy.misc import toimage
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
from numpy import argmax

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
f = open("trafficdb/EvalSet_train", "r")
content_train = f.read()
f.close()
f = open("trafficdb/EvalSet_train", "r")
content_test = f.read()
f.close()
# print(content_test)
content_train = content_train.split("\n")
content_test = content_test.split("\n")
model1 = Sequential()
model1.add(Conv2D(32,input_shape =(64, 64, 3), kernel_size=(3,3), padding='same', activation='relu', activity_regularizer=regularizers.l1(0.00001)))
model1.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
# model1.add(Dense(3, activation='softmax'))
model1.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='relu', activity_regularizer=regularizers.l1(0.00001)))
model1.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
# model1.add(Dense(3, activation='softmax'))
model1.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu', activity_regularizer=regularizers.l1(0.00001)))
model1.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
# model1.add(Dense(3, activation='softmax'))
model1.add(Conv2D(256, kernel_size=(3,3), padding='same', activation='relu', activity_regularizer=regularizers.l1(0.00001)))
model1.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model1.add(Flatten())
# model1.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3),))
model1.add(Dense(3, activation='sigmoid', activity_regularizer=regularizers.l1(0.00001)))
# model1.add(Dense(3, activation='softmax'))
epochs = 3
lrate = 0.0005
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model1.compile(loss='mean_squared_error',optimizer=sgd, metrics=['accuracy'])
model1.summary() 

# model2 = Sequential()
# model2.add(Conv2D(32,input_shape =(64, 64, 3), kernel_size=(3,3), padding='same', activation='relu'))
# model2.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
# # model2.add(Dense(3, activation='softmax'))
# model2.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'))
# model2.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
# # model2.add(Dense(3, activation='softmax'))
# model2.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'))
# model2.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
# model2.add(Dense(3, activation='softmax'))


# print(trsets[0])
k = 0
imgpath = os.getcwd()
imgpath = os.path.join(imgpath, "data")
# for i in range(254):
# for j in range(1,4):
# # im = Image.open("data/"+str(i)+"/"+str(j)+".jpg")
# # newsize = (64, 64) 
# # im = im.resize(newsize)
# imgpath1 = os.path.join(imgpath, str(i))
# imgpath1 = os.path.join(imgpath1, str(j) + ".jpg")
# im = load_img(imgpath1, target_size = (64, 64))
# pixel[i][j] = img_to_array(im)
# toimage(pixel[21][3]).show()
y = loadmat('trafficdb/ImageMaster.mat')
# y1 = y['imagemaster'][0][i].tolist()[0][0][2].tolist()[0]
# print(len(trsets[0]))
for i in range(4):
    trsets = []
    tesets = []
    r = []
    # print(trsets[i][0])
    # t = [None in range(len(content_train[i].split(',')))]
    t = content_train[i].split(',')
    # print("     "+str(len(t)))
    for j in t:
        if j != '':
            r.append(int(j))
    # print(r)
    trsets = r
    t = content_test[i].split(',')
    for j in t:
        if j != '':
            r.append(int(j))
    # print(r)
    tesets = r
    train_x = []
    train_x1 = [i for  i in range(508)]
    train_y = []
    train_y1 = []
    test_x = []
    test_y = []
    test_y1 = []
    # print(len(trsets[i]))
    for j in trsets:
        # print(y['imagemaster'][0][j].tolist()[0][0][1].tolist()[0])
        name = y['imagemaster'][0][j].tolist()[0][0][1].tolist()[0]
        # tag1 = y['imagemaster'][0][j].tolist()[0][0][2][0]
        # tag2 = y['imagemaster'][0][j].tolist()[0][0][2][0]
        train_y.append(y['imagemaster'][0][j].tolist()[0][0][2][0])
        # train_y[i].append(y['imagemaster'][0][j].tolist()[0][0][2][0])
        name = os.path.join(imgpath, name)
        img1path = os.path.join(name, "1.jpg")
        # img2path = os.path.join(name, "3.jpg")
        im1 = load_img(img1path, target_size=(64, 64))
        # im2 = load_img(img2path, target_size=(64, 64))
        im1 = img_to_array(im1)
        # im2 = img_to_array(im2)
        train_x.append(im1)
        # train_x[i].append(im2)
    for j in tesets:
        # print(y['imagemaster'][0][j].tolist()[0][0][1].tolist()[0])
        name = y['imagemaster'][0][j].tolist()[0][0][1].tolist()[0]
        name = y['imagemaster'][0][j].tolist()[0][0][1].tolist()[0]
        # tag = y['imagemaster'][0][j].tolist()[0][0][2][0]
        test_y.append(y['imagemaster'][0][j].tolist()[0][0][2][0])
        # test_y[i].append(y['imagemaster'][0][j].tolist()[0][0][2][0])
        name = os.path.join(imgpath, name)
        img1path = os.path.join(name, "1.jpg")
        # img2path = os.path.join(name, "3.jpg")
        im1 = load_img(img1path, target_size=(64, 64))
        # im2 = load_img(img2path, target_size=(64, 64))
        im1 = img_to_array(im1)
        # im2 = img_to_array(im2)
        test_x.append(im1)
        # test_x[i].append(im2)
    # print(test_x)
    for j in range(len(trsets)):
        if train_y[j] == str("heavy"):
            train_y1.append(0)
        elif train_y[j] == str("medium"):
            train_y1.append(1)
        elif train_y[j] == str("light"):
            train_y1.append(2)
    train_y1 = numpy.array(train_y1)
    # print(train_y1)
    train_y1 = np_utils.to_categorical(train_y1, num_classes=3)
    for j in range(len(tesets)):
        if test_y[j] == str('heavy'):
            test_y1.append(0)
        elif test_y[j] == str('medium'):
            test_y1.append(1)
        elif test_y[j] == str('light'):
            test_y1.append(2)
    train_y1 = numpy.array(train_y1)
    test_y1 = np_utils.to_categorical(test_y1, num_classes=3)
    # print(argmax(np_utils.to_categorical(test_y1[i], 3)))
    # plt.plot(train_x1, train_y1)
    # plt.show()
    # print(train_y1)
    # train_x = numpy.vstack(train_x)

    # print(train_x)
    # print(train_y)
    # images = [cv2.imread(file) for file in glob.glob("C:/Users/manak/Documents/WC-Project/Keras/data/*.jpg")]
    print("SET :"+str(i+1))
    tr_x = numpy.array(train_x)
    tr_y = numpy.array(train_y1)
    te_x = numpy.array(test_x)
    te_y = numpy.array(test_y1)
    print(tr_x.shape)
    print(tr_y.shape)
    print(te_x.shape)
    print(te_y.shape)
    model1.fit(tr_x,tr_y,validation_data=(te_x, te_y), epochs=epochs)
    scores = model1.evaluate(te_x,te_y, verbose=0)
    feature = model1.predict(tr_x)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    # for m in feature:
    #     print(m)
    train_x.clear()
    train_x1.clear()
    train_y.clear()
    # train_y1.clear()
    test_x.clear()
    test_y.clear()
    # test_y1.clear()
    t.clear()
    trsets.clear()
    tesets.clear()