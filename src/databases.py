#!/usr/bin/python

import cv2,os
import numpy as np

from sklearn import preprocessing
from PIL import Image

import pdb

cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)


recognizer = cv2.createLBPHFaceRecognizer()

class DataBase():

    def __init__(self, path):
        self.images = []
        self.labels = []
        self.resizedImages = []
        self.averageImage = None
        self.averageSize = None
        self.eigenVectors = []
        self.path = path

    def get_images_path(self,trainning=True):
        return None
    def get_nbr(self,images):
        return None

    def get_images(self,trainning=True):
        if not self.images:
            for img in self.get_images_path(trainning):
                imagePil = Image.open(img).convert('L')
                image = np.array(imagePil, 'uint8')

                faces = faceCascade.detectMultiScale(image)
                for (x, y, w, h) in faces:
                    self.images.append(image[y: y+h, x:x+w])
                    self.labels.append(self.get_nbr(img))
                    #cv2.imshow("Adding Faces", image[y: y+h, x: x+w])
                    #cv2.waitKey(50)
        return self.images

    def get_labels(self,trainning=True):
        if not self.labels:
            self.get_images(trainning)
        return self.labels

    def get_average_size(self):
        if self.averageSize is None:
            if not self.images:
                self.get_images()

            size = np.array((0,0))
            for img in self.images:
                size = size + img.shape
            size = size/len(self.images)
            self.averageSize=tuple(size)
        return self.averageSize

    def get_resized_images(self):
        if not self.resizedImages:
            size = self.get_average_size()
            for img in self.images:
                resized = cv2.resize(img,size)
                self.resizedImages.append(resized)
        return self.resizedImages

    def get_average_face(self):
        if self.averageImage is None:
            resized_images = self.get_resized_images()
            average = np.zeros(self.get_average_size(), dtype='uint64')
            for img in resized_images:
                average = average + img
            average = average//len(resized_images)
            average = average.astype('uint8')
            self.averageImage = average

            cv2.imshow("Average Face", average)
            cv2.waitKey(6000)
        return self.averageImage

    def get_eigen_vectors(self, k):
        if len(self.eigenVectors) > k:
                return self.eigenVectors[:k]
        imgSize = self.get_average_size()
        imgAverage = self.get_average_face().flatten()
        images = self.get_resized_images()

        matrixA = np.zeros([len(images),len(imgAverage)])
        i=0
        for img in images:
            matrixA[i] = img.flatten() - imgAverage
            i = i + 1
        matrixAT = matrixA.transpose()

        matrixC = np.matmul(matrixA, matrixAT)

        eigenVal, eigenVec = np.linalg.eig(matrixC)

        eigenVal = eigenVal.real
        eigenVec = eigenVec.real

        sortedEigenValIndex = eigenVal.argsort()[::-1]
        vectorK = []
        for eigenIndex in sortedEigenValIndex[:k]:
            autoEigenVec = np.matmul(matrixAT, eigenVec[eigenIndex])
            vectorK.append(autoEigenVec/np.linalg.norm(autoEigenVec))
            #autoEigenVec = preprocessing.minmax_scale(autoEigenVec, (0,255))
            #img = autoEigenVec.reshape(imgSize).astype('uint8')
            #cv2.imshow("EigenFaces",img)
            #cv2.waitKey(5000)
        self.eigenVectors = np.array(vectorK)

        return self.eigenVectors





class Yale(DataBase):
    def get_images_path(self,trainning=True):
        imagePaths = [os.path.join(self.path,f) for f in  os.listdir(self.path) if not f.endswith('.sad') == trainning]
        return imagePaths

    def get_nbr(self,image):
        nbr = int(os.path.split(image)[1].split('.')[0].replace("subject",""))
        return nbr

class Orl(DataBase):
    def get_images_path(self,trainning=True):
        imagesPaths = [os.path.join(r,f) for r,d,files in os.walk(self.path) for f in files if not f.endswith('10.pgm') == trainning]
        return imagesPaths

    def get_nbr(self,image):
        nbr = int(os.path.split(os.path.split(image)[0])[1].replace("s","")) + 10000
        return nbr

