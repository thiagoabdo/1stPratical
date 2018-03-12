#!/usr/bin/python

import cv2,os
import numpy as np

from sklearn import preprocessing
from PIL import Image

import pdb

cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)


recognizer = cv2.createLBPHFaceRecognizer()

def euclidian_dist(x,y):
        return np.sqrt(np.sum((x-y)**2))


class DataBase(object):
    def __init__(self,path):
        self.images = []
        self.labels = []
        self.resizedImages = []
        self.eigenComponents = []
        self.averageImage = None
        self.averageSize = None
        self.eigenVectors = []
        self.path = path

    def reset(self):
        self.images = []
        self.labels = []
        self.resizedImages = []
        self.eigenComponents = []
        self.averageImage = None
        self.averageSize = None
        self.eigenVectors = []


    def get_images_path(self,trainning=True):
        return None
    def get_nbr(self,images):
        return None

    def get_images(self,image_paths):
        self.reset()
        for img in image_paths:
            imagePil = Image.open(img).convert('L')
            image = np.array(imagePil, 'uint8')

            faces = faceCascade.detectMultiScale(image)
            for (x, y, w, h) in faces:
                self.images.append(image[y: y+h, x:x+w])
                self.labels.append(self.get_nbr(img))
                #cv2.imshow("Adding Faces", image[y: y+h, x: x+w])
                #cv2.waitKey(50)
            if faces == ():
                self.images.append(image)
                self.labels.append(self.get_nbr(img))

        return self.images

    def get_labels(self,trainning=True):
        if not self.labels:
            print "Need to Get images first"
            return
        return self.labels

    def return_images_labels_validate(self,images_path):
        images=[]
        labels=[]
        for img in images_path:
            imagePil = Image.open(img).convert('L')
            image = np.array(imagePil, 'uint8')
            faces = faceCascade.detectMultiScale(image)
            for (x, y, w, h) in faces:
                images.append(cv2.resize(image[y: y+h, x:x+w],self.get_average_size()))
                labels.append(self.get_nbr(img))
            if faces == ():
                images.append(cv2.resize(image,self.get_average_size()))
                labels.append(self.get_nbr(img))
        return (images,labels)

    def get_average_size(self):
        if self.averageSize is None:
            if not self.images:
                print "Need to Get Images first"
                return

            size = np.array((0,0))
            for img in self.images:
                size = size + img.shape
            size = size/len(self.images)
            self.averageSize=tuple((size[0],size[0]))
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

            #cv2.imshow("Average Face", average)
            #cv2.waitKey(6000)
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

    def calculate_components(self):
        self.get_resized_images()
        imgAverage = self.get_average_face().flatten()
        for img in self.resizedImages:
            imgComponents = np.matmul(self.eigenVectors,(img.flatten() - imgAverage))
            self.eigenComponents.append(imgComponents)
        compLabel = zip(self.labels,self.eigenComponents)
        internDist = {}
        count = {}
        for iLabel,iComp in compLabel:
            for jLabel,jComp in compLabel:
                if iLabel == jLabel:
                    dist=(euclidian_dist(iComp,jComp))
                    internDist[iLabel] = internDist.get(iLabel,0) + dist
                    count[iLabel] = count.get(iLabel,0) + 1
        maxDist = 0
        minDist = 9999999999999999
        for i in internDist:
            dist = internDist[i]/count[i]
            minDist = min(minDist,dist)
            maxDist = max(maxDist,dist)

        print "  Inside subject maxDist: {} minDist: {}".format(maxDist,minDist)


        return self.eigenComponents

    def calculate_distances(self,(images,labels)):
        imagesLabel =zip(labels,images)
        compLabel = zip(self.labels,self.eigenComponents)
        imgAverage = self.get_average_face().flatten()
        #print imagesLabel
        #print ""
        #print labels
        #maiorInside = 0
        #menorOutside = 999999999999999
        testCase=0
        acertos=0
        distMedAc=0
        distMedEr=0
        for nbr,img in imagesLabel:
            distTo = {}
            count = {}
         #   mediaMesmoSuj=0
         #   qtdMesmoSuj=0
         #   mediaOutroSuj=0
         #   qtdOutroSuj=0
         #   minimalToSub= 9999999999999999999
         #   minimalSubject = 0
            imgComponents = np.matmul(self.eigenVectors, (img.flatten() - imgAverage))
            for dbNbr,dbComp in compLabel:
                dist=(euclidian_dist(imgComponents,dbComp))
                distTo[dbNbr] = distTo.get(dbNbr,0) + dist
                #distTo[dbNbr] = min(distTo.get(dbNbr,999999999999),dist)
                count[dbNbr] = count.get(dbNbr,0) + 1
            minimal = 9999999999
            ident = 0
            for i in distTo:
                distToEachOne = distTo[i]/count[i]
                #distToEachOne = distTo[i]
                if minimal > distToEachOne:
                    minimal = distToEachOne
                    ident = i
            #print "Subject {} identified as {} with distance {}".format(nbr,ident,minimal)
            testCase +=1
            if ident == nbr:
                acertos += 1
                distMedAc += minimal
            else:
                distMedEr += minimal

        #print "Testes: {} Accuracy: {} DistMedAc: {} DistMedErr: {}".format(testCase,acertos/float(testCase),distMedAc/acertos,distMedEr/(testCase-acertos))
        if( testCase == acertos):
            print "  Testes: {} Accuracy 100% DistMedAc: {}".format(testCase, distMedAc/testCase)
        elif (acertos == 0):
            print "  Testes: {} Accuracy 0%".format(testCase)
        else:
            print "  Testes: {} Accuracy: {} DistMedAc: {} DistMedErr: {}".format(testCase,acertos/float(testCase),distMedAc/acertos,distMedEr/(testCase-acertos))
         #       if minimalToSub >= dist:
         #           minimalToSub = dist
         #           minimalSubject = dbNbr
         #   mediaMesmoSuj /= qtdMesmoSuj
         #   mediaOutroSuj /= qtdOutroSuj
         #   maiorInside = max(maiorInside, mediaMesmoSuj)
         #   menorOutside = min(menorOutside, mediaOutroSuj)
            #print "For subject {} we have internal distance of {}({}), to others {}({}) and identifyed1 {}".format(nbr, mediaMesmoSuj, qtdMesmoSuj, mediaOutroSuj, qtdOutroSuj, minimalSubject)

        #print "Maior distancia dentre mesmo sujeito",str(maiorInside)
        #print "Menor distancia entre sujeitos diferentes", str(menorOutside)

        return (acertos/float(testCase))








class Yale(DataBase):
    def __init__(self,path):
        self.imagesPaths = []
        self.expressions = [".centerlight",".glasses",".happy",".leftlight",".noglasses",".normal",".rightlight",".sad",".sleepy",".surprised",".wink"]
        super(Yale,self).__init__(path)

    def get_images_path(self,trainning=True):
        imagePaths = [os.path.join(self.path,f) for f in  os.listdir(self.path) if not f.endswith('.sad') == trainning]
        return imagePaths

    def get_all_images(self):
        if not self.imagesPaths:
            self.imagesPaths = [os.path.join(self.path,f) for f in  os.listdir(self.path)]
        return self.imagesPaths

    def get_nbr(self,image):
        nbr = int(os.path.split(image)[1].split('.')[0].replace("subject",""))
        return nbr

class Orl(DataBase):
    def __init__(self,path):
        self.imagesPaths = []
        super(Orl, self).__init__(path)

    def get_all_images(self):
        if not self.imagesPaths:
            self.imagesPaths = [os.path.join(r,f) for r,d,files in os.walk(self.path) for f in files]
        return self.imagesPaths

    def get_buckets(self):
        self.get_all_images()
        imgPerSub = {}
        for img in self.imagesPaths:
            if not imgPerSub.get(self.get_nbr(img)):
                imgPerSub[self.get_nbr(img)] = []
            imgPerSub[self.get_nbr(img)].append(img)

        print imgPerSub

    def get_images_path(self,trainning=True):
        imagesPaths = [os.path.join(r,f) for r,d,files in os.walk(self.path) for f in files if not f.endswith('10.pgm') == trainning]

    def get_nbr(self,image):
        nbr = int(os.path.split(os.path.split(image)[0])[1].replace("s","")) + 10000
        return nbr

