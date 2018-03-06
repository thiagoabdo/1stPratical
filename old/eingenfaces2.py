#!/usr/bin/python

import cv2, os
import numpy as np


from sklearn import preprocessing
from PIL import Image

import pdb

cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)


recognizer = cv2.createLBPHFaceRecognizer()

def yale_path(trainning=True):
    path = './yale_faces'
    imagePaths = [os.path.join(path,f) for f in  os.listdir(path) if not f.endswith('.sad') == trainning]
    return imagePaths

def yale_nbr(image):
    nbr = int(os.path.split(image)[1].split('.')[0].replace("subject",""))
    return nbr


def orl_path(trainning=True):
    path = './orl_faces'
    imagesPaths = [os.path.join(r,f) for r,d,files in os.walk(path) for f in files if not f.endswith('10.pgm') == trainning]
    return imagesPaths

def orl_nbr(image):
    nbr = int(os.path.split(os.path.split(image)[0])[1].replace("s","")) + 10000
    return nbr

def get_average(images,size):

    average = np.zeros(size, dtype='uint64')
    all_resized = []
    for img in images:
        resized = cv2.resize(img,size)
        average = average + resized
        all_resized.append(resized)
    average = average//len(images)
    average = average.astype('uint8')

   # cv2.imshow("Average Face", average)
   # cv2.waitKey(6000)

    return average,all_resized

def get_images_labels(get_path, get_nbr):
    image_paths = get_path()

    images = []
    labels = []

    for image_path in image_paths:
        image_pil = Image.open(image_path).convert('L')
        image = np.array(image_pil, 'uint8')
        nbr = get_nbr(image_path)

        faces = faceCascade.detectMultiScale(image) #Varias faces nao estao sendo detectadas, descomente as linhas abaixo para ver quais
   #     if faces == ():
   #         cv2.imshow("not adding faces to traning set...", image [y: y+h, x: x+w])
   #         cv2.waitKey(100)
        for (x, y, w, h) in faces:
            images.append(image[y: y+h, x:x +w])
            labels.append(nbr)
            #cv2.imshow("adding faces to traning set...", image [y: y+h, x: x+w])
            #cv2.waitKey(5)

    return images, labels



def validation(get_path, get_nbr):
    image_paths = get_path(trainning=False)

    for image_path in image_paths:
        predict_image_pil = Image.open(image_path).convert('L')
        predict_image = np.array(predict_image_pil, 'uint8')
        faces = faceCascade.detectMultiScale(predict_image)
        for (x, y,w ,h) in faces:
            nbr_predicted, conf = recognizer.predict(predict_image[y: y+h, x:x+w])
            nbr_actual = get_nbr(image_path)
            if nbr_actual == nbr_predicted:
                print "{} is Correctly reconized with confidence {}".format(nbr_actual, conf)
            else:
                print "{} is Incorrect Recognized as {} with confidence {}".format(nbr_actual,nbr_predicted, conf)
            #cv2.imshow("Recognizing Face", predict_image[y: y+h, x:x+w])
            #cv2.waitKey(1000)


def main():

    yale_images, yale_labels = get_images_labels(yale_path,yale_nbr)
    cv2.destroyAllWindows()
    yale_average, yale_resized = get_average(yale_images,(150,150))

    yaleAverageFlatten = yale_average.flatten()
    matrixA = np.zeros([len(yale_images),len(yaleAverageFlatten)])
    i=0
    for img in yale_resized:
        matrixA[i] = img.flatten() - yaleAverageFlatten
        i=i+1
    matrixAT = matrixA.transpose()
    matrixC = np.matmul(matrixA, matrixAT)

    eigen_values, eigen_vec = np.linalg.eig(matrixC)

    #eigen_vec = (eigen_vec/eigen_vec.max())

    eigen_values = eigen_values.real
    eigen_vec = eigen_vec.real

    biggest_value = [0,1,2,3,4]
    i=0
    for value in eigen_values:
        for j in range(0,len(biggest_value)):
            if value > biggest_value[j]:
                biggest_value[j] = i
                break
        i=i+1

    #pdb.set_trace()

    vectorK =[]
    for bg in biggest_value:
        real_eigen_vector  = np.matmul(matrixAT, eigen_vec[bg])
        vectorK.append(real_eigen_vector/np.linalg.norm(real_eigen_vector))
        #real_eigen_vector = preprocessing.minmax_scale(real_eigen_vector.astype('float64'), (0,255))
        #img = real_eigen_vector.reshape((150,150)).astype('uint8')
        #cv2.imshow("eigen_faces", img)
        #cv2.waitKey(5000) 
    vectorK = np.array(vectorK)
    cv2.destroyAllWindows()

    yale_comp =[]
    pdb.set_trace()
    for img in yale_resized:
        comp = np.matmul(vectorK,img.flatten())
        print comp
        yale_comp.append(comp)
#    recognizer.train(yale_images, np.array(yale_labels))
 #   validation(yale_path,yale_nbr)
#
#
#    orl_images, orl_labels = get_images_labels(orl_path,orl_nbr)
#    get_average(orl_images,(75,75))
#    cv2.destroyAllWindows()
#    recognizer.train(orl_images, np.array(orl_labels))
#    validation(orl_path, orl_nbr)




if __name__ == "__main__":
    main()
