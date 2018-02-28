#!/usr/bin/python

import cv2, os
import numpy as np

from PIL import Image

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
            #cv2.waitKey(50)

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
                print "{} is Incorrect Recognized as {} {}".format(nbr_actual,nbr_predicted, conf)
            #cv2.imshow("Recognizing Face", predict_image[y: y+h, x:x+w])
            #cv2.waitKey(1000)


def main():

    yale_images, yale_labels = get_images_labels(yale_path,yale_nbr)
    cv2.destroyAllWindows()

    orl_images, orl_labels = get_images_labels(orl_path,orl_nbr)
    cv2.destroyAllWindows()


    recognizer.train(yale_images+orl_images, np.array(yale_labels+orl_labels))

    validation(yale_path,yale_nbr)
    validation(orl_path, orl_nbr)




if __name__ == "__main__":
    main()
