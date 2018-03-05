#!/usr/bin/python

import cv2, os
import numpy as np

from PIL import Image

import databases.yale

cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)


recognizer = cv2.createLBPHFaceRecognizer()

def get_average(images,size):

    average = np.zeros(size, dtype='uint64')
    for img in images:
        resized = cv2.resize(img,size)
        average = average + resized
    average = average//len(images)
    average = average.astype('uint8')

    cv2.imshow("Average Face", average)
    cv2.waitKey(6000)

    return average

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
            cv2.imshow("adding faces to traning set...", image [y: y+h, x: x+w])
            cv2.waitKey(50)

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
    
    yale = Yale()
    
    yale_images, yale_labels = get_images_labels(yale.get_images_path,yale.get_nbr)
    get_average(yale_images,(150,150))
    cv2.destroyAllWindows()
    recognizer.train(yale_images, np.array(yale_labels))
    validation(yale_path,yale_nbr)


    orl_images, orl_labels = get_images_labels(orl_path,orl_nbr)
    get_average(orl_images,(75,75))
    cv2.destroyAllWindows()
    recognizer.train(orl_images, np.array(orl_labels))
    validation(orl_path, orl_nbr)




if __name__ == "__main__":
    main()
