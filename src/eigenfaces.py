#!/usr/bin/python
import cv2
import pdb
import databases


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

#    yale_images, yale_labels = get_images_labels(yale_path,yale_nbr)
#    cv2.destroyAllWindows()
#   yale_resized = get_resized(yale_images,(150,150))
#    yale_average = get_average(yale_resized)

#
#    eigen_faces(yale_resized,yale_average)
#
#    cv2.destroyAllWindows()
#    cv2.waitKey(1)

#    recognizer.train(yale_images, np.array(yale_labels))
#    validation(yale_path,yale_nbr)

    print "YALE"

    yale = databases.Yale("./yale_faces")
    yale.get_eigen_vectors(6)

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(10)
    cv2.waitKey(10)
    cv2.waitKey(10)
    cv2.waitKey(10)

    print "ORL"

    orl = databases.Orl("./orl_faces")
    orl.get_eigen_vectors(5)


if __name__ == "__main__":
    main()
