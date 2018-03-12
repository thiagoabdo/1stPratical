#!/usr/bin/python
import cv2
import pdb
import databases

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

    #print "YALE"
    #yale = databases.Yale("./yale_faces")
    #accuracy = {}
    #for expression in yale.expressions:
    #    imagesTest=[img for img in yale.get_all_images() if not img.endswith(expression)]
    #    imagesValidate = [img for img in yale.get_all_images() if img.endswith(expression)]
    #    print "Testing for expression({}): {}".format(len(imagesValidate),expression[1:])
    #    yale.get_images(imagesTest)
    #    yale.get_eigen_vectors(50)
    #    yale.calculate_components()
    #    accuracy[expression[1:]] = yale.calculate_distances(yale.return_images_labels_validate(imagesValidate))

    #minAcc = min(accuracy,key=accuracy.get)
    #maxAcc = max(accuracy,key=accuracy.get)
    #print "\n\nWorst Accuracy in expression: {}({})\nBest Accuracy in expression {}({})".format(minAcc,accuracy[minAcc],maxAcc,accuracy[maxAcc])

    print "ORL"

    orl = databases.Orl("./orl_faces")
    orl.get_buckets()


#    orl = databases.Orl("./orl_faces")
#    orl.get_eigen_vectors(20)
#    orl.calculate_components()
#    orl.calculate_distances(orl.get_images_labels_validate())


if __name__ == "__main__":
    main()
