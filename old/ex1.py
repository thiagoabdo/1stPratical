import cv2, os
import numpy as np
from PIL import Image
from sklearn import preprocessing

def get_images_and_labels(path):
    # Append all the absolute image paths in a list image_paths
    # We will not read the image with the .sad extension in the training set
    # Rather, we will use them to test our accuracy of the training
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if not f.endswith('.sad')]
    # images will contains face images
    images = []
    # labels will contains the label that is assigned to the image
    labels = []
    m_altura = 0
    m_largura = 0
    for image_path in image_paths:
        # Read the image and convert to grayscale
        image_pil = Image.open(image_path).convert('L')
        # Convert the image format into numpy array
        image = np.array(image_pil, 'uint8')
        # Get the label of the image
        nbr = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
        # Detect the face in the image
        faces = faceCascade.detectMultiScale(image)
        # If face is detected, append the face to images and the label to labels
        for (x, y, w, h) in faces:
            if h > m_altura:
                m_altura = h
            if w > m_largura:
                m_largura = w
            images.append(image[y: y + h, x: x + w])
            labels.append(nbr)
            # cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
            # cv2.waitKey(50)
    # return the images list and labels list
    #print("altura:{}".format(m_altura))
    #print("largura:{}".format(m_largura))
    return images, labels


def get_mean(images):
    g_w, g_h = 170, 170

    mean =  np.zeros((g_h, g_w), dtype='uint64')
    all_resized = []
    qtd_img=0
    for img in images:
        qtd_img = qtd_img + 1
        resized = cv2.resize(img,(g_w,g_h))
        mean = mean + resized
        all_resized.append(resized)
    mean = mean//qtd_img
    mean = mean.astype('uint8')
    # cv2.imshow("Mean(not girls)", mean)
    # cv2.waitKey(5000)
    return mean, all_resized






cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

recognizer = cv2.createLBPHFaceRecognizer()


# Path to the Yale Dataset
path = 'yale_faces'
# The folder yalefaces is in the same folder as this python script
# Call the get_images_and_labels function and get the face images and the
# corresponding labels
images, labels = get_images_and_labels(path)
mean, images = get_mean(images)

mean_v = mean.flatten()

res = np.zeros([len(images), len(mean_v)])
i=0
for img in images:
    res[i] = img.flatten() - mean_v
    i=i+1

#TODO: fix multiplication - w and h
res_t = res.transpose()

#print(len(res_t))
#print(len(res))
covariance  = np.matmul (res, res_t)

eigen_values, eigen_vectors = np.linalg.eig(covariance)

eigen_values = eigen_values.real

biggest_value = [0,0,0,0,0]
i=0
for value in eigen_values:
    for j in range(0,len(biggest_value)):
        if value > biggest_value[j]:
            biggest_value[j] = i
            break
    i=i+1


for bg in biggest_value:
    real_eigen_vector  = np.matmul(res_t, eigen_vectors[bg]).real

    real_eigen_vector = preprocessing.minmax_scale(real_eigen_vector, (0,255))
    preprocessing.normalize(real_eigen_vector)
   
    img = real_eigen_vector.reshape((170,170)).astype("uint8")
    #print(img)
    cv2.imshow("eigen_faces", img)
    cv2.waitKey(500)
# for i in range(0,eigen_vectors.shape[0]):
#     real_eigen_vector  = np.matmul(res_t, eigen_vectors[i])
#     img = real_eigen_vector.reshape((170,170)).real
#     # cv2.imshow("eigen_faces", img)
#     # cv2.waitKey(5000)

    




# cv2.destroyAllWindows()

# # Perform the training
# recognizer.train(images, np.array(labels))


# # Append the images with the extension .sad into image_paths
# image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.sad')]
# for image_path in image_paths:
#     predict_image_pil = Image.open(image_path).convert('L')
#     predict_image = np.array(predict_image_pil, 'uint8')
#     faces = faceCascade.detectMultiScale(predict_image)
#     for (x, y, w, h) in faces:
#         nbr_predicted, conf = recognizer.predict(predict_image[y: y + h, x: x + w])
#         nbr_actual = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
#         if nbr_actual == nbr_predicted:
#             print "{} is Correctly Recognized with confidence {}".format(nbr_actual, conf)
#         else:
#             print "{} is Incorrectly Recognized as {}".format(nbr_actual, nbr_predicted)
#         cv2.imshow("Recognizing Face", predict_image[y: y + h, x: x + w])
#         cv2.waitKey(1000)
