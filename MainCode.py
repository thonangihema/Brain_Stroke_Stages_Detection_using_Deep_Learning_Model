import numpy as np 
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras import Input


from tensorflow.keras.layers import Dense, Dropout, Conv2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.utils import to_categorical
from numpy import argmax
from sklearn.metrics import confusion_matrix, classification_report
from mlxtend.plotting import plot_confusion_matrix


#Input Data
path = 'C:/Users/Hema/Desktop/Major Project(Brain Stroke)/Sourcecode/Dataset/'

# categories
categories = ['Hemorrhagic','Ischemic','no_stroke','TIA'  ]

# let's display some of the pictures
for category in categories:
    fig, _ = plt.subplots(3,3)
    fig.suptitle(category)
    fig.patch.set_facecolor('xkcd:white')
    for k, v in enumerate(os.listdir(path+category)[:9]):
        img = plt.imread(path+category+'/'+v)
        plt.subplot(3, 3, k+1)
        plt.axis('off')
        plt.imshow(img)
        cv2.imshow(' Image',img)
        cv2.waitKey(0)  
        cv2.destroyAllWindows()
#plt.show()
    
shape0 = []
shape1 = []

for category in categories:
    for files in os.listdir(path+category):
        shape0.append(plt.imread(path+category+'/'+ files).shape[0])
        shape1.append(plt.imread(path+category+'/'+ files).shape[1])
    print(category, ' => height min : ', min(shape0), 'width min : ', min(shape1))
    print(category, ' => height max : ', max(shape0), 'width max : ', max(shape1))
    shape0 = []
    shape1 = []
    
# initialize the data and labels
data = []
labels = []
imagePaths = []
HEIGHT = 65
WIDTH = 65
N_CHANNELS = 3

# grab the image paths and randomly shuffle them
for k, category in enumerate(categories):
    for f in os.listdir(path+category):
        imagePaths.append([path+category+'/'+f, k]) 

import random
random.shuffle(imagePaths)
#print(imagePaths[:10])

# loop over the input images
for imagePath in imagePaths:
# load the image, resize the image to be HEIGHT * WIDTH pixels (ignoring aspect ratio) and store the image in the data list
    image = cv2.imread(imagePath[0])
    image = cv2.resize(image, (WIDTH, HEIGHT))  # .flatten()
    data.append(image)
    
    # extract the class label from the image path and update the
    # labels list
    label = imagePath[1]
    labels.append(label)
    
# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# Let's check everything is ok
fig, _ = plt.subplots(3,3)
fig.suptitle("Sample Input")
fig.patch.set_facecolor('xkcd:white')
for i in range(9):
    plt.subplot(3,3, i+1)
    plt.imshow(data[i])
    plt.axis('off')
    cv2.imshow(' Image',data[1])
    cv2.waitKey(0)  
    cv2.destroyAllWindows()
#    plt.title(categories[labels[i]])
#plt.show()

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)
# Preprocess class labels
trainY =to_categorical(trainY, 4)

print(trainX.shape)
print(testX.shape)
print(trainY.shape)
print(testY.shape)

import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the color CT scan image
image = cv2.imread('C:/Users/Hema/Desktop/Major Project(Brain Stroke)/Sourcecode/Dataset/Hemorrhagic/Tr-me_0010.jpg')

# Convert the image from BGR to RGB format
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Reshape the image to a 2D array of pixels
pixels = image.reshape(-1, 3)

# Define the number of clusters (i.e., segments)
n_clusters = 3

# Fit the k-means model to the pixel data
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(pixels)

# Get the cluster centroids and labels
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# Reshape the labels array to the original image shape
segmented_image = labels.reshape(image.shape[:-1])

# Display the original and segmented images
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(122)
plt.imshow(segmented_image, cmap='jet')
plt.title(' Segmented Image')
plt.axis('off')

plt.tight_layout()
plt.show()


#Classification
'''DENSENET121'''

def build_densenet():
    densenet = DenseNet121(weights='imagenet', include_top=False)

    input = Input(shape=(HEIGHT, WIDTH, N_CHANNELS))
    x = Conv2D(3, (3, 3), padding='same')(input)
    
    x = densenet(x)
    
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    # multi output
    output = Dense(4,activation = 'softmax', name='root')(x)
 

    # model
    model = Model(input,output)
    
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    model.summary()
    
    return model

model = build_densenet()

model.fit(trainX, trainY, batch_size=32, epochs=20, verbose=1)

history=model.history.history
#Plotting the accuracy
train_loss = history['loss']
train_acc = history['accuracy']

    # Performance graph
plt.figure()
plt.plot(train_loss, label='Loss')
plt.plot(train_acc, label='Accuracy')
plt.title('Performance Plot')
plt.legend()
plt.show()

print("Accuracy of the  Testing  CNN Algorithm is:",model.evaluate(trainX,trainY)[1]*100, "%")
#Analytic Results
pred = model.predict(testX)
predictions = argmax(pred, axis=1) 
print('Classification Report')
cr=classification_report(testY, predictions,target_names=categories)
print(cr)
print('Confusion Matrix')
cm = confusion_matrix(testY, predictions)
print(cm)
#Confusion Matrix Plot
plt.figure()
plot_confusion_matrix(cm,figsize=(15,15), class_names = categories,
                      show_normed = True);

plt.title( "Model confusion matrix")
plt.style.use("ggplot")
plt.show()


#Prediction
from tkinter import filedialog
test_data=[]
Image = filedialog.askopenfilename()
head_tail = os.path.split(Image)
fileNo=head_tail[1].split('.')
test_image_o = cv2.imread(Image)
test_image = cv2.resize(test_image_o, (WIDTH, HEIGHT))
#test_data.append(test_image)
# scale the raw pixel intensities to the range [0, 1]
test_data = np.array(test_image, dtype="float") / 255.0
test_data=test_data.reshape([-1,65, 65, 3])
pred = model.predict(test_data)
predictions = argmax(pred, axis=1) # return to label
print ('Prediction : '+categories[predictions[0]])
#Imersing into the plot
fig = plt.figure()
fig.patch.set_facecolor('xkcd:white')
plt.title(categories[predictions[0]])
plt.imshow(test_image_o)




