from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

l_r = 1e-4 #initial learning rate keep it as low as you can for more accurated results
EPOCHS = 20
BS= 32 #batch size

directory = r"D:\Face-Mask-Detection-master\dataset"
categ = ["with_mask", "without_mask"]

print("loading images")

data = [] #data array will store the image no.s present in the dataset
labels = [] #will contain corresponding labels i.e. with or without mask of every image present in data array

for cat in categ:
    path = os.path.join(directory , cat) #its joining the paths of categories and directory
    for img in os.listdir(path):   #os.listdr(path) -  its kinds of list all the images in the directory
        img_path = os.path.join(path , img)    #we are joining the path of the image with its directory and category
        image = load_img(img_path , target_size= (224 , 224))   #preprcessing the images to a particular target size using the libtrary from tensorflow
        image = img_to_array(image)    #converting the image to number array
        image = preprocess_input(image)   #something related to mobile net

        data.append(image)    #appending the image number array to the data array
        labels.append(cat)     # appending corresponding with or without mask label

lb = LabelBinarizer()     #this is a library from sklearn and is used to reduce the characters to 0's and 1's for easy access
labels = lb.fit_transform(labels)   #fitting the labels
labels = to_categorical(labels)

data = np.array(data , dtype="float32")  #converting the lists to numpy arrays
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data , labels , test_size= 0.20 , stratify=labels , random_state=42)


#in this we are gonna take input then pass it on the mobile net which is competetor to convolutional neural network which we usually use for
#image processing , mobile net takes less parameters , then we are gonna perform max pooling flatten the image and then connected components and then output

#using mobile net we are gonna create two models one is the mobile net model whose output we are gonna pass in the normal model , we can call them head and base model resp.

aug = ImageDataGenerator(   #image data generator creates different image interpretations for every image like tilting , rotating it , etc in short it creates additional dataset
    rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest"
)

# load the MobileNetV2 network, ensuring the head FC layer sets are left off
#mobile net has already some pre trained model one of them is imagenet and we are using its pre setted weights for better accuracy
# include top is if to include the fully connected layer in the model , we are gonna connect the full connected layer by ourselves later so here we are setting it false
#shape is nothing jst the shape of the image that is height width and the no. of channels that are = 3 which is RGB
basemodel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

#now we are gonna create our fully connected layer using poolinh

headmodel = basemodel.output #creating a head model object and passing the basemodel's output in it
headmodel = AveragePooling2D(pool_size= (7,7))(headmodel)
headmodel = Flatten(name = "Flatten")(headmodel)  #falting these layers
headmodel = Dense(128 , activation="relu")(headmodel)   #adding a layer with 128 neurons activation layer is relu which means to go tob activation function for non linear cases 
headmodel = Dropout(0.5)(headmodel) #avoid overfitting
headmodel = Dense(2 , activation="softmax")(headmodel)  #output layer with two layers , go with softmax or sigmoid for output as it is based on probability

# place the head FC model on top of the base model (this will become the actual model we will train)
# this model function takes to parameters one is input and other is output

model = Model(inputs=basemodel.input, outputs=headmodel) #we are gonna take input from basemodel and output from headmodel

# loop over all layers in the base model and freeze them so they will *not* be updated during the first training process
# as they are jst our replacement for our cnn so we are freezing them for jst training
for layer in basemodel.layers:
    layer.trainable = False

#we are compiling our model
print("compiling model")
opt = Adam(lr = l_r , decay = l_r/EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,   #here we are using the adam optimizer which is similar to relu which is like go to optimizer
	metrics=["accuracy"])           # loss function is binary cross entropy is used and accuracy is the only meterics we are gonna calculate over here

print("training head...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),     # we are also gonna use the image generator data too as we have a small dataset for training
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

#making prediction for the data
print("predicting the data")
pred = model.predict(testX , batch_size = BS)

# for each image in the testing set we need to find the index of the label with corresponding largest predicted probability
pred = np.argmax(pred, axis=1)

#show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), pred,
	target_names=lb.classes_))

#serialize the model and save
print("saving mask detection model")
model.save("mask_detector.model", save_format="h5")

#plot training the loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")

