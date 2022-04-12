#!/usr/bin/env python
# coding: utf-8

# <h1 style="property:value;color:Turquoise;font-size:300%;text-align:center; ">CISC 867: Project 2</h1>
# <h3 style="property:value;color:Teal;font-size:200%; ">Fashion-MNIST</h3>  
# <h6 style="property:value;color:Sienna;font-size:150%; ">Kariman Karm Mohamed Mousaa </h6> 

# ## 1- loading and exploring data

# In[1]:


#Import Libararies
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# load the train and test data from the csv files
test = pd.read_csv("fashion-mnist_test.csv")
train = pd.read_csv("fashion-mnist_train.csv")
train.head()


# In[3]:


# Describe the data 
train.describe()


# In[4]:


# correlation between data 
train.corr()


# In[5]:


# the shape of data before the preprocessing 
print("train shape:{}\ntest shape:{}".format(train.shape,test.shape))


# In[6]:


#distribution for each category in label
sns.countplot(train["label"])


# In[7]:


# Percentage for each category in label
label = pd.DataFrame(train["label"].value_counts(),columns=["label"])
plt.pie(label["label"], labels=label.index ,autopct='%1.1f%%');
plt.show()


# In[8]:


# split data to X,y
X = train.drop("label",axis=1)
y = train["label"]


# In[9]:


# sample of one of the training images
img = X.iloc[0].values
img = img.reshape(28,28)
plt.imshow(img)
plt.axis("off")
plt.show()


# In[10]:


names = ["T-shirt","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(X.iloc[i].values.reshape((28,28)))
    index = int(y[i])
    plt.axis("off")
    plt.title(names[index])
plt.show() 


# In[11]:


from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from keras.utils.np_utils import to_categorical


# In[12]:


# data preprocessing
def data_preprocessing(raw):
    # transform the labels by one-hot-encoding
    out_y = to_categorical(raw.label, 10)
    num_images = raw.shape[0]
    x_as_array = raw.values[:,1:]
    #reshape the dataset into 4D array
    x_shaped_array = x_as_array.reshape(num_images, 28, 28, 1)
    #normalize value to [0,1]
    out_x = x_shaped_array / 255
    return out_x, out_y


# In[13]:


# prepare the data
from sklearn.model_selection import KFold
X, y = data_preprocessing(train)
X_test, y_test = data_preprocessing(test)
kf = KFold(n_splits=5)
for train_index, val_index in kf.split(X):
   x_train, x_val = X[train_index], X[val_index]
   y_train, y_val = y[train_index], y[val_index]


# In[14]:


# shape of the new data after the preprocessing 
print("x_train shape:{}\ny_train shape:{}".format(x_train.shape,y_train.shape))
print("x_test shape:{}\ny_test shape:{}".format(x_val.shape,y_val.shape))


# In[15]:


from sklearn.metrics import confusion_matrix
import itertools
import tensorflow as tf
from tensorflow import keras
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D,Input
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import keras.callbacks
from sklearn.model_selection import train_test_split


# ## LeNet5 model
# ##### why you think LeNet-5 further improves the accuracy if any at all And if it doesn't, why not?
# * the best to learn first as it is a simple and basic model architecture. it improve the accuracy because LeNet was used in detecting handwritten cheques on MNIST dataset.Fully connected networks and activation functions were previously known in neural networks. LeNet-5 introduced convolutional and pooling layers. LeNet-5 is believed to be the base for all other ConvNets.
# 

# In[16]:


# instantiate an empty model
model = Sequential()
# c1 convolutional layer 
model.add(Conv2D(filters=6, kernel_size=(5,5),strides=(1,1), padding='same', activation='tanh',input_shape=(28,28,1)))
# s2 pooling layer 
model.add(AveragePooling2D(pool_size=(2,2), strides=(1,1),padding='valid'))
#c3 convolutional layer 
model.add(Conv2D(filters=16, kernel_size=(5,5),strides=(1,1), padding='same', activation='tanh'))
# s4 pooling layer 
model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2),padding='valid'))
#c5 full connected convolutional layer 
model.add(Conv2D(filters=120, kernel_size=(5,5),strides=(1,1), padding='valid', activation='tanh'))
# flatten the cnn output so that we can connect it with full connected layers
model.add(Flatten())
#Fc6 full connected layer
model.add(Dense(84,activation='tanh'))
#output layer with softmax activation
model.add(Dense(10,activation="softmax"))
# optimizer adam with learning rate=.01 in second trial .001
opt = tf.optimizers.Adam(learning_rate=0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt,
metrics=["accuracy"])


# In[17]:


# fit model with batch_size=250 and epochs=50 then in trial2 batch_size=200,epochs=20
history= model.fit(x_train,
                   y_train,
                   batch_size=200,
                   validation_data=(x_val,y_val),
                   epochs=20, steps_per_epoch=x_train.shape[0]//250)


# In[18]:


model.summary()


# In[19]:


score = model.evaluate(x_val,y_val)
print("Accuracy:%{:.2f}".format(score[1]*100))


# In[20]:


#draw 
plt.plot(history.history["loss"],color="r",label="Loss");
plt.plot(history.history["accuracy"],color="b",label="Accuracy")
plt.title("Accuracy and Loss")
plt.legend()


# In[21]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


# ## CNN model

# In[22]:


cnn1 = Sequential()
cnn1.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
cnn1.add(BatchNormalization())

cnn1.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
cnn1.add(BatchNormalization())
cnn1.add(MaxPooling2D(pool_size=(2, 2)))
cnn1.add(Dropout(0.25))

cnn1.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
cnn1.add(BatchNormalization())
cnn1.add(Dropout(0.25))
cnn1.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
cnn1.add(BatchNormalization())
cnn1.add(MaxPooling2D(pool_size=(2, 2)))
cnn1.add(Dropout(0.25))

cnn1.add(Flatten())

cnn1.add(Dense(512, activation='relu'))
cnn1.add(BatchNormalization())
cnn1.add(Dropout(0.5))

cnn1.add(Dense(128, activation='relu'))
cnn1.add(BatchNormalization())
cnn1.add(Dropout(0.5))

cnn1.add(Dense(10, activation='softmax'))


# In[23]:


optm= tf.optimizers.Adam(lr=0.001, beta_1 = 0.9, beta_2=0.999)
cnn1.compile(optimizer=optm, loss = "categorical_crossentropy",metrics=["accuracy"])
hist= cnn1.fit(x_train,
                   y_train,
                   batch_size=200,
                   validation_data=(x_val,y_val),
                   epochs=20,
                   steps_per_epoch=x_train.shape[0]//250)


# In[24]:


score = cnn1.evaluate(x_val,y_val)
print("Accuracy:%{:.2f}".format(score[1]*100))


# In[25]:


#draw 
plt.plot(hist.history["loss"],color="r",label="Loss");
plt.plot(hist.history["accuracy"],color="b",label="Accuracy")
plt.title("Accuracy and Loss")
plt.legend()


# In[26]:


acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']

loss = hist.history['loss']
val_loss = hist.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


# 
# ## VGG16 model

# In[27]:


# X forms the training images, and y forms the training labels
X = np.array(train.iloc[:, 1:])
# Converting Labels to one hot encoded format
y = to_categorical(np.array(train.iloc[:, 0]))


# In[28]:


# Convert the images into 3 channels
train_X=np.dstack([X] * 3)
train_y= np.array (y)
train_X.shape


# In[29]:


# Reshape images as per the tensor format required by tensorflow
train_X = train_X.reshape(-1, 28,28,3)
train_X.shape


# In[30]:


# Resize the images 48*48 as required by VGG16
from keras.preprocessing.image import img_to_array, array_to_img
train_X = np.asarray([img_to_array(array_to_img(im, scale=False).resize((150,150))) for im in train_X])


# In[31]:


# Normalize the data and change data type
train_X = train_X/ 255
train_X = train_X.astype('float32')


# In[32]:


# Splitting train data as train and validation data
train_X,valid_X,train_y,val_y = train_test_split(train_X,y,test_size=0.2,random_state=13)
train_X.shape


# In[33]:


# Preprocessing the input 
from keras.applications.vgg16 import preprocess_input
# Preprocessing the input 
train_X = preprocess_input(train_X)
val_X = preprocess_input(valid_X)
train_X.shape[1:]


# In[34]:


#  Create base model of VGG16
# Define the parameters for instanitaing VGG16 model. 
IMG_WIDTH = 48
IMG_HEIGHT = 48
IMG_DEPTH = 3
BATCH_SIZE = 16
vgg = VGG16(weights='imagenet', include_top=False, input_shape = (150, 150, 3), classes = 10)
vgg.summary()


# In[35]:


# Extracting features
train_X = vgg.predict(np.array(train_X), batch_size=256, verbose=1)
val_X = vgg.predict(np.array(val_X), batch_size=256, verbose=1)


# In[36]:


# Saving the features so that they can be used for future
np.savez("train_features", train_X, train_y)
np.savez("val_features", val_X, val_y)
# Current shape of features
print(train_X.shape, "\n", val_X.shape)


# In[38]:


# Flatten extracted features
train_X = np.reshape(train_X, (3028, 4*4*512))
val_X = np.reshape(val_X, (758, 4*4*512))


# In[39]:


# Add Dense and Dropout layers on top of VGG16 pre-trained
model =Sequential()
model.add(Dense(512, activation='relu', input_dim=4 * 4 * 512))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))


# In[40]:


optm= tf.optimizers.Adam(lr=0.001, beta_1 = 0.9, beta_2=0.999)
model.compile(optimizer=optm, loss = "categorical_crossentropy",metrics=["accuracy"])


# In[41]:


# fit the model
history = model.fit(train_X, train_y,
          batch_size=256,
          epochs=50,
          verbose=1,
          validation_data=(val_X, val_y))


# In[42]:


score = model.evaluate(val_X,val_y)
print("Accuracy:%{:.2f}".format(score[1]*100))


# In[43]:


#draw 
plt.plot(history.history["loss"],color="r",label="Loss");
plt.plot(history.history["accuracy"],color="b",label="Accuracy")
plt.title("Accuracy and Loss")
plt.legend()


# In[45]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss =history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.show()


# In[ ]:




