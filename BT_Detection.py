#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt


# In[2]:


IMAGE_SIZE = [224,224]


# In[3]:


train_path = '/dataset/brain_tumor_dataset/Train'
valid_path = '/dataset/brain_tumor_dataset/Test'


# In[4]:


vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)


# In[5]:


for layer in vgg.layers:
    layer.trainable = False


# In[6]:


folders = glob('/dataset/brain_tumor_dataset/Train/*')


# In[7]:


vgg.layers


# In[8]:


from keras_preprocessing.image import ImageDataGenerator


# In[9]:


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('/dataset/brain_tumor_dataset/Train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('/dataset/brain_tumor_dataset/Test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')


# In[10]:


num_pixels = IMAGE_SIZE[1]*IMAGE_SIZE[1]
num_pixels


# In[11]:


def base_model():
    x = Flatten()(vgg.output)
    x = Dense(units=2,input_dim=num_pixels, activation='relu')(x)
    top_model = Dense(len(folders), activation='softmax')(x)
    model = Model(inputs=vgg.input, outputs=top_model)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# In[14]:


model = base_model()


# In[15]:


model.summary()



r=model.fit_generator(training_set,
  validation_data=test_set,
  epochs=1,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set))

model.save('braintumour_new_model2.h5')


test_accuracy=r.history['val_acc'][0]


accuracy = test_accuracy*100

file1=open("result.txt","w")


file1.write(str(accuracy))


file1.close()






