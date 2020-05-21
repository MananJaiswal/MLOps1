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
                         samples_per_epoch = 64,
                         nb_epoch = 1,
                         validation_data = test_set,
                         nb_val_samples = 32)

# In[20]:


r.history


# In[21]:


train_accuracy=r.history['acc'][-1]
train_accuracy


# In[22]:


test_accuracy=r.history['val_acc'][-1]
test_accuracy


# In[23]:


import tensorflow as tf

from keras.models import load_model


# In[24]:


test_accuracy = test_accuracy*100
accuracy=test_accuracy
accuracy


# In[25]:


model.save('braintumour_new_model2.h5')


# In[26]:


file1=open("result.txt","w")


# In[27]:


file1.write(str(accuracy))


# In[28]:


file1.close()


# In[29]:


#from keras.models import load_model


# In[30]:


#m = load_model('braintumour_new_model.h5')


# In[31]:


#from keras.preprocessing import image


# In[32]:


#test_image = image.load_img('C:/Users/LENOVO/Desktop/Brain Tumour Detection/brain_tumor_dataset/Y170.jpg', 
               #target_size=(224,224))


# In[33]:


#type(test_image)


# In[34]:


#test_image


# In[35]:


#test_image = image.img_to_array(test_image)


# In[36]:


#type(test_image)


# In[37]:


#test_image.shape


# In[38]:


#import numpy as np 


# In[39]:


#test_image = np.expand_dims(test_image, axis=0)


# In[40]:


#test_image.shape


# In[41]:


#result = m.predict(test_image)


# In[42]:


#result


# In[43]:


#if result[0][0] == 1.0:
   # print('Yes')
#else:
   # print('No')


# In[ ]:




