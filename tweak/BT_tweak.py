#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


IMAGE_SIZE = [224,224]


# In[ ]:


train_path = 'C:/Users/LENOVO/Desktop/Brain Tumour Detection/brain_tumor_dataset/Train'
valid_path = 'C:/Users/LENOVO/Desktop/Brain Tumour Detection/brain_tumor_dataset/Test'


# In[ ]:


vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)


# In[ ]:


for layer in vgg.layers:
    layer.trainable = False


# In[ ]:


folders = glob('C:/Users/LENOVO/Desktop/Brain Tumour Detection/brain_tumor_dataset/Train/*')


# In[ ]:


from keras_preprocessing.image import ImageDataGenerator


# In[ ]:


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('C:/Users/LENOVO/Desktop/Brain Tumour Detection/brain_tumor_dataset/Train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('C:/Users/LENOVO/Desktop/Brain Tumour Detection/brain_tumor_dataset/Test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')
num_pixels = IMAGE_SIZE[1]*IMAGE_SIZE[1]
num_pixels


# In[ ]:


def base_model(neuron):
    x = Flatten()(vgg.output)
    x = Dense(units=neuron,input_dim=num_pixels,activation='relu')(x)
    x = Dense(1024,activation='relu')(x)
    top_model = Dense(len(folders), activation='softmax')(x)
    model = Model(inputs=vgg.input, outputs=top_model)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# In[ ]:





# In[ ]:


neuron = 5
model = base_model(neuron)
accuracy = 0.0


# In[ ]:


def build_model():
    r=model.fit_generator(training_set,
                         samples_per_epoch = 64,
                         nb_epoch = 5,
                         validation_data = test_set,
                         nb_val_samples = 32)
    test_accuracy=r.history['val_accuracy'][-1]
    accuracy=test_accuracy*100
    return accuracy


# In[ ]:


build_model()
count = 0
best_accuracy = accuracy
best_neuron = 0


# In[ ]:


def resetWeights():
    print("Resetting all the weights....")
    w = model.get_weights()
    w = [[j*0 for j in i] for i in w]
    model.set_weights(w)


# In[ ]:


while accuracy < 85 and count < 10:
    print("Updating model....")
    model = base_model(neuron*2)
    neuron = neuron*2
    count = count + 1
    accuracy = build_model()
    if best_accuracy < accuracy:
        best_accuracy = accuracy
        best_neuron = neuron
    print()
    resetWeights()


# In[ ]:


print("******************************")
print(best_neuron)
model = base_model(best_neuron)
build_model()
model.save("brain_detection_model_updated.h5")
print("Model Saved!")


# In[ ]:


file1=open("result.txt","w")
file1=write(str(best_accuracy))
file1.close()

