# TensorFlow and tf.keras
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras import layers
from keras.preprocessing.image import ImageDataGenerator
import streamlit as st
import matplotlib.pyplot as plt
import os

################################### Title ###############################################################
st.title("Image Classification") #Show the title of the app on Streamlit
st.subheader(':blue[_Created by Wouter Selis_] :male-technologist:', divider='rainbow') #Show a subheader on Streamlit with text in blue, an emoji and a rainbow line under the text


categories=["apples", "bananas", "carrots", "oranges", "tomatoes"]

st.slider(
    "My Slider", 0.0, 100.0, 1.0, step=1.0, key="myslider"
)

################################### Split data into training, test and validation ###############################
train_val_datagen = ImageDataGenerator(validation_split=0.2,
                                   rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_val_datagen.flow_from_directory('datasets/training_set',
                                                 subset='training',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical') 

validation_set = train_val_datagen.flow_from_directory('datasets/training_set',
                                                 subset='validation',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('datasets/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical')


############################################# show total per category ########################
root_directory = ["datasets/training_set/","datasets/test_set/" ]

def count_pictures(root_directory, category):
    counter = 0
    for foldername, subfolders, filenames in os.walk(root_directory):
        for filename in filenames:
            if category in filename:
                counter += 1
    return counter

total_count = 0
for category in categories:
    # For every category count the total images in the training_set and test_set folder and print them
    count = count_pictures(root_directory[0], category)
    count += count_pictures(root_directory[1], category)
    st.write(f"Total images of {category}: {count}")
    total_count += count
    count = 0

st.write(f"Total images: {total_count}")

################################### Model ###############################
NUM_CLASSES = 5

model = tf.keras.Sequential([
  layers.RandomFlip("horizontal"),
  layers.RandomTranslation(0.2,0.2),
  layers.RandomZoom(0.2),

  layers.Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation="relu"),
  layers.MaxPooling2D((2, 2)),
  layers.Dropout(0.2),
  layers.Conv2D(32, (3, 3), activation="relu"),
  layers.MaxPooling2D((2, 2)),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation="relu"),
  layers.Dense(NUM_CLASSES, activation="softmax")
])

################################### Compile model ###############################
model.compile(optimizer = optimizers.Adam(learning_rate=0.001), 
              loss = 'categorical_crossentropy', 
              metrics = ['accuracy'])


################################### Model training ###############################
epoch=20
if st.session_state.myslider != 1:
    epoch=int(st.session_state.myslider)

history2 = model.fit(training_set, validation_data=validation_set, epochs=epoch)


################################### Visualize loss and accurracy ###############################
# Create a figure and a grid of subplots with a single call
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

# Plot the loss curves on the first subplot
ax1.plot(history2.history['loss'], label='training loss')
ax1.plot(history2.history['val_loss'], label='validation loss')
ax1.set_title('Loss curves')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()

# Plot the accuracy curves on the second subplot
ax2.plot(history2.history['accuracy'], label='training accuracy')
ax2.plot(history2.history['val_accuracy'], label='validation accuracy')
ax2.set_title('Accuracy curves')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()

# Adjust the spacing between subplots
fig.tight_layout()

# Show the figure
st.pyplot(fig)