# Convolutional Neural Network

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
# We will use 32 filters with size 3 x 3
classifier.add(Conv2D(32, (3,3), input_shape=(64,64,3), activation='relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3,3), input_shape=(64,64,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))


# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

# Compiling the CNN
# Adam is the algorithm used to adjust the weights
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

# Generate more images in different configurations (scale, zoom, flip, rotation) from training data
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

# Generate more image configurations for test sets
test_datagen = ImageDataGenerator(rescale=1./255)

# Generate training set from local image folders. The number of classes is detected automatically according to folder number
training_set = train_datagen.flow_from_directory('dataset/training_set',target_size=(64, 64),batch_size=32, class_mode='binary')

# Generate test set from local images
test_set = test_datagen.flow_from_directory('dataset/test_set',target_size=(64, 64),batch_size=32,class_mode='binary')

# Fit the model
classifier.fit_generator(training_set,steps_per_epoch=2000,epochs=50,validation_data=test_set,validation_steps=800)


# Part 3 - Making new predictions



