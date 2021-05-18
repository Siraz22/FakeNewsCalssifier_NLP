# -*- coding:utf-8 -*-
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from keras.models import Model
from pickle import dump


from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, Concatenate, LeakyReLU
from tensorflow.keras.optimizers import Adam


def DeepFake_Detector_Model(image_path, model_path):

    # load an image from file
    image = load_img('dog.jpg', target_size=(224, 224))

    # convert the image pixels to a numpy array
    image = img_to_array(image)

    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

    # prepare the image for the VGG model
    image = preprocess_input(image)

    # load model
    model = VGG16()

    # remove the output layer
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

    # get extracted features
    features = model.predict(image)

    # load the trained random forest model
    rf_model = pickle.load(open(filename, 'rb'))

    # Predict the nature of the image (real = 1 or fake = 0)
    prediction = rf_model.predict(features)

    if prediction < 0.5:
        return 0
    else:
        return 1

