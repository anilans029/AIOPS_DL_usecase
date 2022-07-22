# from tensorflow.keras import as tf
import keras.api._v2.keras as keras
import os
import logging as lg

def get_VGG16_model(inputShape, model_path):
    model_vgg16  = keras.applications.vgg16.VGG16(input_shape=inputShape,
                                                    weights="imagenet",
                                                    include_top=False
                                                )
    model_vgg16.save(model_path)
    lg.info(f"VGG16 base model is saved at : {model_path}")
    return model_vgg16

def prepare_full_model( model,CLASSES ,freeze_all , freeze_till ,learning_rate):
    if freeze_all==True:
        for layer in model.layers:
            layer.trainable= False
    elif (freeze_till is not None ) and (freeze_till > 0):
        for layer in model.layers[:-freeze_till]:
            layer.trainable= False

    ### adding Dense layers
    flatten_in = keras.layers.Flatten()(model.output)
    prediction = keras.layers.Dense(
                    units = CLASSES,
                    activation="softmax",
    )(flatten_in)

    full_model = keras.models.Model(
        inputs = model.input,
        outputs = prediction
    )

    full_model.compile(optimizer=keras.optimizers.SGD(learning_rate=learning_rate),
                     loss="categorical_crossentropy",
                     metrics="accuracy")
    
    lg.info("custom model is compiled and ready to be trained")
    return full_model

def load_full_model(untrained_full_model_path):
    model = keras.models.load_model(untrained_full_model_path)
    lg.info(f"untrained model is loaded from{untrained_full_model_path}")
    return model