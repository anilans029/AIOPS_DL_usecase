import logging as lg
import keras.api._v2.keras as keras






def train_and_valid_generator(data_dir, image_size,do_data_augmentation,batchSize):
    data_generator_kwargs = dict(
        rescale = 1./255,
        validation_split = 0.20
    )

    dataflow_kwargs = dict(
            target_size = image_size,
            batch_size = batchSize,
            interpolation = "bilinear"
    )

    valid_datagenerator = keras.preprocessing.image.ImageDataGenerator(
        **data_generator_kwargs
    )

    valid_generator = valid_datagenerator.flow_from_directory(data_dir,
                                        subset='validation',
                                        shuffle = False,
                                        **dataflow_kwargs)


    if do_data_augmentation:
        train_datagenerator = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=40,
            horizontal_flip=True,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            **data_generator_kwargs
        )
        lg.info("data augmentation is used for training")
    else:
        train_datagenerator = valid_datagenerator

    train_generator = train_datagenerator.flow_from_directory(
            directory=data_dir,
            subset = "training",
            shuffle = True,
            **dataflow_kwargs
        )
    lg.info("train and valid generator are created")

    return train_generator, valid_generator