from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(train_directory, val_directory, test_directory, target_size, batch_size, seed):
    train_datagen = ImageDataGenerator(rescale=1.0 / 255)
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_directory,
        target_size=target_size,
        batch_size=batch_size,
        seed=seed,
        class_mode='categorical',
        shuffle=True,
    )

    val_generator = val_datagen.flow_from_directory(
        val_directory,
        target_size=target_size,
        batch_size=batch_size,
        seed=seed,
        class_mode='categorical',
        shuffle=False,
    )

    test_generator = test_datagen.flow_from_directory(
        test_directory,
        target_size=target_size,
        batch_size=batch_size,
        seed=seed,
        class_mode='categorical',
        shuffle=False,
    )

    return train_generator, val_generator, test_generator
