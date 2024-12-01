from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def train_model(model, train_generator, val_generator, checkpoint_path, epochs=50):
    checkpoint_callback = ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1,
    )

    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=12,
        restore_best_weights=True,
        verbose=1,
    )

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=[checkpoint_callback, early_stopping],
        verbose=1,
    )

    return history
