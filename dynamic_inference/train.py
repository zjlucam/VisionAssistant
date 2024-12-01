from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def train_model(model, train_generator, val_generator, train_steps, val_steps, checkpoint_path, weights_path, epochs):
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True),
        ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True),
        ModelCheckpoint(weights_path, monitor='val_accuracy', save_best_only=True, save_weights_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=18, min_lr=1e-6)
    ]
    model.fit(train_generator, validation_data=val_generator, steps_per_epoch=train_steps,
              validation_steps=val_steps, epochs=epochs, callbacks=callbacks)
