from config.dynamic_3dcnnbase_config import *
from dynamic_inference.3dcnnbase.model import build_3d_cnn
from dynamic_inference.3dcnnbase.data_loader import memory_data_generator, load_preprocessed_frames
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def main():
    # Load preprocessed frames
    train_videos, train_labels = load_preprocessed_frames(train_dir, classes)
    val_videos, val_labels = load_preprocessed_frames(val_dir, classes)

    # Build data generators
    train_generator = memory_data_generator(train_videos, train_labels, BATCH_SIZE, len(classes))
    val_generator = memory_data_generator(val_videos, val_labels, BATCH_SIZE, len(classes))

    # Build the model
    input_shape = (FRAMES_PER_VIDEO, *FRAME_SIZE, 3)
    model = build_3d_cnn(input_shape, len(classes))

    # Define callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1),
        ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=18, min_lr=1e-6, verbose=1),
    ]

    # Train the model
    train_steps = len(train_videos) // BATCH_SIZE
    val_steps = len(val_videos) // BATCH_SIZE
    model.fit(
        train_generator,
        steps_per_epoch=train_steps,
        validation_data=val_generator,
        validation_steps=val_steps,
        epochs=EPOCHS,
        callbacks=callbacks
    )

if __name__ == "__main__":
    main()
