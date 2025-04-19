import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from dynamic_inference.2d3dhybridcnn.data_loader import memory_data_generator, load_and_split_frames
from dynamic_inference.2d3dhybridcnn.model import build_hybrid_2d3d_cnn
from config.dynamic_2d3dhybridcnn_config import *

def train_hybrid_model():
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()

    # Load and split data
    train_videos, train_labels, val_videos, val_labels, _, _ = load_and_split_frames(data_dir, classes)
    train_generator = memory_data_generator(train_videos, train_labels, BATCH_SIZE, len(classes), seed=SEED)
    val_generator = memory_data_generator(val_videos, val_labels, BATCH_SIZE, len(classes), seed=SEED)

    # Build the model
    model = build_hybrid_2d3d_cnn(FRAME_SIZE, FRAMES_PER_VIDEO, len(classes))

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6, verbose=1)

    # Train the model
    model.fit(
        train_generator,
        steps_per_epoch=len(train_videos) // BATCH_SIZE,
        validation_data=val_generator,
        validation_steps=len(val_videos) // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[early_stopping, model_checkpoint, reduce_lr]
    )
