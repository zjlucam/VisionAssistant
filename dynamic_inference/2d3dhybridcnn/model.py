from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D, TimeDistributed,
    Conv3D, MaxPooling3D, GlobalAveragePooling3D, Dense, Dropout
)

def build_hybrid_2d3d_cnn(frame_shape, num_frames, num_classes):
    # 2D CNN for spatial features
    input_2d = Input(shape=frame_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_2d)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)
    feature_extractor = Model(inputs=input_2d, outputs=x)

    # Video input
    video_input = Input(shape=(num_frames, *frame_shape))
    time_distributed = TimeDistributed(feature_extractor)(video_input)
    time_distributed = TimeDistributed(GlobalAveragePooling2D())(time_distributed)

    # 3D CNN for temporal features
    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(time_distributed)
    x = MaxPooling3D((2, 2, 2))(x)
    x = Dropout(0.15)(x)
    x = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(x)
    x = MaxPooling3D((2, 2, 2))(x)
    x = Dropout(0.2)(x)
    x = GlobalAveragePooling3D()(x)

    output = Dense(256, activation='relu')(x)
    output = Dropout(0.35)(output)
    output = Dense(num_classes, activation='softmax')(output)

    model = Model(inputs=video_input, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
