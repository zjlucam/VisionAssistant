from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, GlobalAveragePooling3D, Dense, Dropout

def build_3d_cnn(input_shape, num_classes):
    model = Sequential([
        Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same', input_shape=input_shape),
        MaxPooling3D(pool_size=(1, 2, 2)), Dropout(0.1),
        Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same'),
        MaxPooling3D(pool_size=(1, 2, 2)), Dropout(0.1),
        Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same'),
        MaxPooling3D(pool_size=(1, 2, 2)), Dropout(0.15),
        Conv3D(256, kernel_size=(3, 3, 3), activation='relu', padding='same'),
        GlobalAveragePooling3D(),
        Dense(256, activation='relu'), Dropout(0.35),
        Dense(num_classes, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
