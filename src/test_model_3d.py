import tensorflow as tf
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model

def test_model_3d(input_shape=(1024, 1024, 1024, 1)):
    """3D model with progressive downsampling and memory optimization."""
    inputs = Input(shape=input_shape)
    
    # Progressive downsampling to reduce memory usage
    x = Conv3D(16, (3, 3, 3), activation='relu', padding='same', use_bias=False)(inputs)
    x = MaxPooling3D((2, 2, 2))(x)
    
    x = Conv3D(32, (3, 3, 3), activation='relu', padding='same', use_bias=False)(x)
    x = MaxPooling3D((2, 2, 2))(x)
    
    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same', use_bias=False)(x)
    x = MaxPooling3D((2, 2, 2))(x)
    
    # Global average pooling instead of flatten to reduce parameters
    x = tf.keras.layers.GlobalAveragePooling3D()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model
