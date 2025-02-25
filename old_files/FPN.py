import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, Activation
from tensorflow.keras.models import Model

def conv_block(x, filters):
    """Standard Convolutional Block: Conv -> BN -> ReLU"""
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def fpn(input_shape=(256, 256, 3)):
    """Feature Pyramid Network (FPN) for image segmentation"""
    inputs = Input(shape=input_shape)

    # Encoder (Backbone)
    c1 = conv_block(inputs, 32)
    p1 = MaxPooling2D((2, 2))(c1)  # (128x128)

    c2 = conv_block(p1, 64)
    p2 = MaxPooling2D((2, 2))(c2)  # (64x64)

    c3 = conv_block(p2, 128)
    p3 = MaxPooling2D((2, 2))(c3)  # (32x32)

    c4 = conv_block(p3, 256)  # Bottleneck

    # FPN Top-down Pathway (Upsampling + Feature Fusion)
    u5 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = concatenate([u5, c3])  # Merge with encoder features
    u5 = conv_block(u5, 128)

    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(u5)
    u6 = concatenate([u6, c2])
    u6 = conv_block(u6, 64)

    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(u6)
    u7 = concatenate([u7, c1])
    u7 = conv_block(u7, 32)

    # Final segmentation output
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(u7)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model