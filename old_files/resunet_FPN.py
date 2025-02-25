import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation, 
    MaxPooling2D, Conv2DTranspose, UpSampling2D, 
    concatenate, Add, Multiply
)
from tensorflow.keras.models import Model
from load_data import IMG_HEIGHT, IMG_WIDTH

# Residual Block
def res_block(x, filters):
    """Residual block with two convolutional layers and a shortcut connection."""
    shortcut = x  # Store input as shortcut

    # If input channel does not match filters, apply a 1x1 convolution to match shape
    if x.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), padding="same")(shortcut)

    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)

    # Add shortcut connection
    x = Add()([shortcut, x])
    x = Activation('relu')(x)
    return x

# Attention Block
def attention_block(x, g, filters):
    """Attention Gate to filter out irrelevant features from skip connections."""
    theta_x = Conv2D(filters, (1, 1), padding='same')(x)
    phi_g = Conv2D(filters, (1, 1), padding='same')(g)
    add_xg = Add()([theta_x, phi_g])
    act_xg = Activation('relu')(add_xg)
    psi = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(act_xg)
    return Multiply()([x, psi])

# Feature Pyramid Network (FPN) Block
def fpn_block(x, skip_connection, filters):
    """Feature Pyramid block to refine segmentation features from multiple scales."""
    x = UpSampling2D((2, 2))(x)  # Upsample feature map
    x = concatenate([x, skip_connection])  # Merge with encoder features
    x = Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
    return x

# Attention ResUNet with FPN
def attention_resunet_fpn(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)):
    """Attention ResUNet with Feature Pyramid Network (FPN) decoder for multi-scale segmentation."""
    inputs = Input(shape=input_shape)

    # Encoder (Extract Features)
    c1 = res_block(inputs, 32)  
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = res_block(p1, 64)  
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = res_block(p2, 128)  
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = res_block(p3, 256)  

    # FPN Aggregation for Multi-Scale Learning
    fpn3 = fpn_block(c4, c3, 128)
    fpn2 = fpn_block(fpn3, c2, 64)
    fpn1 = fpn_block(fpn2, c1, 32)

    # Decoder with Attention & FPN
    u5 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c4)
    att5 = attention_block(fpn3, u5, 128)
    u5 = concatenate([u5, att5])
    u5 = res_block(u5, 128)

    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(u5)
    att6 = attention_block(fpn2, u6, 64)
    u6 = concatenate([u6, att6])
    u6 = res_block(u6, 64)

    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(u6)
    att7 = attention_block(fpn1, u7, 32)
    u7 = concatenate([u7, att7])
    u7 = res_block(u7, 32)

    # Output Layer (Segmentation Mask)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(u7)  # Binary segmentation output

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# Run model summary if script is executed
if __name__ == "__main__":
    model = attention_resunet_fpn()
    model.summary()