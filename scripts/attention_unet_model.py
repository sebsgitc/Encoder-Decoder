import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Activation, Add, Multiply
from tensorflow.keras.models import Model

# Attention Gate
def attention_block(skip_input, gating_input, filters):
    """
    Attention Gate:
    - skip_input: Features from encoder (skip connection)
    - gating_input: Features from decoder (upsampling layer)
    - filters: Number of output channels
    """
    theta_x = Conv2D(filters, (1,1), strides=(1,1), padding='same')(skip_input) 
    phi_g = Conv2D(filters, (1,1), strides=(1,1), padding='same')(gating_input)
    attention = Add()([theta_x, phi_g]) 
    attention = Activation('relu')(attention)
    attention = Conv2D(1, (1,1), strides=(1,1), padding='same', activation='sigmoid')(attention)
    return Multiply()([skip_input, attention])


# Attention U-Net 
def attention_unet_model(input_size=(256, 256, 1)):
    inputs = Input(input_size)

    # Encoder
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Bottleneck
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)

    # Decoder w. Attention Gates
    up5 = UpSampling2D(size=(2, 2))(conv4)
    att5 = attention_block(conv3, up5, 128)
    merge5 = concatenate([att5, up5], axis=-1)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(merge5)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)

    up6 = UpSampling2D(size=(2, 2))(conv5)
    att6 = attention_block(conv2, up6, 64)
    merge6 = concatenate([att6, up6], axis=-1)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(merge6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)

    up7 = UpSampling2D(size=(2, 2))(conv6)
    att7 = attention_block(conv1, up7, 32) 
    merge7 = concatenate([att7, up7], axis=-1)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(merge7)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)

    # Output Layer
    output = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(conv7)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model