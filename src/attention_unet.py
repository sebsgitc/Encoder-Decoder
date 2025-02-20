import tensorflow as tf
from tensorflow.keras import layers, Model

def attention_block(x, g, inter_channel):
    """Attention Gate for U-Net"""
    theta_x = layers.Conv2D(inter_channel, (1, 1), strides=(1, 1), padding='same')(x)
    phi_g = layers.Conv2D(inter_channel, (1, 1), strides=(1, 1), padding='same')(g)
    add_xg = layers.Add()([theta_x, phi_g])
    act_xg = layers.Activation('relu')(add_xg)
    psi = layers.Conv2D(1, (1, 1), strides=(1, 1), padding='same', activation='sigmoid')(act_xg)
    return layers.Multiply()([x, psi])

def attention_unet(input_shape=(256, 256, 1)):
    """Attention U-Net for segmentation."""
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)  

    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    # Bottleneck
    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c4)

    # Decoder with Attention Gates
    u5 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c4)
    att5 = attention_block(c3, u5, 128)
    u5 = layers.concatenate([u5, att5])
    c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u5)

    u6 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c5)
    att6 = attention_block(c2, u6, 64)
    u6 = layers.concatenate([u6, att6])
    c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u6)

    u7 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c6)
    att7 = attention_block(c1, u7, 32)
    u7 = layers.concatenate([u7, att7])
    c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u7)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c7)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

if __name__ == "__main__":
    model = attention_unet()
    model.summary()