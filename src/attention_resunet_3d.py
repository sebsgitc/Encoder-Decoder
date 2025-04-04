import tensorflow as tf
from tensorflow.keras.layers import Input, Conv3D, BatchNormalization, Activation, MaxPooling3D, Conv3DTranspose, concatenate, Add, Multiply
from tensorflow.keras.models import Model

def res_block_3d(x, filters):
    """Residual block with two convolutional layers and a shortcut connection."""
    shortcut = x  # Store input as shortcut

    # If the input channel does not match filters, apply a 1x1 convolution to match shape
    if x.shape[-1] != filters:
        shortcut = Conv3D(filters, (1, 1, 1), padding="same")(shortcut)

    x = Conv3D(filters, (3, 3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(filters, (3, 3, 3), padding='same')(x)
    x = BatchNormalization()(x)

    # Add shortcut connection
    x = Add()([shortcut, x])
    x = Activation('relu')(x)
    return x

def attention_block_3d(x, g, filters):
    """Attention Gate to filter out irrelevant features from skip connections."""
    theta_x = Conv3D(filters, (1, 1, 1), padding='same')(x)
    phi_g = Conv3D(filters, (1, 1, 1), padding='same')(g)
    
    # Ensure the shapes are compatible for addition
    if theta_x.shape[1] != phi_g.shape[1]:
        phi_g = tf.image.resize(phi_g, size=(theta_x.shape[1], theta_x.shape[2], theta_x.shape[3]))
    
    add_xg = Add()([theta_x, phi_g])
    act_xg = Activation('relu')(add_xg)
    psi = Conv3D(1, (1, 1, 1), padding='same', activation='sigmoid')(act_xg)
    return Multiply()([x, psi])

def attention_resunet_3d(input_shape=(1024, 1024, 1024, 1)):
    """Attention ResUNet model for 3D data."""
    inputs = Input(shape=input_shape)

    # Encoder
    c1 = res_block_3d(inputs, 32)
    p1 = MaxPooling3D((2, 2, 2))(c1)

    c2 = res_block_3d(p1, 64)
    p2 = MaxPooling3D((2, 2, 2))(c2)

    c3 = res_block_3d(p2, 128)
    p3 = MaxPooling3D((2, 2, 2))(c3)

    c4 = res_block_3d(p3, 256)

    # Decoder with Attention Gates
    u5 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(c4)
    att5 = attention_block_3d(c3, u5, 128)
    u5 = concatenate([u5, att5])
    u5 = res_block_3d(u5, 128)

    u6 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(u5)
    att6 = attention_block_3d(c2, u6, 64)
    u6 = concatenate([u6, att6])
    u6 = res_block_3d(u6, 64)

    u7 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(u6)
    att7 = attention_block_3d(c1, u7, 32)
    u7 = concatenate([u7, att7])
    u7 = res_block_3d(u7, 32)

    outputs = Conv3D(1, (1, 1, 1), activation='sigmoid')(u7)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

if __name__ == "__main__":
    model = attention_resunet_3d()
    model.summary()