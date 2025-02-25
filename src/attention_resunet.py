import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Conv2DTranspose, concatenate, Add, Multiply
from tensorflow.keras.models import Model
from load_data import IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES

# Maybe try these two lines below to save memory
#from tensorflow.keras import mixed_precision
#mixed_precision.set_global_policy("mixed_float16")

def res_block(x, filters):
    """Residual block with two convolutional layers and a shortcut connection."""
    shortcut = x  # Store input as shortcut

    # If the input channel does not match filters, apply a 1x1 convolution to match shape
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

def attention_block(x, g, filters):
    """Attention Gate to filter out irrelevant features from skip connections."""
    theta_x = Conv2D(filters, (1, 1), padding='same')(x)
    phi_g = Conv2D(filters, (1, 1), padding='same')(g)
    add_xg = Add()([theta_x, phi_g])
    act_xg = Activation('relu')(add_xg)
    psi = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(act_xg)
    return Multiply()([x, psi])

def attention_resunet(input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)):
    """Attention ResUNet model for border segmentation."""
    inputs = Input(shape=input_shape)

    # Encoder
    c1 = res_block(inputs, 32)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = res_block(p1, 64)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = res_block(p2, 128)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = res_block(p3, 256)

    # Decoder with Attention Gates
    u5 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c4)
    att5 = attention_block(c3, u5, 128)
    u5 = concatenate([u5, att5])
    u5 = res_block(u5, 128)

    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(u5)
    att6 = attention_block(c2, u6, 64)
    u6 = concatenate([u6, att6])
    u6 = res_block(u6, 64)

    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(u6)
    att7 = attention_block(c1, u7, 32)
    u7 = concatenate([u7, att7])
    u7 = res_block(u7, 32)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(u7)
    # Maybe work inthe row below to save memory
    #outputs = tf.keras.layers.Activation("softmax", dtype="float32")(outputs)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

if __name__ == "__main__":
    model = attention_resunet()
    model.summary()