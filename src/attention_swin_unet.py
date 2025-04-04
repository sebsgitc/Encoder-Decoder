import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, concatenate, Activation, Add, Multiply, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import SwinTransformer
from load_data import IMG_HEIGHT, IMG_WIDTH

def attention_block(x, g, filters):
    """Attention Gate to filter out irrelevant features from skip connections."""
    theta_x = Conv2D(filters, (1, 1), padding='same')(x)
    phi_g = Conv2D(filters, (1, 1), padding='same')(g)
    add_xg = Add()([theta_x, phi_g])
    act_xg = Activation('relu')(add_xg)
    psi = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(act_xg)
    return Multiply()([x, psi])

def attention_swin_unet(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)):
    """Attention U-Net model with Swin Transformer encoder."""
    inputs = Input(shape=input_shape)

    # Swin Transformer Encoder
    swin_encoder = SwinTransformer(include_top=False, input_shape=input_shape, pooling=None)
    encoder_outputs = swin_encoder(inputs)

    # Decoder with Attention Gates
    u5 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(encoder_outputs)
    att5 = attention_block(encoder_outputs, u5, 128)
    u5 = concatenate([u5, att5])
    u5 = Conv2D(128, (3, 3), padding='same', activation='relu')(u5)

    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(u5)
    att6 = attention_block(encoder_outputs, u6, 64)
    u6 = concatenate([u6, att6])
    u6 = Conv2D(64, (3, 3), padding='same', activation='relu')(u6)

    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(u6)
    att7 = attention_block(encoder_outputs, u7, 32)
    u7 = concatenate([u7, att7])
    u7 = Conv2D(32, (3, 3), padding='same', activation='relu')(u7)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(u7)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

if __name__ == "__main__":
    model = attention_swin_unet()
    model.summary()