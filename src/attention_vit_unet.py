import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, concatenate, Activation, Add, Multiply, Dense, LayerNormalization, Dropout, Reshape
from tensorflow.keras.models import Model
from load_data import IMG_HEIGHT, IMG_WIDTH

def attention_block(x, g, filters):
    """Attention Gate to filter out irrelevant features from skip connections."""
    theta_x = Conv2D(filters, (1, 1), padding='same')(x)
    phi_g = Conv2D(filters, (1, 1), padding='same')(g)
    add_xg = Add()([theta_x, phi_g])
    act_xg = Activation('relu')(add_xg)
    psi = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(act_xg)
    return Multiply()([x, psi])

class PatchEmbedding(tf.keras.layers.Layer):
    def __init__(self, patch_size, num_patches, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.projection = None

    def build(self, input_shape):
        self.projection = Dense(self.embed_dim)
        self.position_embedding = self.add_weight(name="pos_embed", shape=(1, self.num_patches, self.embed_dim))

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(images=images,
                                           sizes=[1, self.patch_size, self.patch_size, 1],
                                           strides=[1, self.patch_size, self.patch_size, 1],
                                           rates=[1, 1, 1, 1],
                                           padding='VALID')
        patches = tf.reshape(patches, [batch_size, self.num_patches, -1])
        embeddings = self.projection(patches) + self.position_embedding
        return embeddings

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        return (batch_size, self.num_patches, self.embed_dim)

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.projection_dim = embed_dim // num_heads
        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.combine_heads = Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        attention = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        output = self.combine_heads(concat_attention)
        return output

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def vit_encoder(input_shape, patch_size, num_layers, embed_dim, num_heads, ff_dim, num_patches, training=False):
    inputs = Input(shape=input_shape)
    patches = PatchEmbedding(patch_size, num_patches, embed_dim)(inputs)
    for _ in range(num_layers):
        patches = TransformerBlock(embed_dim, num_heads, ff_dim)(patches, training=training)
    return Model(inputs, patches)

def attention_vit_unet(input_shape=(IMG_HEIGHT, IMG_WIDTH, 1), patch_size=16, num_layers=8, embed_dim=64, num_heads=4, ff_dim=128):
    """Attention U-Net model with ViT encoder."""
    inputs = Input(shape=input_shape)
    num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)

    # ViT Encoder
    vit_encoder_model = vit_encoder(input_shape, patch_size, num_layers, embed_dim, num_heads, ff_dim, num_patches, training=True)
    encoder_outputs = vit_encoder_model(inputs)

    # Reshape encoder outputs to match the expected 4D shape
    height = input_shape[0] // patch_size
    width = input_shape[1] // patch_size
    encoder_outputs = Reshape((height, width, embed_dim))(encoder_outputs)

    # Decoder with Attention Gates
    u5 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(encoder_outputs)
    att5 = attention_block(u5, u5, 128)
    u5 = concatenate([u5, att5])
    u5 = Conv2D(128, (3, 3), padding='same', activation='relu')(u5)

    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(u5)
    att6 = attention_block(u6, u6, 64)
    u6 = concatenate([u6, att6])
    u6 = Conv2D(64, (3, 3), padding='same', activation='relu')(u6)

    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(u6)
    att7 = attention_block(u7, u7, 32)
    u7 = concatenate([u7, att7])
    u7 = Conv2D(32, (3, 3), padding='same', activation='relu')(u7)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(u7)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

if __name__ == "__main__":
    model = attention_vit_unet()
    model.summary()