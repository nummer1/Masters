# https://machinetalk.org/2019/04/29/create-the-transformer-with-tensorflow-2-0/

import tensorflow as tf
import numpy as np


def positional_embedding(pos, model_size):
    # returns positional embedding of pos
    PE = np.zeros((1, model_size))
    for i in range(model_size):
        if i % 2 == 0:
            PE[:, i] = np.sin(pos / 10000 ** (i / model_size))
        else:
            PE[:, i] = np.cos(pos / 10000 ** ((i - 1) / model_size))
    return PE


class MultiHeadAttention(tf.keras.Model):
    def __init__(self, model_size, h):
        super(MultiHeadAttention, self).__init__(name="MHA")
        self.query_size = model_size // h
        self.key_size = model_size // h
        self.value_size = model_size // h
        self.h = h
        self.wq = [tf.keras.layers.Dense(self.query_size) for _ in range(h)]
        self.wk = [tf.keras.layers.Dense(self.key_size) for _ in range(h)]
        self.wv = [tf.keras.layers.Dense(self.value_size) for _ in range(h)]
        self.wo = tf.keras.layers.Dense(model_size)

    def call(self, query, value):
        # query has shape (batch, query_len, model_size)
        # value has shape (batch, value_len, model_size)
        heads = []
        for i in range(self.h):
            score = tf.matmul(self.wq[i](query), self.wk[i](value), transpose_b=True)

            # Here we scale the score as described in the paper
            score /= tf.math.sqrt(tf.dtypes.cast(self.key_size, tf.float32))
            # score has shape (batch, query_len, value_len)

            alignment = tf.nn.softmax(score, axis=2)
            # alignment has shape (batch, query_len, value_len)

            head = tf.matmul(alignment, self.wv[i](value))
            # head has shape (batch, decoder_len, value_size)
            heads.append(head)

        # Concatenate all the attention heads
        # so that the last dimension summed up to model_size
        heads = tf.concat(heads, axis=2)
        heads = self.wo(heads)
        # heads has shape (batch, query_len, model_size)
        return heads


class Transformer(tf.keras.Model):
    # TODO: switch Batch Normalization for Layer Normalization
    def __init__(self, vocab_size, model_size, num_layers, h, pes):
        super(Transformer, self).__init__(name="transformer")

        self.model_size = model_size
        self.num_layers = num_layers
        self.h = h
        self.pes = pes

        # One Embedding layer
        self.embedding = tf.keras.layers.Embedding(vocab_size, model_size)

        # num_layers Multi-Head Attention and Normalization layers
        self.attention = [MultiHeadAttention(model_size, h) for _ in range(num_layers)]
        self.attention_norm = [tf.keras.layers.BatchNormalization() for _ in range(num_layers)]

        # num_layers FFN and Normalization layers
        self.dense_1 = [tf.keras.layers.Dense(model_size * 4, activation='relu') for _ in range(num_layers)]
        self.dense_2 = [tf.keras.layers.Dense(model_size) for _ in range(num_layers)]
        self.ffn_norm = [tf.keras.layers.BatchNormalization() for _ in range(num_layers)]

    def call(self, sequence):
        embed_out = self.embedding(sequence)
        embed_out += self.pes[:sequence.shape[1], :]

        sub_in = embed_out

        for i in range(self.num_layers):
            sub_out = self.attention[i](sub_in, sub_in)
            sub_out = sub_in + sub_out
            sub_out = self.attention_norm[i](sub_out)

            ffn_in = sub_out

            ffn_out = self.dense_2[i](self.dense_1[i](ffn_in))
            ffn_out = ffn_in + ffn_out
            ffn_out = self.ffn_norm[i](ffn_out)

            sub_in = ffn_out

        return ffn_out
