# TODO: build_tf_policy` and `build_torch_policy'


from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models import ModelCatalog
from ray.rllib.models.preprocessors import Preprocessor
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.recurrent_tf_modelv2 import RecurrentTFModelV2
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.annotations import override

import tensorflow as tf
from tensorflow.keras.layers import  Activation, Add, Attention, Concatenate, Conv2D, Dense, \
        Flatten, Input, Lambda, LayerNormalization, LSTM, Multiply, Reshape, TimeDistributed
# import keras
import numpy as np

# import transformer


def convNetwork(input_layer):
    reshape = TimeDistributed(Reshape((64, 64, 3)))(input_layer)
    conv1 = TimeDistributed(Conv2D(
        filters=64, kernel_size=(8, 8), strides=(4, 4), name="conv1",
        padding='same', data_format=None, dilation_rate=(1, 1), activation=tf.nn.relu))(reshape)
    # maxpool1 = TimeDistributed(MaxPooling2D(
        # pool_size=(2, 2), name="maxpool1", strides=None, padding='valid', data_format=None))(conv1)
    conv2 = TimeDistributed(Conv2D(
        filters=64, kernel_size=(4, 4), strides=(2, 2), name="conv2",
        padding='same', data_format=None, dilation_rate=(1, 1), activation=tf.nn.relu))(conv1)

    flatten = TimeDistributed(Flatten(name="flatten"))(conv2) # data_format="channels_last" (default)
    return flatten


def gate(x, y):
    """
    gating mechanism from GRU
    """
    x_shape = x.shape[-1]

    wr = Dense(x_shape)(y)
    ur = Dense(x_shape)(x)
    print("wr", wr.shape)
    print("ur", ur.shape)
    r = Activation('sigmoid')(Add()([wr, ur]))
    wz = Dense(x_shape)(y)
    uz = Dense(x_shape)(x)
    print("r", r.shape)
    print("wz", wz.shape)
    print("uz", uz.shape)
    z = Activation('sigmoid')(Add()([wz, uz]))  # TODO: add bias to activation
    wg = Dense(x_shape)(y)
    ug = Dense(x_shape)(Multiply()([r, x]))
    print("z", z.shape)
    print("wg", wg.shape)
    print("ug", ug.shape)
    h = Activation('tanh')(Add()([wg, ug]))
    print("h", h.shape)
    g = Add()(
            [Multiply()([Lambda(lambda var: 1. - var)(z), x]),
            Multiply()([z, h])])
    print("g", g.shape)

    return g


def transformer(input_layer, d_model, n_heads):
    # causal = True: adds mask to prevent flow from future to past
    def head(inp):
        q = Dense(d_model)(inp)
        k = Dense(d_model)(inp)
        v = Dense(d_model)(inp)
        attention = Attention(use_scale=True, causal=True)([q, v, k])
        return attention

    norm1 = LayerNormalization(axis=-1, scale=False, trainable=True)(input_layer)
    print("norm1", norm1.shape)

    heads = []
    for i in range(n_heads):
        heads.append(head(norm1))

    conc = Concatenate(axis=-1)(heads)
    print("conc", conc.shape)
    gate1 = gate(input_layer, conc)
    print("gate1", gate1.shape)
    norm2 = LayerNormalization(axis=-1, scale=False, trainable=True)(gate1)
    print("norm2", norm2.shape)
    pmlp = Dense(input_layer.shape[-1])(norm2)
    print("pmlp", pmlp.shape)
    gate2 = gate(gate1, pmlp)
    print("gate2", gate2.shape)
    # TODO: setting bias greater than 0 can greatly improve learning speed (according to paper)

    return gate2


class TransformerCustomModel(RecurrentTFModelV2):
    """
    input shape is (batch, time_steps, input_size), which is same as LSTM
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, cell_size=256, d_model=8, n_heads=8):
        super(TransformerCustomModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        self.cell_size = cell_size

        input_layer = Input(
            shape=(None, 64 * 64 * 3), name="inputs")
        # seq_in = Input(shape=(), name="seq_in", dtype=tf.int32)

        # d_model & n_heads == 0
        flatten = convNetwork(input_layer)
        shorten = Dense(256, activation=tf.nn.relu, name="shorten")(flatten)

        # TODO: add positional encoding
        trans1 = transformer(shorten, d_model, n_heads)
        trans2 = transformer(trans1, d_model, n_heads)

        logits = Dense(
            15, activation=tf.keras.activations.linear, name="logits")(trans2)
        values = Dense(
            1, activation=None, name="values")(trans2)
        print("logits", logits.shape)
        print("values", values.shape)

        # Create the RNN model
        self.rnn_model = tf.keras.Model(
            inputs=[input_layer],
            outputs=[logits, values])
        self.register_variables(self.rnn_model.variables)
        self.rnn_model.summary()

    @override(RecurrentTFModelV2)
    def forward_rnn(self, inputs, state, seq_lens):
        model_out, self._value_out = self.rnn_model([inputs, seq_lens])
        # return output and new states
        return model_out, state

    @override(ModelV2)
    def get_initial_state(self):
        # initial hidden state in model
        return [
            np.zeros(self.cell_size, np.float32),
            np.zeros(self.cell_size, np.float32),
        ]

    @override(ModelV2)
    def value_function(self):
        return tf.reshape(self._value_out, [-1])


class LSTMCustomModel(RecurrentTFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, cell_size=256):
        super(LSTMCustomModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        self.cell_size = cell_size

        input_layer = Input(
            shape=(None, 64 * 64 * 3), name="inputs")
        state_in_h = Input(shape=(cell_size, ), name="h")
        state_in_c = Input(shape=(cell_size, ), name="c")
        seq_in = Input(shape=(), name="seq_in", dtype=tf.int32)

        flatten = convNetwork(input_layer)
        # reshape = TimeDistributed(Reshape((64, 64, 3)))(input_layer)
        # conv1 = TimeDistributed(Conv2D(
        #     filters=16, kernel_size=(4, 4), strides=(1, 1), name="conv1",
        #     padding='same', data_format=None, dilation_rate=(1, 1), activation=tf.nn.relu))(reshape)
        # maxpool1 = TimeDistributed(MaxPooling2D(
        #     pool_size=(2, 2), name="maxpool1", strides=None, padding='valid', data_format=None))(conv1)
        #
        # flatten = TimeDistributed(Flatten(name="flatten"))(maxpool1) # data_format="channels_last" (default)

        # Preprocess observation with a hidden layer and send to LSTM cell
        dense1 = TimeDistributed(Dense(
             256, activation=tf.nn.relu, name="dense1"))(flatten)

        lstm_out, state_h, state_c = LSTM(
            cell_size, return_sequences=True, return_state=True, name="lstm")(
                inputs=dense1,
                mask=tf.sequence_mask(seq_in),
                initial_state=[state_in_h, state_in_c])

        # Postprocess LSTM output with another hidden layer and compute values
        logits = Dense(
            15, activation=tf.keras.activations.linear, name="logits")(lstm_out)
        values = Dense(
            1, activation=None, name="values")(lstm_out)

        # Create the RNN model
        self.rnn_model = tf.keras.Model(
            inputs=[input_layer, seq_in, state_in_h, state_in_c],
            outputs=[logits, values, state_h, state_c])
        self.register_variables(self.rnn_model.variables)
        self.rnn_model.summary()

    @override(RecurrentTFModelV2)
    def forward_rnn(self, inputs, state, seq_lens):
        # Call the model with the given input tensors and state.
        # model_out, self._value_out, h, c = self.rnn_model(input_dict["obs"])
        model_out, self._value_out, h, c = self.rnn_model([inputs, seq_lens] + state)
        return model_out, [h, c]
        # model_out, self._value_out = self.rnn_model([inputs, seq_lens] + state)
        # return model_out

    @override(ModelV2)
    def get_initial_state(self):
        # Get the initial recurrent state values for the model.
        return [
            np.zeros(self.cell_size, np.float32),
            np.zeros(self.cell_size, np.float32),
        ]

    @override(ModelV2)
    def value_function(self):
        return tf.reshape(self._value_out, [-1])


class ProcgenPreprocessor(Preprocessor):
    def _init_shape(self, obs_space, options):
        return obs_space.shape

    def transform(self, observation):
        # observation is shape [64, 64, 3]
        return observation


# if __name__ == "__main__":
#     import gym
#     env = gym.make('GuessingGame-v0')
#     print(env.observation_space)
#     model = LSTMCustomModel(np.array([2, ]), env.action_space, 1, None, 'test')
