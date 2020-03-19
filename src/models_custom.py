# TODO: build_tf_policy` and `build_torch_policy'


from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models import ModelCatalog
from ray.rllib.models.preprocessors import Preprocessor
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.recurrent_tf_modelv2 import RecurrentTFModelV2
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.annotations import override

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import  Activation, Add, Attention, Concatenate, Conv2D, Dense, \
        Flatten, Input, Lambda, Layer, LayerNormalization, LSTM, Multiply, Reshape, TimeDistributed
from tensorflow.keras import backend as K

# import transformer


class AdvancedAdd(Layer):
    def __init__(self, activation=None, use_bias=False,
            bias_initializer='zeros', bias_regularizer=None, bias_constraint=None, **kwargs):
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        super(AdvancedAdd, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        assert len(input_shape) == 2
        assert input_shape[0][-1] == input_shape[1][-1]

        # Create a trainable weight variable for this layer.
        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                    trainable=True, shape=(input_shape[0][-1],), initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer, constraint=self.bias_constraint)
        else:
            self.bias = None
        super(AdvancedAdd, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        assert isinstance(inputs, list)
        assert len(inputs) == 2

        output = inputs[0] + inputs[1]
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        assert len(input_shape) == 2
        assert input_shape[0][-1] == input_shape[1][-1]
        return input_shape[0]


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
    gating mechanism from "Stabilizing transformers for RL" paper
    """
    x_shape = x.shape[-1]

    wr = Dense(x_shape)(y)
    ur = Dense(x_shape)(x)
    r = AdvancedAdd(activation='sigmoid', use_bias=False)([wr, ur])
    wz = Dense(x_shape)(y)
    uz = Dense(x_shape)(x)
    z = AdvancedAdd(activation='sigmoid', use_bias=True)([wz, uz])
    wg = Dense(x_shape)(y)
    ug = Dense(x_shape)(Multiply()([r, x]))
    h = AdvancedAdd(activation='tanh', use_bias=False)([wg, ug])
    g = Add()(
            [Multiply()([Lambda(lambda var: 1. - var)(z), x]),
            Multiply()([z, h])])
    return g


def transformer(input_layer, d_model, n_heads):
    def head(inp):
        q = Dense(d_model)(inp)
        k = Dense(d_model)(inp)
        v = Dense(d_model)(inp)
        # causal = True: adds mask to prevent flow from future to past
        attention = Attention(use_scale=True, causal=True)([q, v, k])
        return attention

    norm1 = LayerNormalization(axis=-1, scale=False, trainable=True)(input_layer)

    heads = []
    for i in range(n_heads):
        heads.append(head(norm1))

    conc = Concatenate(axis=-1)(heads)
    gate1 = gate(input_layer, conc)
    norm2 = LayerNormalization(axis=-1, scale=False, trainable=True)(gate1)
    pmlp = Dense(input_layer.shape[-1])(norm2)
    gate2 = gate(gate1, pmlp)
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
        model_out, self._value_out, h, c = self.rnn_model([inputs, seq_lens] + state)
        return model_out, [h, c]

    @override(ModelV2)
    def get_initial_state(self):
        # Get the initial recurrent state values for the model
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
