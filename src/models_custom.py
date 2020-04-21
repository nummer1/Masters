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
        Flatten, Input, Lambda, Layer, LayerNormalization, LSTM, MaxPooling2D, Multiply, Reshape, \
        TimeDistributed
from tensorflow.keras import backend as K


def preproc(inputs):
    inputs = tf.math.scalar_mul(1/255, inputs)
    # inputs = tf.reshape(inputs, [None, 64, 64, 3])
    return inputs


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


def res_block(input_layer, filters, name):
    relu = TimeDistributed(Activation('relu'), name=name+"_relu")(input_layer)
    conv1 = TimeDistributed(Conv2D(
        filters=filters, kernel_size=(4, 4), strides=(1, 1), padding='same',
        dilation_rate=(1, 1), activation=tf.nn.relu), name=name+"_conv1")(relu)
    conv2 = TimeDistributed(Conv2D(
        filters=filters, kernel_size=(4, 4), strides=(1, 1), padding='same',
        dilation_rate=(1, 1), activation=tf.keras.activations.linear), name=name+"_conv2")(conv1)
    add = Add(name=name+"_add")([input_layer, conv2])
    return add


def res_block_notime(input_layer, filters, name):
    relu = Activation('relu', name=name+"_relu")(input_layer)
    conv1 = Conv2D(
        filters=filters, kernel_size=(4, 4), strides=(1, 1), padding='same',
        dilation_rate=(1, 1), activation=tf.nn.relu, name=name+"_conv1")(relu)
    conv2 = Conv2D(
        filters=filters, kernel_size=(4, 4), strides=(1, 1), padding='same',
        dilation_rate=(1, 1), activation=tf.keras.activations.linear, name=name+"_conv2")(conv1)
    add = Add(name=name+"_add")([input_layer, conv2])
    return add


def conv_block(input_layer, filters, name):
    conv = TimeDistributed(Conv2D(
        filters=filters, kernel_size=(4, 4), strides=(1, 1), padding='same',
        dilation_rate=(1, 1), activation=tf.keras.activations.linear), name=name+"_conv1")(input_layer)
    max = TimeDistributed(MaxPooling2D(
        pool_size=(2, 2), strides=(2, 2), padding='same'), name=name+"_pool")(conv)
    res1 = res_block(max, filters, name+"_res1")
    res2 = res_block(res1, filters, name+"_res2")
    return res2


def conv_block_notime(input_layer, filters, name):
    conv = Conv2D(
        filters=filters, kernel_size=(4, 4), strides=(1, 1), padding='same',
        dilation_rate=(1, 1), activation=tf.keras.activations.linear, name=name+"_conv1")(input_layer)
    max = MaxPooling2D(
        pool_size=(2, 2), strides=(2, 2), padding='same', name=name+"_pool")(conv)
    res1 = res_block_notime(max, filters, name+"_res1")
    res2 = res_block_notime(res1, filters, name+"_res2")
    return res2


def conv_network(input_layer):
    # x3 conv_block with [16, 32, 32] filters
    reshape = TimeDistributed(Reshape((64, 64, 3)), name="reshape")(input_layer)
    block1 = conv_block(reshape, 16, "block1")
    block2 = conv_block(block1, 32, "block2")
    block3 = conv_block(block2, 32, "block3")
    relu = TimeDistributed(Activation('relu'), name='relu_out')(block3)
    flatten = TimeDistributed(Flatten(), name="flatten")(relu) # data_format="channels_last" (default)
    return flatten


def conv_network_notime(input_layer):
    # x3 conv_block with [16, 32, 32] filters
    reshape = Reshape((64, 64, 3), name="reshape")(input_layer)
    block1 = conv_block_notime(reshape, 16, "block1")
    block2 = conv_block_notime(block1, 32, "block2")
    block3 = conv_block_notime(block2, 32, "block3")
    relu = Activation('relu', name='relu_out')(block3)
    flatten = Flatten(name="flatten")(relu) # data_format="channels_last" (default)
    return flatten


def simple_conv_network(input_layer):
    reshape = TimeDistributed(Reshape((64, 64, 3)), name="reshape")(input_layer)
    conv1 = TimeDistributed(Conv2D(
        filters=16, kernel_size=(8, 8), strides=(4, 4), padding='same',
        dilation_rate=(1, 1), activation=tf.nn.relu), name="conv1")(reshape)
    conv2 = TimeDistributed(Conv2D(
        filters=32, kernel_size=(4, 4), strides=(2, 2), padding='same',
        dilation_rate=(1, 1), activation=tf.nn.relu), name="conv2")(conv1)
    flatten = TimeDistributed(Flatten(), name="flatten")(conv2) # data_format="channels_last" (default)
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
    # NOTE: setting bias greater than 0 can greatly improve learning speed (according to paper)

    return gate2


class TransformerCustomModel(RecurrentTFModelV2):
    """
    input shape is (batch, time_steps, input_size), which is same as LSTM
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name,
            cell_size=256, d_model=32, n_heads=8, plot_model=False):
        super(TransformerCustomModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        self.cell_size = cell_size

        input_layer = Input(
            shape=(None, 64 * 64 * 3), name="input")
        # seq_in = Input(shape=(), name="seq_in", dtype=tf.int32)

        flatten = conv_network(input_layer)
        shorten = Dense(256, activation=tf.nn.relu, name="shorten")(flatten)

        # NOTE: add propper positional encoding
        trans1 = transformer(shorten, d_model, n_heads)
        # trans2 = transformer(trans1, d_model, n_heads)

        logits = Dense(15, activation=tf.keras.activations.softmax, name="logits")(trans1)
        values = Dense(1, activation=None, name="values")(trans1)

        # Create the RNN model
        self.rnn_model = tf.keras.Model(
            inputs=[input_layer],
            outputs=[logits, values])
        self.register_variables(self.rnn_model.variables)
        self.rnn_model.summary()

        if plot_model:
            tf.keras.utils.plot_model(self.rnn_model, to_file='model_transformer.png', show_shapes=True)

    @override(RecurrentTFModelV2)
    def forward_rnn(self, inputs, state, seq_lens):
        inputs = preproc(inputs)
        # model_out, self._value_out = self.rnn_model([inputs, seq_lens])
        model_out, self._value_out = self.rnn_model([inputs])
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
    def __init__(self, obs_space, action_space, num_outputs, model_config, name,
            cell_size=256, plot_model=False):
        super(LSTMCustomModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        self.cell_size = cell_size

        input_layer = Input(
            shape=(None, 64 * 64 * 3), name="input")
        state_in_h = Input(shape=(cell_size, ), name="h")
        state_in_c = Input(shape=(cell_size, ), name="c")
        seq_in = Input(shape=(), name="seq_in", dtype=tf.int32)

        flatten = conv_network(input_layer)

        # Preprocess observation with a hidden layer and send to LSTM cell
        dense1 = Dense(256, activation=tf.nn.relu, name="dense1")(flatten)

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

        if plot_model:
            tf.keras.utils.plot_model(self.rnn_model, to_file='model_lstm.png', show_shapes=True)

    @override(RecurrentTFModelV2)
    def forward_rnn(self, inputs, state, seq_lens):
        inputs = preproc(inputs)
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


class SimpleCustomModel(RecurrentTFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name,
            cell_size=256, plot_model=False):
        super(SimpleCustomModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        self.cell_size = cell_size

        input_layer = Input(
            shape=(None, 64 * 64 * 3), name="inputs")
        state_in_h = Input(shape=(cell_size, ), name="h")
        state_in_c = Input(shape=(cell_size, ), name="c")
        seq_in = Input(shape=(), name="seq_in", dtype=tf.int32)

        flatten = simple_conv_network(input_layer)

        # Preprocess observation with a hidden layer and send to LSTM cell
        dense1 = Dense(256, activation=tf.nn.relu, name="dense1")(flatten)

        lstm_out, state_h, state_c = LSTM(
            cell_size, return_sequences=True, return_state=True, name="lstm")(
                inputs=dense1,
                mask=tf.sequence_mask(seq_in),
                initial_state=[state_in_h, state_in_c])

        # Postprocess LSTM output with another hidden layer and compute values
        logits = Dense(
            15, activation=tf.keras.activations.softmax, name="logits")(lstm_out)
        values = Dense(
            1, activation=None, name="values")(lstm_out)

        # Create the RNN model
        self.rnn_model = tf.keras.Model(
            inputs=[input_layer, seq_in, state_in_h, state_in_c],
            outputs=[logits, values, state_h, state_c])
        self.register_variables(self.rnn_model.variables)
        self.rnn_model.summary()

        if plot_model:
            tf.keras.utils.plot_model(self.rnn_model, to_file='model_simple.png', show_shapes=True)

    @override(RecurrentTFModelV2)
    def forward_rnn(self, inputs, state, seq_lens):
        inputs = preproc(inputs)
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


class DenseCustomModel(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name,
            cell_size=256, plot_model=False):
        super(DenseCustomModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        self.cell_size = cell_size

        input_layer = Input(
            shape=(None, 64 * 64 * 3), name="input")

        flatten = conv_network_notime(input_layer)

        # Preprocess observation with a hidden layer and send to LSTM cell
        dense1 = Dense(256, activation=tf.nn.relu, name="dense1")(flatten)

        # Postprocess LSTM output with another hidden layer and compute values
        logits = Dense(
            15, activation=tf.keras.activations.softmax, name="logits")(dense1)
        values = Dense(
            1, activation=None, name="values")(dense1)

        # Create the model
        self.rnn_model = tf.keras.Model(
            inputs=[input_layer],
            outputs=[logits, values])
        self.register_variables(self.rnn_model.variables)
        self.rnn_model.summary()

        if plot_model:
            tf.keras.utils.plot_model(self.rnn_model, to_file='model_dense.png', show_shapes=True)

    @override(TFModelV2)
    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.rnn_model(input_dict["obs"])
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


class LSTMGuessingGameModel(RecurrentTFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(LSTMGuessingGameModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        self.cell_size = model_config["custom_options"]["cell_size"]

        input_layer = Input(
            shape=(None, 4), name="input")
        state_in_h = Input(shape=(self.cell_size, ), name="h")
        state_in_c = Input(shape=(self.cell_size, ), name="c")
        seq_in = Input(shape=(), name="seq_in", dtype=tf.int32)

        # Preprocess observation with a hidden layer and send to LSTM cell
        dense1 = Dense(self.cell_size, activation=tf.nn.relu, name="dense1")(input_layer)
        dense2 = Dense(self.cell_size, activation=tf.nn.relu, name="dense2")(dense1)

        lstm_out, state_h, state_c = LSTM(
            self.cell_size, return_sequences=True, return_state=True, name="lstm")(
                inputs=dense2,
                mask=tf.sequence_mask(seq_in),
                initial_state=[state_in_h, state_in_c])

        dense3 = Dense(self.cell_size, activation=tf.nn.relu, name="dense3")(lstm_out)

        # Postprocess LSTM output with another hidden layer and compute values
        logits = Dense(
            num_outputs, activation=tf.keras.activations.softmax, name="logits")(dense3)
        values = Dense(
            1, activation=None, name="values")(dense3)

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


class TransformerGuessingGameModel(RecurrentTFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(TransformerGuessingGameModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        self.cell_size = model_config["custom_options"]["cell_size"]
        self.n_heads = model_config["custom_options"]["n_heads"]

        input_layer = Input(
            shape=(None, 4), name="input")

        # Preprocess observation with a hidden layer and send to LSTM cell
        dense1 = Dense(self.cell_size, activation=tf.nn.relu, name="dense1")(input_layer)
        dense2 = Dense(self.cell_size, activation=tf.nn.relu, name="dense2")(dense1)

        trans1 = transformer(dense2, self.cell_size, self.n_heads)
        trans2 = transformer(trans1, self.cell_size, self.n_heads)

        # Postprocess LSTM output with another hidden layer and compute values
        logits = Dense(
            num_outputs, activation=tf.keras.activations.linear, name="logits")(trans2)
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
        model_out, self._value_out = self.rnn_model([inputs])
        return model_out, state

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



class DenseGuessingGameModel(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(DenseGuessingGameModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        self.cell_size = model_config["custom_options"]["cell_size"]

        self.inputs = Input(shape=obs_space.shape, name="input")

        # Preprocess observation with a hidden layer and send to LSTM cell
        dense1 = Dense(self.cell_size, activation=tf.nn.relu, name="dense1")(self.inputs)
        dense2 = Dense(self.cell_size, activation=tf.nn.relu, name="dense2")(dense1)
        dense3 = Dense(self.cell_size, activation=tf.nn.relu, name="dense3")(dense2)
        dense4 = Dense(self.cell_size, activation=tf.nn.relu, name="dense4")(dense3)

        # Postprocess LSTM output with another hidden layer and compute values
        logits = Dense(num_outputs, activation=tf.keras.activations.linear, name="logits")(dense4)
        values = Dense(1, activation=None, name="values")(dense4)

        self.base_model = tf.keras.Model(self.inputs, [logits, values])
        self.register_variables(self.base_model.variables)
        self.base_model.summary()

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


class ProcgenPreprocessor(Preprocessor):
    """
    Custom preprocessor is depreaceated by rllib, not in use
    """
    def _init_shape(self, obs_space, options):
        # obs_space is Box(64, 64, 3)
        return obs_space.shape

    def transform(self, observation):
        # observation is numpy array with shape [64, 64, 3]
        return observation
