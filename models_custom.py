# TODO: build_tf_policy` and `build_torch_policy'


from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models import ModelCatalog
from ray.rllib.models.preprocessors import Preprocessor
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.recurrent_tf_modelv2 import RecurrentTFModelV2
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.annotations import override

import tensorflow as tf
import numpy as np

import transformer


def convNetwork(input_layer):
    reshape = tf.keras.layers.TimeDistributed(tf.keras.layers.Reshape((64, 64, 3)))(input_layer)
    conv1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(
        filters=16, kernel_size=(4, 4), strides=(1, 1), name="conv1",
        padding='same', data_format=None, dilation_rate=(1, 1), activation=tf.nn.relu))(reshape)
    maxpool1 = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2), name="maxpool1", strides=None, padding='valid', data_format=None))(conv1)

    flatten = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten(name="flatten"))(maxpool1) # data_format="channels_last" (default)
    return flatten


class TransformerCustomModel(RecurrentTFModelV2):
    """
    input is embedding from previous layer
        dimension = T(time steps) x D(hidden dimensions)
        D is some embedding of the time steps in an RL setting
            (eg. image ran through convolutional network)


    input shape is (batch, time_steps, input_size), which is same as LSTM
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, hiddens_size=16, cell_size=8):
        super(TransformerCustomModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        self.cell_size = cell_size

        input_layer = tf.keras.layers.Input(
            shape=(None, 64 * 64 * 3), name="inputs")
        seq_in = tf.keras.layers.Input(shape=(), name="seq_in", dtype=tf.int32)

        # d_model & n_heads == 0
        flatten = convNetwork(input_layer)

        # TODO: add embedding layer
        # TODO: add positional encoding

        # self, d_model, num_heads, dff, rate=0.1
        # TODO: d_model = 16384 ?
        # trans = transformer.EncoderLayer(64*64*3, 2, 10, rate=0.1)(flatten, training=True, mask=tf.sequence_mask(seq_in))
        # works with d_model = 16384
        trans = transformer.EncoderLayer(64, 2, 64, rate=0.1)(flatten, training=True, mask=None)

        logits = tf.keras.layers.Dense(
            15, activation=tf.keras.activations.linear, name="logits")(trans)
        values = tf.keras.layers.Dense(
            1, activation=None, name="values")(trans)

        # Create the RNN model
        self.rnn_model = tf.keras.Model(
            inputs=[input_layer, seq_in],
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
        # return tf.reshape(self._value_out, [-1])
        return self._value_out


class LSTMCustomModel(RecurrentTFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, hiddens_size=16, cell_size=8):
        super(LSTMCustomModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        self.cell_size = cell_size

        input_layer = tf.keras.layers.Input(
            shape=(None, 64 * 64 * 3), name="inputs")
        state_in_h = tf.keras.layers.Input(shape=(cell_size, ), name="h")
        state_in_c = tf.keras.layers.Input(shape=(cell_size, ), name="c")
        seq_in = tf.keras.layers.Input(shape=(), name="seq_in", dtype=tf.int32)

        flatten = convNetwork(input_layer)
        # reshape = tf.keras.layers.TimeDistributed(tf.keras.layers.Reshape((64, 64, 3)))(input_layer)
        # conv1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(
        #     filters=16, kernel_size=(4, 4), strides=(1, 1), name="conv1",
        #     padding='same', data_format=None, dilation_rate=(1, 1), activation=tf.nn.relu))(reshape)
        # maxpool1 = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(
        #     pool_size=(2, 2), name="maxpool1", strides=None, padding='valid', data_format=None))(conv1)
        #
        # flatten = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten(name="flatten"))(maxpool1) # data_format="channels_last" (default)

        # Preprocess observation with a hidden layer and send to LSTM cell
        # dense1 = tf.keras.layers.Dense(
        #     16, activation=tf.nn.relu, name="dense1")(maxpool1)
        # print("dense1:", dense1.shape)flatten = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten(name="flatten"))(maxpool1)

        lstm_out, state_h, state_c = tf.keras.layers.LSTM(
            cell_size, return_sequences=True, return_state=True, name="lstm")(
                inputs=flatten,
                mask=tf.sequence_mask(seq_in),
                initial_state=[state_in_h, state_in_c])

        # Postprocess LSTM output with another hidden layer and compute values
        logits = tf.keras.layers.Dense(
            15, activation=tf.keras.activations.linear, name="logits")(lstm_out)
        values = tf.keras.layers.Dense(
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
