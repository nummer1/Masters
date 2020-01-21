# TODO: build_tf_policy` and `build_torch_policy'


from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.recurrent_tf_modelv2 import RecurrentTFModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override

import torch.nn as nn
import tensorflow as tf
import numpy as np


class TransformerCustomModel(TorchModelV2, nn.Module):
    """Pytorch Transformer"""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, hiddens_size=16, cell_size=8):
    # , hiddens_size=16, cell_size=8):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        # self.cell_size = cell_size
        self.obs_size = _get_size(obs_space)
        self.rnn_hidden_dim = model_config["lstm_cell_size"]
        self.fc1 = nn.Linear(self.obs_size, self.rnn_hidden_dim)
        self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.fc2 = nn.Linear(self.rnn_hidden_dim, num_outputs)

    @override(TorchModelV2)
    def get_initial_state(self):
        # return [
        #     np.zeros(self.cell_size, np.float32),
        #     np.zeros(self.cell_size, np.float32),
        # ]
        return [self.fc1.weight.new(1, self.rnn_hidden_dim).zero_().squeeze(0)]

    @override(TorchModelV2)
    def forward(self, input_dict, hidden_state, seq_lens):
        x = nn.functional.relu(self.fc1(input_dics["obs_flat".float()]))
        h_in = hidden_state[0].reshape(-1, self.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, [h]

    # @override(ModelV2)
    # def value_function(self):
    #     return tf.reshape(self._value_out, [-1])


class LSTMCustomModel(RecurrentTFModelV2):
    """Example of using the Keras functional API to define a RNN model."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, hiddens_size=16, cell_size=8):
        super(LSTMCustomModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        self.cell_size = cell_size

        # Define input layers
        input_layer = tf.keras.layers.Input(
            shape=(None, obs_space.shape[0]), name="inputs")
        state_in_h = tf.keras.layers.Input(shape=(cell_size, ), name="h")
        state_in_c = tf.keras.layers.Input(shape=(cell_size, ), name="c")
        seq_in = tf.keras.layers.Input(shape=( ), name="seq_in", dtype=tf.int32)

        # Preprocess observation with a hidden layer and send to LSTM cell
        dense1 = tf.keras.layers.Dense(
            hiddens_size, activation=tf.nn.relu, name="dense1")(input_layer)
        lstm_out, state_h, state_c = tf.keras.layers.LSTM(
            cell_size, return_sequences=True, return_state=True, name="lstm")(
                inputs=dense1,
                mask=tf.sequence_mask(seq_in),
                initial_state=[state_in_h, state_in_c])

        # Postprocess LSTM output with another hidden layer and compute values
        print("NUM_OUTPUTS:", num_outputs)
        print("OBS_SPACE:", obs_space)
        print("ACTION_SPACE:", action_space)

        logits = tf.keras.layers.Dense(
            self.num_outputs,
            activation=tf.keras.activations.linear,
            name="logits")(lstm_out)
        values = tf.keras.layers.Dense(
            1, activation=None, name="values")(lstm_out)

        # Create the RNN model
        self.rnn_model = tf.keras.Model(
            inputs=[input_layer, seq_in, state_in_h, state_in_c],
            outputs=[logits, values, state_h, state_c])
        self.register_variables(self.rnn_model.variables)
        # self.register_variables(input_layer)
        self.rnn_model.summary()

    @override(RecurrentTFModelV2)
    def forward_rnn(self, inputs, state, seq_lens):
        model_out, self._value_out, h, c = self.rnn_model([inputs, seq_lens] + state)
        return model_out, [h, c]
        # model_out, self._value_out = self.rnn_model([inputs, seq_lens] + state)
        # return model_out

    @override(ModelV2)
    def get_initial_state(self):
        return [
            np.zeros(self.cell_size, np.float32),
            np.zeros(self.cell_size, np.float32),
        ]

    @override(ModelV2)
    def value_function(self):
        return tf.reshape(self._value_out, [-1])


def _get_size(obs_space):
    return get_preprocessor(obs_space)(obs_space).size


# if __name__ == "__main__":
#     import gym
#     env = gym.make('GuessingGame-v0')
#     print(env.observation_space)
#     model = LSTMCustomModel(np.array([2, ]), env.action_space, 1, None, 'test')
