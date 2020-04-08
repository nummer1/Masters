import os
import tensorflow as tf
import numpy as np

# os.environ["TF_KERAS"] = "1"
os.environ["TF_EAGER"] = "0"

import keras_transformer_xl.keras_transformer_xl.transformer_xl as tr_xl
import keras_transformer_xl.keras_transformer_xl.sequence as sequence


class TransformerXLCustomModel(RecurrentTFModelV2):
    """
    based on https://github.com/CyberZHG/keras-transformer
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name,
            cell_size=256, d_model=8, n_heads=8, plot_model=False):
        super(TransformerXLCustomModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        self.cell_size = cell_size

        input_layer = Input(
            shape=(None, 64 * 64 * 3), name="input")
        # seq_in = Input(shape=(), name="seq_in", dtype=tf.int32)

        # d_model & n_heads == 0
        flatten = conv_network(input_layer)
        shorten = Dense(256, activation=tf.nn.relu, name="shorten")(flatten)

        trans = tr_xl.tr_xl.build_transformer_xl(
             units=64,
             embed_dim=256,
             hidden_dim=64,
             num_token=20,
             num_block=2,
             num_head=4,
             batch_size=15000,
             memory_len=20,
             target_len=256,
             clamp_len=10,
        )(shorten)

        # trans1 = transformer(shorten, d_model, n_heads)
        # trans2 = transformer(trans1, d_model, n_heads)

        logits = Dense(15, activation=tf.keras.activations.linear, name="logits")(trans)
        values = Dense(1, activation=None, name="values")(trans)

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


if __name__ == "__main__":
    class DummySequence(tf.keras.utils.Sequence):

        def __init__(self):
            pass

        def __len__(self):
            return 10

        def __getitem__(self, index):
            return np.ones((3, 5 * (index + 1))), np.ones((3, 5 * (index + 1), 3))


    model = tr_xl.build_transformer_xl(
        units=4,
        embed_dim=4,
        hidden_dim=4,
        num_token=3,
        num_block=3,
        num_head=2,
        batch_size=3,
        memory_len=20,
        target_len=10,
    )

    seq = sequence.MemorySequence(
        model=model,
        sequence=DummySequence(),
        target_len=10,
    )

    print(tf.__version__)

    print(model)
    print(seq)
    print(dir(seq))
    print(seq.model)
    print(seq.sequence.shape)
    print(seq.shape)

    pred = model.predict(seq, verbose=True)
    # print(pred)
