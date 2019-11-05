import tensorflow as tf
import gym
from gated_transformer import *


MAX_LENGTH = 10  # size of max size data
MODEL_SIZE = 100  # size of Transformer output
H = 1  # number of attention heads
NUM_LAYERS = 1  # number of Transformers
VOCAB_SIZE = 10  # size of embedding

pes = []
for i in range(MAX_LENGTH):
    pes.append(positional_embedding(i, MODEL_SIZE))

pes = np.concatenate(pes, axis=0)
pes = tf.constant(pes, dtype=tf.float32)

transformer = Transformer(VOCAB_SIZE, MODEL_SIZE, NUM_LAYERS, H, pes)

sequence_in = tf.constant([[1, 2, 3, 4, 6, 7, 8, 0, 0],
                           [1, 2, 3, 4, 6, 7, 8, 0, 0]])
transformer_out = transformer(sequence_in)

print('Input vocabulary size', VOCAB_SIZE)
print('Encoder input shape', sequence_in.shape)
print('Encoder output shape', transformer_out.shape)


transformer.compile('rmsprop', loss='categorical_crossentropy', metrics={'ffn_out': ['accuracy']})
# transformer.fit(x=None, y=None, batch_size=32, epochs=1, verbose=1, validation_split=0.0, shuffle=True)


class RandomAgent:
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


if __name__ == '__main__':
    # env = gym.make('Copy-v0')
    env = gym.make('FrozenLake-v0')
    env.reset()

    agent = RandomAgent(env.action_space)
    for i in range(10):
        action = agent.act(env, 0, False)
        ob, reward, done, info = env.step(action)
        print(ob, reward, done, info)
        if done:
            break

    env.render(mode='human')
