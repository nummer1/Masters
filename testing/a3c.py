# https://cgnicholls.github.io/reinforcement-learning/2017/03/27/a3c.html

# initialise parameters theta_pi for policy pi
# initialise parameters theta_v for value estimate V
# choose max_t (e.g. 20)
# choose max_training_steps (e.g. 100 million)
# choose discount factor gamma
#
# # Update parameters once each time through this loop
# for T in range(max_training_steps):
#     s = current state of emulator
#     initialise array rewards
#     for t in range(max_t):
#         sample action a from pi(a | s; theta_pi)
#         append a to actions
#         append s to states
#         perform action a and get new state s' and reward r
#         append r to rewards
#         s' = s
#         break if terminal
#     # Now train the parameters
#     R = 0
#     if not terminal
#         R = V(s, theta_v) (the estimate of the latest state)
#     # Compute discounted rewards
#     append R to rewards array
#     rewards = discounted(rewards, gamma)
#     # Compute the gradients
#     for each i
#         R_i = sum(rewards[i:])
#         val_i = V(s_i, theta_v)
#         compute grad_theta_pi log pi(a_i, s_i, theta_pi) * (R_i - val_i)
#         compute grad_theta_v (R_i - val_i)^2
#
#     update theta_pi, theta_v by the computed gradients


# # global shared parameter: actor_w, critic_w
# # thread specific parameters: actor_w', critic_w'
#
# T = 0  # global counter
# t = 1  # thread counter
#
# while T < T_max:
#     d_actor = 0
#     d_critic = 0
#
#     # syncrhonize all threads
#     # actor_w' = actor_w, critic_w' = critic_2
#
#     # t_start = t
#     # get state of s_t
#     while True:  # terminal state or t - t_start == t_max
#         # perform a_t according to policy of current thread
#         # recieve reward and new state
#         t += 1
#         T += 1
#     # R = 0 for terminal s_t, V(s_t, critic_w') for non terminal s_t // bootstrat from last state
#     for i in range(t-1, t_start+1):  # including t_start
#         R = r[i] + gamma * R
#         # accumulate gradients with respect to actor_w'
#         # d_actor = d_actor + derivative actor_w (ln(policy(a[i] | s[i], actor_w')) * (R - V(s[i], critic_w')))
#         # accumulate gradients with respect to critic_w'
#         # d_critic = d_critic + derivative critic_w (R - V(s[i], critic_w'))^2 / critic_w'
#     # perform asyncronous update of actor_w using d_actor, critic_w using d_critic



import tensorflow as tf
import gym
import random
from threading import Thread


EPSILON = 0
GAMMA = 0.995


def get_sample(memory, n):
        r = 0.0
        for i in range(n):
            r += memory[i][2] * (GAMMA ** i)
        s, a, _, _  = memory[0]
        _, _, _, s_ = memory[n-1]
        return s, a, r, s_


class Environment(Thread):
    def __init__(self, threadID, name, counter):
        Thread.__init__(self)
        self.env = 0
        self.agent = 0

    def run(self):
        while not self.stop_signal:
            self.runEpisode()

    def runEpisode(self):
        s = self.env.reset()
        while True:
            time.sleep(THREAD_DELAY) # yield
            a = self.agent.act(s)
            sNext, r, done, info = self.env.step(a)
            if done: # terminal state
                sNext = None
            self.agent.train(s, a, r, sNext)
            s = sNext
            if done or self.stop_signal:
                break


class Actor_Critic_Model():
    def __init__(self):
        #Input and visual encoding layers
        # super(Actor_Critic_Model, self).__init__(name="AC")
        # self.inputs = 0
        # self.policy = 0
        # self.value = 0
        self.input = Input(batch_shape=(None, NUM_STATE))
        self.dense = Dense(16, activation='relu')(input)
        self.out_actions = Dense(NUM_ACTIONS, activation='softmax')(dense)
        self.out_value   = Dense(1, activation='linear')(dense)
        self.model = Model(inputs=[input], outputs=[out_actions, out_value])

    def train_push(self, s, a, r, sNext):
        with self.lock_queue:
            self.train_queue[0].append(s)
            self.train_queue[1].append(a)
            self.train_queue[2].append(r)

            if sNext is None:
                self.train_queue[3].append(NONE_STATE)
                self.train_queue[4].append(0.)
            else:
                self.train_queue[3].append(sNext)
                self.train_queue[4].append(1.)


class AC_Agent():
    def __init__(self):
        pass

    def act(self, states):
        if random.random() < epsilon:
            return random.randint(0, NUM_ACTIONS-1)
        else:
            probability = brain.predict_probability(state)
            return np.random.choice(NUM_ACTIONS, p=probability)

    def train(self, s, a, r, sNext):
        a_onehot = np.zeros(NUM_ACTIONS)
        a_onehot[a] = 1
        self.memory.append((s, a_onehot, r, sNext))

        if len(self.memory) >= N_STEP_RETURN:
            s, a, r, sNext_ = get_sample(self.memory, N_STEP_RETURN)
            brain.train_push(s, a, r, sNext)
            self.memory.pop(0)

        if sNext is None:  # terminal state
            while len(self.memory) > 0:
                n = len(self.memory)
                s, a, r, sNext = get_sample(self.memory, n)
                brain.train_push(s, a, r, sNext)
                self.memory.pop(0)

    def train(self, states, actions, rewards):
        # R = 0
        # if not done:
        #     R = # value of s
        # rewards.append(R)
        # cum_rewards = []
        # discounted_sum_r = 0
        # discount = 1
        # for r in reversed(rewards):
        #     discounted_sum_r += r * discount
        #     discount *= discount_factor
        #     cum_rewards.append(discounted_sum_r)
        # cum_rewards.reverse()
        pass


def to_state_array(observation):
    # should be inside some class
    pass


def main():
    max_t = 20
    max_training_steps = 100000
    discount_factor = 0.995
    agent = AC_Agent()
    env = gym.make('FrozenLake-v0')

    for T in range(max_training_steps):
        ob = env.reset()  # get initial observation
        s = to_state_array(ob)
        rewards = []
        actions = []
        states = []
        for t in range(max_t):
            a = agent.act(ob)
            states.append(s)
            actions.append(a)
            ob, r, done, info = env.step(a)
            rewards.append(r)
            s = to_state_array(ob)
            if done:
                break

        # train parameters
        agent.train(states, actions, rewards)


if __name__ == "__main__":
    main()
