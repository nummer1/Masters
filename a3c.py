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


class Actor_Critic_Model(tf.keras.Model):
    def __init__(self, s_size, a_size):
        #Input and visual encoding layers
        super(Actor_Critic_Model, self).__init__(name="AC")
        self.inputs = 0
        self.policy = 0
        self.value = 0


class AC_Agent():
    def __init__(self):
        pass

    def act(self, observation):
        pass

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
