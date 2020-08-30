import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import gym
import random
import numpy as np
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import Adam, RMSprop
from collections import deque


#env = gym.make("CartPole-v1")
'''
#This definition does not work well.
def neural_network_definition(input_shape, output_shape):
    model = Sequential()
    model.add(Input(input_shape=input_shape))
    model.add(Dense(512, input_shape=input_shape, activation="relu", kernel_initializer='he_uniform'))
    model.add(Dense(256, activation="relu", kernel_initializer='he_uniform'))
    model.add(Dense(64, activation="relu", kernel_initializer='he_uniform'))
    model.add(Dense(output_shape, activation="relu", kernel_initializer='he_uniform'))
    model.compile(loss="mse", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])
    model.summary()

    return model
'''
def neural_network_definition(input_shape, output_shape):
    X_input = Input(input_shape)

    # 'Dense' is the basic form of a neural network layer
    # Input Layer of state size(4) and Hidden Layer with 512 nodes
    X = Dense(128, input_shape=input_shape, activation="relu", kernel_initializer='he_uniform')(X_input)
    # Hidden layer with 256 nodes
    X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)
    
    # Hidden layer with 64 nodes
    X = Dense(32, activation="relu", kernel_initializer='he_uniform')(X)

    # Output Layer with # of actions: 2 nodes (left, right)
    X = Dense(output_shape, activation="linear", kernel_initializer='he_uniform')(X)

    model = Model(inputs = X_input, outputs = X, name='CartPole_DQN_model')
    model.compile(loss="mse", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])

    model.summary()
    return model


class Cart_Pole_DQNAgent:
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.state_size = self.env.observation_space.shape[0]   #4
        self.action_size = self.env.action_space.n              #2
        self.EPISODES = 1000
        self.MEMORY_Q = deque(maxlen=2000)

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999
        self.training_batch_size = 64
        self.training_start = 1000

        self.model = neural_network_definition(input_shape=(self.state_size,), output_shape = self.action_size)

    def remember(self, state, action, reward, next_state, done):
        self.MEMORY_Q.append((state, action, reward, next_state, done))
        if len(self.MEMORY_Q) > self.training_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def replay(self):
        if len(self.MEMORY_Q) < self.training_start:
            return
        minibatch = random.sample(self.MEMORY_Q, min(len(self.MEMORY_Q), self.training_batch_size))     # min(lenght memQ, batch size) samples of Memory_Q
        state = np.zeros((self.training_batch_size, self.state_size))   #64*4
        next_state = np.zeros((self.training_batch_size, self.state_size))  #64*4
        action, reward, done = [], [], []

        for i in range(self.training_batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])            

        target = self.model.predict(state)
        target_next = self.model.predict(next_state)

        for i in range(self.training_batch_size):
            if done[i]:
                target[i][action[i]] = reward[i]        #If its done, then assign the same reward value
            else:
                target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))

        self.model.fit(state, target, batch_size=self.training_batch_size, verbose=0)

    def load(self, name):
        self.model = load_model(name)

    def save(self, name):
        self.model.save(name)

    def run(self):
        for e in range(self.EPISODES):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size]) #1*4
            done = False
            i = 0
            while not done:
                self.env.render()

                if np.random.random() <= self.epsilon:
                    action = random.randrange(self.action_size)
                else:
                    action = np.argmax(self.model.predict(state))

                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])

                if not done or i == self.env._max_episode_steps-1:
                    reward = reward
                else:   #Enters here when i = env._max_episode_steps and it goes to done
                    reward = -100

                print("next_state: {}, reward: {}".format(next_state[0][0], reward))

                self.remember(state, action, reward, next_state, done)

                state = next_state
                i= i + 1
                if done:                   
                    print("episode: {}/{}, score: {}, e: {:.2}".format(e, self.EPISODES, i, self.epsilon))
                    if i == 500:
                        print("Saving trained model as cartpole-dqn.h5")
                        self.save("cartpole-dqn.h5")
                        return
                self.replay()

    def test(self):
        self.load("cartpole-dqn.h5")
        for e in range(self.EPISODES):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                self.env.render()
                action = np.argmax(self.model.predict(state))
                next_state, reward, done, _ = self.env.step(action)
                state = np.reshape(next_state, [1, self.state_size])
                i += 1
                if done:
                    print("episode: {}/{}, score: {}".format(e, self.EPISODES, i))
                    break    

if __name__ == "__main__":
    agent = Cart_Pole_DQNAgent()
    agent.run()
    #agent.test()





