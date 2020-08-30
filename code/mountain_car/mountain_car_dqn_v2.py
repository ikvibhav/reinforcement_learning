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
def neural_network_definition(input_shape, output_shape):
    model = Sequential()
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
        self.env = gym.make('MountainCar-v0')
        self.state_size = self.env.observation_space.shape[0]   #2
        self.action_size = self.env.action_space.n              #3       
        self.EPISODES = 1000
        self.SHOW_EPISODES = 10
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
        minibatch = random.sample(self.MEMORY_Q, min(len(self.MEMORY_Q), self.training_batch_size))
        state = np.zeros((self.training_batch_size, self.state_size))
        next_state = np.zeros((self.training_batch_size, self.state_size))
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
                target[i][action[i]] = reward[i]
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
            state = np.reshape(state, [1, self.state_size])
            done = False
            render = False
            i = 0

            if e % self.SHOW_EPISODES == 0:
                render = True
            else:
                render = False

            while not done:
                if render:
                    self.env.render()

                if np.random.random() > self.epsilon:
                    action = np.argmax(self.model.predict(state))
                else:
                    action = np.random.randint(0, self.action_size)

                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])

                #print(next_state[0][0])
                #print()

                if not done:
                    reward = reward
                elif next_state[0][0] < self.env.goal_position and next_state[0][0] > -self.env.goal_position:   
                    reward =  next_state[0][0] - self.env.goal_position       #To try, reward =  next_state[0][0] - self.env.goal_position 
                    
                print("next_state: {}, reward: {}".format(next_state[0][0], reward))

                self.remember(state, action, reward, next_state, done)

                state = next_state
                i= i + 1
                if done:                   
                    print("episode: {}/{}, episode steps: {}, e: {:.2}".format(e, self.EPISODES, i, self.epsilon))
                    if next_state[0][0] >= self.env.goal_position:
                        print("Saving trained model as mountaincar-dqn.h5")
                        self.save("mountaincar-dqn.h5")
                        return
                self.replay()

    def test(self):
        self.load("mountaincar-dqn.h5")
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