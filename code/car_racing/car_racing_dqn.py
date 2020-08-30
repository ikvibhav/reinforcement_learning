import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import gym
import random
import numpy as np
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Conv2D, Flatten
from keras.optimizers import Adam, RMSprop
from collections import deque


#env = gym.make("CartPole-v1")

'''
#This definition does not work well.
def neural_network_definition(input_shape, output_shape):
    model = Sequential()
    model.add(Input(input_shape=input_shape))
    model.add(Conv2D(16, 5, 5, activation='relu'))   
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, 5, 5, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, 5, 5, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(output_shape, activation="relu", kernel_initializer='he_uniform'))
    
    model.compile(loss="mse", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])
    model.summary()

    return model
'''
def neural_network_definition(input_shape, output_shape):
    X_input = Input(input_shape)

    # 'Dense' is the basic form of a neural network layer
    # Input Layer of state size(4) and Hidden Layer with 512 nodes
    X = Conv2D(16, (8, 8), activation='elu')(X_input)
    X = Conv2D(32, (5, 5), activation='elu')(X)
    X = Conv2D(32, (3, 3), activation='elu')(X)
    X = Flatten()(X)
    X = Dense(16, activation="relu", kernel_initializer='he_uniform')(X)

    X = Dense(output_shape, activation="linear", kernel_initializer='he_uniform')(X)

    model = Model(inputs = X_input, outputs = X, name='CarRacing_DQN_model')
    model.compile(loss="mse", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])

    model.summary()
    return model


class Cart_Pole_DQNAgent:
    def __init__(self):
        self.env = gym.make('CarRacing-v0')
        self.state_size = self.env.observation_space.shape
        self.action_size = self.env.action_space.shape[0]
        self.EPISODES = 1000
        self.MEMORY_Q = deque(maxlen=2000)

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999
        self.training_batch_size = 64
        self.training_start = 1000

        self.model = neural_network_definition(input_shape=self.state_size, output_shape = self.action_size)

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

        #tO CEHCK IF TARGET IS q-VALUE Or ACTION
        target = self.model.predict(state)
        target_next = self.model.predict(next_state)

        for i in range(self.training_batch_size):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))

        self.model.fit(state, target, batch_size=self.training_batch_size, verbose=0)

    def run(self):
        for e in range(self.EPISODES):
            state = self.env.reset()
            #state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                self.env.render()

                #if np.random.random() <= self.epsilon:
                #    action = random.randrange(self.action_size)
                #else:
                print(self.state_size)
                print(np.shape(state))
                action = self.model.predict(state)

                print(action)
                next_state, reward, done, _ = self.env.step(action)
                #next_state = np.reshape(next_state, [1, self.state_size])

                #if not done or i == self.env._max_episode_steps-1:
                #    reward = reward
                #else:   #Enters here when i = env._max_episode_steps and it goes to done
                #   reward = -100

                self.remember(state, action, reward, next_state, done)

                state = next_state
                i= i + 1
                if done:                   
                    print("episode: {}/{}, score: {}, e: {:.2}".format(e, self.EPISODES, i, self.epsilon))
                    if i == 500:
                        print("Saving trained model as carracing-dqn.h5")
                        self.save("carracing-dqn.h5")
                        return
                self.replay()

    def test(self):
        self.load("carracing-dqn.h5")
        for e in range(self.EPISODES):
            state = self.env.reset()
            #state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                self.env.render()
                action = np.argmax(self.model.predict(state))
                next_state, reward, done, _ = self.env.step(action)
                #state = np.reshape(next_state, [1, self.state_size])
                i += 1
                if done:
                    print("episode: {}/{}, score: {}".format(e, self.EPISODES, i))
                    break    

if __name__ == "__main__":
    agent = Cart_Pole_DQNAgent()
    agent.run()
    #agent.test()





