

# class DQNAgent
import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers


class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.001, load = ''):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1024)

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate

        #self.model = self.build_model()
        if load != '':
            self.model = tf.keras.models.load_model(load)
        else:
            self.build_model()

    def build_model(self):
        inputs = layers.Input(shape=(self.state_size,))
        x = layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(0.001))(inputs)
        #x = layers.BatchNormalization()(x)    
        #x = layers.BatchNormalization()(x)    
        x = layers.Dense(64, activation='relu',
                     kernel_regularizer=regularizers.l2(0.001))(x)        
        x = layers.Dense(32, activation='relu',
                     kernel_regularizer=regularizers.l2(0.001))(x)
        #x = layers.BatchNormalization()(x)        
        outputs = layers.Dense(self.action_size, activation='linear')(x)
    
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=self.learning_rate))
        self.model = model    
        

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            if np.random.rand() <= 0.001:
                print(f'random decision')
                return random.randrange(self.action_size)
            else:  #  Wähle aus Maske der sinnigen Actions
                #print(f'In agent class last action-size entries of state submitted: {state[0][-self.action_size:]}')
                usable_actions = [index for (index, item) in enumerate(state[0][-self.action_size:]) if item == 1]
                random.shuffle(usable_actions)
                #print(f'usable actions: {usable_actions}')
                return usable_actions[0]

        ### Vorfiltern sinniger Fahraufträge
        q_values = self.model.predict(state, verbose=0)
        filter_vec = [x*50 for x in state[0][-self.action_size:]]  # Hier erhalten sinnvolle Aufträge (1) ein extremes Gewicht
        action_vec = q_values[0] + filter_vec
        return np.argmax(action_vec)

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        states, targets = [], []

        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(np.array([state]), verbose=0)[0]
            if done:
                target[action] = reward
            else:
                next_q = self.model.predict(np.array([next_state]), verbose=0)[0]
                target[action] = reward + self.gamma * np.max(next_q)
            states.append(state)
            targets.append(target)
        #print(np.array(states))
        #print(np.array(targets))

        self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
