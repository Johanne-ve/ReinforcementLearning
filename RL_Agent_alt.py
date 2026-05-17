

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
        self.memory = deque(maxlen=20000)  # größerer Buffer

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        
        # Target Network: Update-Counter
        self.update_target_counter = 0
        self.update_target_every = 10  # alle 100 Trainingsschritte synchronisieren

        if load != '':
            self.model = tf.keras.models.load_model(load)
            self.target_model = tf.keras.models.load_model(load)  # Target als Kopie
        else:
            self.model = self.build_model()
            self.target_model = self.build_model()  # identisches Netz
            self.update_target_model()  # initial synchronisieren

    def build_model(self):
        inputs = layers.Input(shape=(self.state_size,))
        x = layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(0.001))(inputs)
        x = layers.Dense(64, activation='relu',
                     kernel_regularizer=regularizers.l2(0.001))(x)        
        x = layers.Dense(32, activation='relu',
                     kernel_regularizer=regularizers.l2(0.001))(x)
        outputs = layers.Dense(self.action_size, activation='linear')(x)
    
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=self.learning_rate))
        return model  # WICHTIG: return statt self.model = model

    def update_target_model(self):
        """Kopiere Gewichte vom Haupt-Modell ins Target-Modell"""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            if np.random.rand() <= 0.001:
                return random.randrange(self.action_size)
            else:  #  Wähle aus Maske der sinnigen Actions
                usable_actions = [index for (index, item) in enumerate(state[0][-self.action_size:]) if item == 1]
                random.shuffle(usable_actions)
                return usable_actions[0]

        ### Vorfiltern sinniger Fahraufträge
        q_values = self.model.predict(state, verbose=0)
        filter_vec = [x*50 for x in state[0][-self.action_size:]]  # Original-Maske
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
                # KRITISCHE ÄNDERUNG: Verwende target_model für stabile Q-Targets
                next_q = self.target_model.predict(np.array([next_state]), verbose=0)[0]
                target[action] = reward + self.gamma * np.max(next_q)
            states.append(state)
            targets.append(target)

        self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)

        # Target Network periodisch updaten
        self.update_target_counter += 1
        if self.update_target_counter >= self.update_target_every:
            self.update_target_model()
            self.update_target_counter = 0
