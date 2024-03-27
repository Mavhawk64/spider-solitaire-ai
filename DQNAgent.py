import os
import tensorflow as tf
import numpy as np
import random

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, discount_factor=0.95, exploration_rate=1.0, exploration_decay_rate=0.995, min_exploration_rate=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.min_exploration_rate = min_exploration_rate

        self.model = self.create_model()

    def create_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(tf.keras.layers.Dense(24, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))

        print(model.summary())
        return model

    def choose_action(self, state, valid_actions):
        if np.random.rand() < self.exploration_rate:
            return random.choice(valid_actions)  # Explore
        else:
            q_values = self.model.predict(state)
            # Choose the best action from q_values considering only valid actions
            return valid_actions[np.argmax(q_values[0][valid_actions])]
    
    def decay_exploration_rate(self):
        # Reduce the exploration rate, but not below the minimum exploration rate
        self.exploration_rate = max(self.exploration_rate * self.exploration_decay_rate, self.min_exploration_rate)
    
    def update_model(self, state, action, reward, next_state, done, save_path=None):
        target = reward
        if not done:
            target = reward + self.discount_factor * np.amax(self.model.predict(next_state)[0])
        target_f = self.model.predict(state)
        target_f[0][action] = target

        if save_path is not None:
            checkpoint_path = save_path + "/cp.weights.h5"
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=0)
            self.model.fit(state, target_f, epochs=11, verbose=0, callbacks=[cp_callback])
        else:
            self.model.fit(state, target_f, epochs=11, verbose=0)

        if self.exploration_rate > self.min_exploration_rate:
            self.exploration_rate *= self.exploration_decay_rate

    def save_model(self, filename):
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        self.model.save(filename)

    def load_model(self, filename):
        self.model = tf.keras.models.load_model(filename)
        self.model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
