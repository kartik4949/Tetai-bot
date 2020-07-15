from collections import deque
import random
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
import numpy as np
import skimage
from  skimage.color import rgb2gray
from  skimage.transform import resize
from skimage.exposure import rescale_intensity
from bot_model import tetai_model as model
from play import Play

GAME = 'Tetris'
CONFIG = 'nothreshold'
ACTIONS = 4
GAMMA = 0.95
OBSERVATION = 10000
EXPLORE = 3000000
FINAL_EPSILON = 0.0001
INITIAL_EPSILON = 0.1
REPLAY_MEMORY = 50000
BATCH = 32
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-4



class Train:
    def __init__(self):
        self.model = model()
        self.optimizer = tf.keras.optimizers.Adam(lr=LEARNING_RATE)

    @staticmethod
    def _preprocess_image(frame):
        frame = rgb2gray(frame)
        frame = resize(frame,(80,80))
        frame = rescale_intensity(frame,out_range=(0,255))
        frame = frame / 255.0
        return frame
    def loss(self , y , y_hat):
        return tf.keras.losses.mean_squared_error(y,y_hat)

    @tf.function
    def _train_step(self,inputs,targets):
        y_hat = self.model(inputs)
        loss = self.loss(targets , y_hat)
        return loss


    def __call__(self , mode = "Train"):
        game_state = Play()
        D = deque()
        do_nothing = 3
        frame, action, done = game_state.frame_step(do_nothing)
        frame = self.__class__._preprocess_image(frame)
        stack_frame = np.stack((frame, frame, frame, frame), axis=2)
        #In Keras, need to reshape
        stack_frame = stack_frame.reshape(1, stack_frame.shape[0], stack_frame.shape[1], stack_frame.shape[2])  #1*80*80*4
        if mode == 'Run':
            OBSERVE = 999999999
            epsilon = FINAL_EPSILON
            print ("Now we load weight")
            self.model.load_weights("model.h5")
            print ("Weight load successfully")
        else:                       #We go to training mode
            OBSERVE = OBSERVATION
            epsilon = INITIAL_EPSILON

        t = 0
        while True:
            loss = 0
            Q_sa = 0
            action_index = 0
            reward = 0
            a_t = ACTIONS
            #choose an action epsilon greedy
            if t % FRAME_PER_ACTION == 0:
                if random.random() <= epsilon:
                    print("----------Random Action----------")
                    action_index = random.randrange(ACTIONS)
                    a_t = action_index
                else:
                    q = self.model(stack_frame,training=False)
                    max_Q = np.argmax(q)
                    a_t=max_Q

            #We reduced the epsilon gradually
            if epsilon > FINAL_EPSILON and t > OBSERVE:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

            #run the selected action and observed next state and reward
            frame1_colored, reward, done = game_state.frame_step(a_t)

            #Restart game if game is over
            if done:
                game_state.restart()

            frame1 = self._preprocess_image(frame1_colored)
            frame1 = frame1.reshape(1, frame1.shape[0], frame1.shape[1], 1) #1x80x80x1
            stack_frame1 = np.append(frame1, stack_frame[:, :, :, :3], axis=3)

            # store the transition in D
            D.append((stack_frame, action_index, reward, stack_frame1, done))
            if len(D) > REPLAY_MEMORY:
                D.popleft()

            #only train if done observing
            if t > OBSERVE:
                #sample a minibatch to train on
                minibatch = random.sample(D, BATCH)

                #Now we do the experience replay
                state_t, action_t, reward_t, state_t1, done = zip(*minibatch)
                state_t = np.concatenate(state_t)
                state_t1 = np.concatenate(state_t1)
                targets = self.model(state_t , training=False)
                _targets = np.zeros_like(targets)
                Q_sa = self.model(state_t1, training=False)
                _targets[range(BATCH), action_t] = reward_t + GAMMA*np.max(Q_sa, axis=1)*np.invert(done)
                with tf.GradientTape() as tape:
                    _loss = self._train_step(state_t,_targets)
                    gradients = tape.gradient(_loss, self.model.trainable_variables)
                    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                    print("APPLIED GRADIENT")


            stack_frame = stack_frame1
            t = t + 1

            # save progress every 10000 iterations
            #BUG
            if t % 1000 == 0:
                print("Now we save model")
                self.model.save_weights("model.h5", overwrite=True)
                #with open("model.json", "w") as outfile:
                    #json.dump(self.model.to_json(), outfile)

            if t <= OBSERVE:
                state = "observe"
            elif t > OBSERVE and t <= OBSERVE + EXPLORE:
                state = "explore"
            else:
                state = "train"
            print('Done = ', done)

            print("TIMESTEP", t, "/ STATE", state, \
                "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", reward, \
                "/ Q_MAX " , np.max(Q_sa), "/ Loss ", loss)

        print("Episode finished!")
        print("************************")


if __name__ == "__main__":
    bot = Train()
    bot()
