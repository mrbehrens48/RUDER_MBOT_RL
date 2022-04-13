import tensorflow as tf
import threading
import sys
import cv2
from typing import Optional
from vimba import *
import multiprocessing as mp
import time
import numpy as np
import microrobot_environment as mbot
import os
from tqdm import tqdm
from tensorflow.keras import layers
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import tkinter as tk
from math import pi

tf.keras.backend.set_floatx('float32') #set the default data type in the keras model to float32

from myconfig import * #the myconfig file is essential for setting up experimental parameters, and should be updated each time the program is run. 

if CONVOLUTIONAL:
    STATE_SHAPE = (STATE_SIZE,STATE_SIZE,N_STEPS) #the size of the pictures going into the convNet, and the number of pictures
else:
    STATE_SHAPE = (3+NUM_ACTIONS,N_STEPS) 
    if VERBOSE: print(f'state_shape = {STATE_SHAPE}')

tensorboard_save_path = 'microrobot_tensorboard'
log_save_path = f'logs/{MODEL_NAME}_logs'
#model_save_path = 'models'
model_save_path = 'C:\\Users\Windows\OneDrive - University of Pittsburgh\Documents\GitHub\RUDER_MBOT_RL\models'
buffer_save_path = f'buffers/{MODEL_NAME}_buffers'

pi_model_name = f'pi_model_{MODEL_NAME}'
q1_model_name = f'q1_model_{MODEL_NAME}'
q2_model_name = f'q2_model_{MODEL_NAME}'

LOG_STD_MIN = -20
LOG_STD_MAX = 2

lower_bound = -1
upper_bound = 1

target_entropy = -NUM_ACTIONS

learner_device_ID = 0 #on the GPU
env_device_ID = 1 #on the GPU
master_device_ID = 0 #on the CPU

gamma = 0.99
tau = 0.005
########################################################################################################
order = 20
    
'''
mu_M_x_2c = np.load('logs/microrobot june 2 convolutional deterministic eval_logs/M_x_log.npy')
mu_M_y_2c = np.load('logs/microrobot june 2 convolutional deterministic eval_logs/M_y_log.npy')
mu_phi_x_2c = np.load('logs/microrobot june 2 convolutional deterministic eval_logs/phi_x_log.npy')
mu_phi_y_2c = np.load('logs/microrobot june 2 convolutional deterministic eval_logs/phi_y_log.npy')
mu_theta_2c = np.load('logs/microrobot june 2 convolutional deterministic eval_logs/theta_log.npy')
mu_theta_dot_2c = np.load('logs/microrobot june 2 convolutional deterministic eval_logs/theta_dot_log.npy')

mu_M_x_2c = mu_M_x_2c[1:3001]
mu_M_y_2c = mu_M_y_2c[1:3001]
mu_phi_x_2c = mu_phi_x_2c[1:3001]
mu_phi_y_2c = mu_phi_y_2c[1:3001]
mu_theta_2c = mu_theta_2c[1:3001]
mu_theta_dot_2c = mu_theta_dot_2c[1:3001]

x = mu_theta_2c
y = mu_phi_x_2c
c_var = mu_theta_dot_2c
x = x[c_var > 3]
y = y[c_var > 3]
c_var = c_var[c_var > 3]
phi_x_model = np.poly1d(np.polyfit(x, y, order))

x = mu_theta_2c
y = mu_phi_y_2c
c_var = mu_theta_dot_2c
x = x[c_var > 3]
y = y[c_var > 3]
c_var = c_var[c_var > 3]
phi_y_model = np.poly1d(np.polyfit(x, y, order))

x = mu_theta_2c
y = mu_M_x_2c
c_var = mu_theta_dot_2c
x = x[c_var > 3]
y = y[c_var > 3]
c_var = c_var[c_var > 3]
M_x_model = np.poly1d(np.polyfit(x, y, order))

x = mu_theta_2c
y = mu_M_y_2c
c_var = mu_theta_dot_2c
x = x[c_var > 3]
y = y[c_var > 3]
c_var = c_var[c_var > 3]
M_y_model = np.poly1d(np.polyfit(x, y, order))
###############################################################################################
'''
#log_alpha = tf.Variable(0.0, trainable = True)

if not os.path.isdir(model_save_path):
    os.makedirs(model_save_path)
    
if not os.path.isdir(log_save_path):
    os.makedirs(log_save_path)
    
if not os.path.isdir(buffer_save_path):
    os.makedirs(buffer_save_path)
    
if not os.path.isdir(tensorboard_save_path):
    os.makedirs(tensorboard_save_path)
    
def run_updates():
    global alpha
    print(f'updating alpha from {alpha} to {alpha*2}')
    if alpha == 0:
        alpha = 0.001
    else:
        alpha *= 10
    print(f'alpha is now {alpha}')
    
class GUI(tk.Frame):
    def __init__(self, master, pause_q, abort_q, env_metrics_q, training_logs_q):
        super().__init__(master)
        self.master = master
        self.pause_q = pause_q
        self.abort_q = abort_q
        self.env_metrics_q = env_metrics_q
        self.training_logs_q = training_logs_q
        self.pack()
        self.create_widgets()
        

    def create_widgets(self):
        self.pause_button = tk.Button(self)
        self.pause_button["text"] = "Pause"
        self.pause_button["command"] = self.pause
        self.pause_button.pack(side="top")
        
        self.quit = tk.Button(self, text="QUIT", fg="red",
                              command=self.master.destroy)
        self.quit.pack(side="bottom")

    def pause(self):
        print("Pausing the environment, please wait for the current episode to finish")
        self.pause_q.put(1)

def pauser(pause_q, abort_q, env_metrics_q, training_logs_q):

    root = tk.Tk()
    app = GUI(root, pause_q, abort_q, env_metrics_q, training_logs_q)
    app.mainloop()
    
# a tensorboard modified to work with reinforcement learning. from stentdex. 
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = os.path.join(self.log_dir, MODEL_NAME)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)
    
    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()
 


#@tf.function
def update(state_batch, action_batch, reward_batch, next_state_batch, done_batch, pi_model, q1_model, q2_model, target_q1, target_q2):
    
    global alpha
    critic_optimizer = tf.keras.optimizers.Adam(CRITIC_LR)
    actor_optimizer = tf.keras.optimizers.Adam(ACTOR_LR)
    

    _,action_next,logp_action_next = policy(next_state_batch, pi_model)

    q1_target_value = target_q1([next_state_batch, action_next])

    q2_target_value = target_q2([next_state_batch, action_next])

    minimum_q_target = tf.minimum(q1_target_value, q2_target_value)

    #print(f'in update, about to calculate y with alpha = {alpha}')
    y = reward_batch + gamma * (1 - done_batch) * (minimum_q_target - alpha * logp_action_next)
    #print(f'y calculated as : {y}')

    #update the critics
    with tf.GradientTape() as tape:
        
        q1_value = q1_model([state_batch, action_batch], training=True)

        q1_loss = tf.math.reduce_mean(tf.math.square(q1_value - y))

    q1_grad = tape.gradient(q1_loss, q1_model.trainable_variables)
    critic_optimizer.apply_gradients(
        zip(q1_grad, q1_model.trainable_variables)
    )
    
    with tf.GradientTape() as tape:
        
        q2_value = q2_model([state_batch, action_batch], training=True)

        q2_loss = tf.math.reduce_mean(tf.math.square(q2_value - y))

    q2_grad = tape.gradient(q2_loss, q2_model.trainable_variables)
    critic_optimizer.apply_gradients(
        zip(q2_grad, q2_model.trainable_variables)
    )
    
    #update the actor
    with tf.GradientTape() as tape:
        
        _,action,logp_action = policy(state_batch, pi_model)
        q1_value = q1_model([state_batch, action], training = True)
        q2_value = q2_model([state_batch, action], training = True)
        minimum_q = tf.minimum(q1_value, q2_value)

        pi_loss = -tf.math.reduce_mean(minimum_q - alpha * logp_action)

    pi_grad = tape.gradient(pi_loss, pi_model.trainable_variables)
    actor_optimizer.apply_gradients(
        zip(pi_grad, pi_model.trainable_variables)
    )
    
    return q1_loss, q2_loss, pi_loss

#@tf.function
def gaussian_liklihood(x,mu,log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+1e-8))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis = 1) 


#@tf.function
def policy(state, pi_model):
    mu_actions = pi_model(state, training = True)  
    log_std = pi_model(state, training = True) 
    
    log_std = tf.clip_by_value(log_std, LOG_STD_MIN,LOG_STD_MAX)
    std = tf.exp(log_std)

    pi_actions = mu_actions + tf.random.normal(tf.shape(mu_actions))* std
    logp_pi_actions = gaussian_liklihood(pi_actions,mu_actions,log_std)

    mu, pi, logp_pi = apply_squashing_function(mu_actions, pi_actions, logp_pi_actions)
    
    return mu, pi, logp_pi

#@tf.function
def apply_squashing_function(mu,pi,logp_pi):
    logp_pi -= tf.reduce_sum(2*(np.log(2) - pi - tf.nn.softplus(-2*pi)), axis = 1)

    mu = tf.tanh(mu)
    pi = tf.tanh(pi)
    
    legal_mu = tf.clip_by_value(mu, lower_bound, upper_bound)
    legal_pi = tf.clip_by_value(pi, lower_bound, upper_bound)
        
    return legal_mu, legal_pi, logp_pi

def get_conv_actor(state_shape, NUM_ACTIONS):
    print(f'generating a CNN with state shape {state_shape} and {NUM_ACTIONS} actions')
    last_init = tf.random_uniform_initializer(minval=-0.0003, maxval=0.0003)
    inputs = layers.Input(shape=(state_shape), dtype = tf.float32)
    
    out = layers.Conv2D(16, (3,3), activation = 'relu')(inputs)
    out = layers.MaxPooling2D(2,2)(out)
    out = layers.Conv2D(32, (3,3), activation = 'relu')(out)
    out = layers.MaxPooling2D(2,2)(out)
    out = layers.Conv2D(64, (3,3), activation = 'relu')(out)
    out = layers.MaxPooling2D(2,2)(out)
    out = layers.Flatten()(out)
            
    out = layers.Dense(64, activation="relu")(out)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(NUM_ACTIONS, activation=None, kernel_initializer=last_init)(out)

    model = tf.keras.Model(inputs, outputs)
    print('finished making the CNN')
    return model


def get_conv_critic(state_shape, NUM_ACTIONS):

    state_inputs = layers.Input(shape=(state_shape), dtype = tf.float32)
    state_out = layers.Conv2D(16, (3,3), activation = 'relu')(state_inputs)
    state_out = layers.MaxPooling2D(2,2)(state_out)
    state_out = layers.Conv2D(32, (3,3), activation = 'relu')(state_out)
    state_out = layers.MaxPooling2D(2,2)(state_out)
    state_out = layers.Conv2D(64, (3,3), activation = 'relu')(state_out)
    state_out = layers.MaxPooling2D(2,2)(state_out)
    state_out = layers.Flatten()(state_out)
    state_out = layers.Dense(64, activation = 'relu')(state_out)
    
    action_inputs = layers.Input(shape=(NUM_ACTIONS))
    action_out = layers.Dense(16, activation="relu")(action_inputs)

    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1, activation=None)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_inputs, action_inputs], outputs)
    return model

def get_state_actor(state_shape, NUM_ACTIONS):

    last_init = tf.random_uniform_initializer(minval=-0.0003, maxval=0.0003)
    inputs = layers.Input(shape=(state_shape), dtype = tf.float32)
            
    out = layers.Flatten()(inputs)
    out = layers.Dense(256, activation="relu")(out)
    out = layers.Dropout(0.2)(out)
    out = layers.Dense(256, activation="relu")(out)
    out = layers.Dropout(0.2)(out)
    outputs = layers.Dense(NUM_ACTIONS, activation=None, kernel_initializer=last_init)(out)

    model = tf.keras.Model(inputs, outputs)
    return model


def get_state_critic(state_shape, NUM_ACTIONS):

    if VERBOSE: print('making a q model')
    state_input = layers.Input(shape=(state_shape), dtype = tf.float32)
    state_out = layers.Flatten()(state_input)
    state_out = layers.Dense(16, activation = 'relu')(state_out)
    
    if VERBOSE: print(f'made a state input with shape {state_shape}')
    action_input = layers.Input(shape=(NUM_ACTIONS))
    action_out = layers.Dense(16, activation="relu")(action_input)
    if VERBOSE: print(f'made an action input with shape {NUM_ACTIONS}')
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dropout(0.2)(out)
    out = layers.Dense(256, activation="relu")(out)
    out = layers.Dropout(0.2)(out)
    outputs = layers.Dense(1, activation=None)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model


def learn(state_batch, action_batch, reward_batch, next_state_batch, done_batch, pi_model, q1_model, q2_model, target_q1, target_q2, log_alpha, minibatches):
  
    #print('learn function called')
    q1_loss, q2_loss, pi_loss = update(state_batch, action_batch, reward_batch, next_state_batch, done_batch, pi_model, q1_model, q2_model, target_q1, target_q2)
    #print('update function called')
    alpha_loss, log_alpha, entropy = update_alpha(state_batch, pi_model, log_alpha)
    if UPDATE_ALPHA:
        #print('updating alpha')
        global alpha 
        alpha = tf.math.exp(log_alpha)
        #print(f'alpha = {alpha}')

    update_target(target_q1.variables, q1_model.variables, tau)
    update_target(target_q2.variables, q2_model.variables, tau)
    
    #print('targets updated')
    
    return q1_loss, q2_loss, pi_loss, alpha_loss, entropy#, alpha_loss, alpha

#@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))
        
#@tf.function(experimental_relax_shapes=True)
def update_alpha(state_batch, pi_model, log_alpha):
    
    alpha_optimizer = tf.optimizers.Adam(ALPHA_LR)
            
    #print(f'updating alpha, log_alpha is: {log_alpha}')
    _,action,logp_action = policy(state_batch, pi_model)

    with tf.GradientTape() as tape:
        #print('in the gradient tape')
        alpha_losses = -(log_alpha * tf.stop_gradient(logp_action + target_entropy))
        #print('calculated alpha losses')

        alpha_loss = tf.nn.compute_average_loss(alpha_losses)
        #print(f'alpha_loss is {alpha_loss}')

    alpha_gradients = tape.gradient(alpha_loss, [log_alpha])
    #print(f'alpha_gradients is {alpha_gradients}')
    alpha_optimizer.apply_gradients(zip(alpha_gradients, [log_alpha]))
    #print(f'done updating alpha, log_alpha is now: {log_alpha}')
    return alpha_loss, log_alpha, -np.mean(logp_action)    
    
class Buffer:
    def __init__(self, buffer_capacity=1_000_000, batch_size=64, state_shape = (2,), num_actions = 3):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element 
        if LOAD_BUFFER == 1:
            print('loading a buffer')
            self.state_buffer = np.load(f'{buffer_load_path}/state_buffer.npy')
            self.next_state_buffer = np.load(f'{buffer_load_path}/next_state_buffer.npy')
            self.reward_buffer = np.load(f'{buffer_load_path}/reward_buffer.npy')
            self.action_buffer = np.load(f'{buffer_load_path}/action_buffer.npy')
            self.done_buffer = np.load(f'{buffer_load_path}/done_buffer.npy')
            self.buffer_counter = np.load(f'{buffer_load_path}/buffer_counter.npy')
            #self.buffer_counter = 200_000
            print(f'the buffer counter of the loaded buffer is {self.buffer_counter}')
        else:
            self.buffer_counter = 0
            self.state_buffer = np.zeros((self.buffer_capacity,) + state_shape)
            self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
            self.reward_buffer = np.zeros((self.buffer_capacity, 1))
            self.done_buffer = np.zeros((self.buffer_capacity, 1))
            self.next_state_buffer = np.zeros((self.buffer_capacity,) + state_shape)

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        self.done_buffer[index] = obs_tuple[4]

        self.buffer_counter += 1
        print(f"buffer counter: {self.buffer_counter}")
            
    def get_minibatch(self):
        record_range = min(self.buffer_counter, self.buffer_capacity)
        minibatch_indices = np.random.choice(record_range,self.batch_size)
        
        state_batch = self.state_buffer[minibatch_indices]
        action_batch = self.action_buffer[minibatch_indices]
        reward_batch = self.reward_buffer[minibatch_indices]
        next_state_batch = self.next_state_buffer[minibatch_indices]
        done_batch = self.done_buffer[minibatch_indices]
        #print(f'done batch is of type {type(done_batch)} and of shape {np.shape(done_batch)}')
        if VERBOSE: print(f'here is the done batch: {done_batch}')
        
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch


def print_preamble():
    print('///////////////////////////////////////////////////////')
    print('/// Vimba API Asynchronous Grab with OpenCV Example ///')
    print('///////////////////////////////////////////////////////\n')


def print_usage():
    print('Usage:')
    print('    python asynchronous_grab_opencv.py [camera_id]')
    print('    python asynchronous_grab_opencv.py [/h] [-h]')
    print()
    print('Parameters:')
    print('    camera_id   ID of the camera to use (using first camera if not specified)')
    print()



def abort(reason: str, return_code: int = 1, usage: bool = False):
    print(reason + '\n')

    if usage:
        print_usage()

    sys.exit(return_code)


def parse_args() -> Optional[str]:
    args = sys.argv[1:]
    argc = len(args)

    for arg in args:
        if arg in ('/h', '-h'):
            print_usage()
            sys.exit(0)

    if argc > 1:
        abort(reason="Invalid number of arguments. Abort.", return_code=2, usage=True)

    return None if argc == 0 else args[0]


def get_camera(camera_id: Optional[str]) -> Camera:
    with Vimba.get_instance() as vimba:
        if camera_id:
            try:
                return vimba.get_camera_by_id(camera_id)

            except VimbaCameraError:
                abort('Failed to access Camera \'{}\'. Abort.'.format(camera_id))

        else:
            cams = vimba.get_all_cameras()
            if not cams:
                abort('No Cameras accessible. Abort.')

            return cams[0]


def setup_camera(cam: Camera):
    with cam:
        # Enable auto exposure time setting if camera supports it
        '''
        try:
            cam.ExposureAuto.set('Continuous')

        except (AttributeError, VimbaFeatureError):
            pass
        '''
        # Enable white balancing if camera supports it
        try:
            cam.BalanceWhiteAuto.set('Continuous')

        except (AttributeError, VimbaFeatureError):
            pass

        # Try to adjust GeV packet size. This Feature is only available for GigE - Cameras.
        try:
            cam.GVSPAdjustPacketSize.run()

            while not cam.GVSPAdjustPacketSize.is_done():
                pass

        except (AttributeError, VimbaFeatureError):
            pass

        # Query available, open_cv compatible pixel formats
        # prefer color formats over monochrome formats
        
        cv_fmts = intersect_pixel_formats(cam.get_pixel_formats(), OPENCV_PIXEL_FORMATS)
        color_fmts = intersect_pixel_formats(cv_fmts, COLOR_PIXEL_FORMATS)

        if color_fmts:
            cam.set_pixel_format(color_fmts[0])

        else:
            mono_fmts = intersect_pixel_formats(cv_fmts, MONO_PIXEL_FORMATS)

            if mono_fmts:
                cam.set_pixel_format(mono_fmts[0])

            else:
                abort('Camera does not support a OpenCV compatible format natively. Abort.')
                

class Handler:
    def __init__(self, frame_q, frame_request_q):
        self.image = 0
        self.frame_q = frame_q
        self.frame_request_q = frame_request_q
        self.shutdown_event = threading.Event()

    def __call__(self, cam: Camera, frame: Frame):

        if frame.get_status() == FrameStatus.Complete:
            capture_time = time.time()
            #print('{} acquired {}'.format(cam, capture_time), flush=True)

            im = frame.as_opencv_image()                  # Read image
        
            scale_percent = 40 # percent of original size
            width = int(im.shape[1] * scale_percent / 100)
            #print('width = {}'.format(t))
            height = int(im.shape[0] * scale_percent / 100)
            dim = (width, height)
            imS = cv2.resize(im, dim)
            crop_img = imS[int(height*0.2):int(height*0.8), int(width*0.28):int(width*0.72)]
            dim = (FRAME_SIZE, FRAME_SIZE)
            origin_dim = (ORIGINAL_FRAME_SIZE, ORIGINAL_FRAME_SIZE)
            original_image = cv2.resize(crop_img, origin_dim)
            image = cv2.resize(crop_img, dim)
            #self.image = image
            
            if not self.frame_request_q.empty():
                self.frame_request_q.get()
                #print('sending frame')
                self.frame_q.put((image, capture_time))
            cv2.imshow('the original', original_image)
            cv2.waitKey(1)
        cam.queue_frame(frame)
        
def square_policy(theta):
    if theta < 180:
        M_x = 0
    else:
        M_x = 0.75
    if theta < 90 or theta > 270:
        phi_y = 5
    else:
        phi_y = 0.8
    phi_y = (phi_y*2 / (2*pi))-1
    M_y = -1
    phi_x = 1
    phi_x = (phi_x*2 / (2*pi))-1
    return ((M_x,M_y,phi_x,phi_y))

def sin_policy(theta):
    M_x = np.sin(theta*pi/180+pi)
    phi_y = 2.3*np.sin(theta*pi/180-pi/2)+pi
    #OldRange = (OldMax - OldMin)  = 2pi - 0
    #NewRange = (NewMax - NewMin)  1--1 = 2
    #NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
    phi_y = (phi_y*2 / (2*pi))-1
    M_y = -1
    phi_x = 1
    phi_x = (phi_x*2 / (2*pi))-1
    return ((M_x,M_y,phi_x,phi_y))    
    
def mix_policy(theta):
    if theta < 180:
        M_x = -0.77
    else:
        M_x = 0.71
        
    phi_y = 2*np.sin(theta*pi/180-pi/2)+2.58
    #OldRange = (OldMax - OldMin)  = 2pi - 0
    #NewRange = (NewMax - NewMin)  1--1 = 2
    #NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
    phi_y = (phi_y*2 / (2*pi))-1
    M_y = -0.96
    phi_x = 0.96
    phi_x = (phi_x*2 / (2*pi))-1
    return ((M_x,M_y,phi_x,phi_y)) 

    

def poly_policy(theta):
    phi_x = phi_x_model(theta)
    phi_y = phi_y_model(theta)
    M_x = M_x_model(theta)
    M_y = M_y_model(theta)
    
    phi_y = (phi_y*2 / (2*pi))-1
    phi_x = (phi_x*2 / (2*pi))-1
    return ((M_x,M_y,phi_x,phi_y)) 

def camera_process(frame_q, frame_request_q, abort_q):
    try:
        print('camera process id:', os.getpid())
        cam_id = parse_args()
    
        with Vimba.get_instance():
            with get_camera(cam_id) as cam:
    
                setup_camera(cam)
                handler = Handler(frame_q, frame_request_q) #does all the things to the frame, and returns it
    
                try:
                    cam.start_streaming(handler=handler, buffer_count=10)
                    handler.shutdown_event.wait()
    
                finally:
                    cam.stop_streaming()
    except:
        abort_q.put(1)
        print('camera process raised an error')
        
def environment_process(frame_q, frame_request_q, quit_q, begin_q, env_device_ID, env_metrics_q, observation_q, new_pi_q, pause_q, abort_q, reset_q):
    tf.random.set_seed(time.time())
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    
    with tf.device('/device:cpu:0'):    
        try:
            while quit_q.empty() and abort_q.empty():
         
                print('beginning environment')
                print(' environment process id:', os.getpid())
                if not SIMULATION:
                    env = mbot.Microrobot_Env(frame_q, frame_request_q, STATE_SIZE, N_STEPS, VERBOSE, CONVOLUTIONAL, NUM_ACTIONS, log_save_path)
                    print('successfully loaded the robot environment')
                else:
                    env = mbot.Microrobot_Sim(STATE_SIZE, N_STEPS, VERBOSE, CONVOLUTIONAL, NUM_ACTIONS)
                    print('successfully loaded the simulation environment')
                frames = 0
                abort_count = 0
                load_pi = LOAD_PI #convert the global variable to a mutable local. this is necessary due to multiprocessing
                episodes_this_epoch = 0
                episode = 0
                fresh_model = 1
                '''
                print('you are here')
                pi_model = tf.keras.models.load_model(f'{model_save_path}\{load_pi_model_name}.h5')
                Q1_model = tf.keras.models.load_model(f'{model_save_path}\{load_q1_model_name}.h5')
                print('now here')
                
                pi_model.summary()
                Q1_model.summary()
                '''
    
                print('environment waiting for start signal')
                while begin_q.empty() and abort_q.empty():
                    pass #wait for the signal from the master process to begin training
                print('environment starting')
                        
                start_time = time.time()
                while quit_q.empty() and abort_q.empty(): #train until we get the signal to quit
                    '''
                    if time.time() - start_time > ARDUINO_RESET_TIME:
                        print('starting arduino reset')
                        start_time = time.time()
                        env.reset_arduino()
                        print('\n\n\nsuccessfully reset the robot environment\n\n\n')
                    '''
                    if not reset_q.empty():
                        reset_q.get()
                        print('environment resetting')
                        break
                    if pause_q.empty(): #if we are not paused, do an episode
                        
                        #Load a new pi model each epoch
                        if VERBOSE: print('Environment running an episode')      
                        if load_pi == 1:
                            pi_model = tf.keras.models.load_model(f'{model_save_path}\{load_pi_model_name}.h5')
                            print('loaded from the Load pi model')
                            load_pi = 0
                        elif not new_pi_q.empty():
                            try:
                                try: 
                                    new_pi_q.get()
                                    
                                except:
                                    pass
                                pi_model = tf.keras.models.load_model(f'{model_save_path}\{pi_model_name}.h5')
                                print('loaded from the new pi queue')
                                load_pi = 0 #after the first time, we will only load if there is a new model in the q
                                episodes_this_epoch = 0
                            except:
                                print('failed to load a new pi model')
                        
                        elif fresh_model == 1:
                            if CONVOLUTIONAL:
                                print('trying to create a new CNN')
                                pi_model = get_conv_actor(STATE_SHAPE, NUM_ACTIONS)
                                print('loaded a fresh convolutional pi model')
                                fresh_model = 0
                                episodes_this_epoch = 0
                            else:
                                print('trying to create a new NN')
                                pi_model = get_state_actor(STATE_SHAPE, NUM_ACTIONS)
                                print('loaded a fresh state pi model')
                                fresh_model = 0
                                episodes_this_epoch = 0
                        else:
                            print('keeping the same pi_model')
                            episodes_this_epoch += 1
                            
                        
                        pi_model.summary()
                        if episode == 0:
                            action = np.random.uniform(-1,1,NUM_ACTIONS)
                        elif SQUARE_POLICY == 1:
                            action = square_policy(theta)
                            print(f'square policy action = {action}')
                        elif SIN_POLICY == 1:
                            action = sin_policy(theta)
                            print(f'sin policy action = {action}')
                        elif MIX_POLICY == 1:
                            action = mix_policy(theta)
                            print(f'mix policy action = {action}')
                        elif POLY_POLICY == 1:
                            action = poly_policy(theta)
                            print(f'poly policy action = {action}')
                        else:
                            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
                            tf_prev_state = tf.cast(tf_prev_state, tf.float32)
                            _,action,_ = policy(tf_prev_state, pi_model)
                            action = tf.squeeze(action)
                            
                        print(f'\n\n  EPISODE **** {episode} ****\n\n')
                        
                        
                        #print('environment doing a reset')
                        prev_state, reward, done, duration, success, abort, theta_dot, theta, net_theta = env.reset(action)
                        #print(f'state shape is: {np.shape(prev_state)}')
                        ep_reward = 0
                        ep_success = 0
                        ep_theta_dot = 0
                        M_x_log = []
                        M_y_log = []
                        M_z_log = []
                        freq_log = []
                        phi_x_log = []
                        phi_y_log = []
                        ep_duration = 0
                        if episode % EVALUATION == 0 and episode > 0:
                            evaluation = 1
                        else:
                            evaluation = 0
                        while done == 0 and quit_q.empty() and abort_q.empty(): #goes until the environment sends the done signal
                            if not pause_q.empty():
                                print('paused training!!!')
                                pause_q.get()
                                env.pause()
                                while pause_q.empty():
                                    pass
                                pause_q.get()
                            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
                            tf_prev_state = tf.cast(tf_prev_state, tf.float32)
                               
                            if SQUARE_POLICY == 1:
                                action = square_policy(theta)
                                print(f'square policy action = {action}')
                            elif SIN_POLICY == 1:
                                action = sin_policy(theta)
                                print(f'sin policy action = {action}')
                            elif MIX_POLICY == 1:
                                action = mix_policy(theta)
                                print(f'mix policy action = {action}')
                            elif POLY_POLICY == 1:
                                action = poly_policy(theta)
                                print(f'poly policy action = {action}')
                            elif frames >= IMPLEMENT_POLICY_AFTER:
                                if evaluation == 1 or CONTINUOUS_EVALUATION == 1:
                                    print('evaluation episode')
                                    action,_,_ = policy(tf_prev_state, pi_model)
                                else:
                                    _,action,_ = policy(tf_prev_state, pi_model)
                                action = tf.squeeze(action)
                            
                            else:
                                #action = (1,-1,1,1)
                                action = np.random.uniform(-1,1,NUM_ACTIONS)
                                                          
                            
                            try:
                                print('starting a step')
                                state, reward, done, duration, success, abort, theta_dot, theta, net_theta = env.step(action)
                                print('successful step')
                            except:
                                print('step failed')
                                abort = 1
                            
                            M_x = action[0]
                            M_y = action[1]
                            if tf.is_tensor(M_x):
                                M_x = float(M_x.numpy())
                            if tf.is_tensor(M_y):
                                M_y = float(M_y.numpy())
                            M_x_log.append(M_x)
                            M_y_log.append(M_y)
                            
                            M_z = max(abs(M_x), abs(M_y))
                            if tf.is_tensor(M_z):
                                M_z = float(M_z.numpy())
                            M_z_log.append(M_z)
                            
                            freq = 100
                            if tf.is_tensor(freq):
                                freq = float(freq.numpy())
                            freq_log.append(freq)
                            
                            phi_x = action[2]
                            phi_y = action[3]
                            if tf.is_tensor(phi_x):
                                phi_x = float(phi_x.numpy())
                            if tf.is_tensor(phi_y):
                                phi_y = float(phi_y.numpy())
                            phi_x_log.append(phi_x)
                            phi_y_log.append(phi_y)
                            

                            #print(f'state shape is: {np.shape(state)}')
                            
                            if abort:
                                print(f'abort at frame {frames}')
                                #this means the the camera did not find exactly 2 objects, and so the input image was corrupted. 
                                abort_count += 1
                                break
                            
                            ep_reward += reward
                            
                            if success == 1:
                                ep_success = 1
                            #state = np.squeeze(state)
                            #prev_state = np.squeeze(prev_state)
                            print(f'the shape of state is {np.shape(state)} and the shape of prev_state is {np.shape(prev_state)}')
                            if VERBOSE: print(f'done is {done}')
                            
                            memory = (prev_state, action, reward, state, done)
                            print(f'here are all the types {type(prev_state)}, {type(action)}, {type(reward)},{type(state)}')
                            observation_q.put(memory)
                
                            frames += 1
                            ep_theta_dot += theta_dot
                
                            prev_state = state
                            
                            
                        print('episode complete')    
                        ep_duration = duration
                        try:
                            average_M_x = np.mean(M_x_log)
                            max_M_x = max(M_x_log)
                            min_M_x = min(M_x_log)
                            std_M_x = np.std(M_x_log)
                            
                            average_M_y = np.mean(M_y_log)
                            max_M_y = max(M_y_log)
                            min_M_y = min(M_y_log)
                            std_M_y = np.std(M_y_log)
                            
                            average_M_z = np.mean(M_z_log)
                            min_M_z = min(M_z_log)
                            max_M_z = min(M_z_log)
                            std_M_z = np.std(M_z_log)
                            
                            average_freq = np.mean(freq_log)
                            min_freq = min(freq_log)
                            max_freq = min(freq_log)
                            std_freq = np.std(freq_log)
                        
                            average_phi_x = np.mean(phi_x_log)
                            max_phi_x = max(phi_x_log)
                            min_phi_x = min(phi_x_log)
                            std_phi_x = np.std(phi_x_log)
                            
                            average_phi_y = np.mean(phi_y_log)
                            max_phi_y = max(phi_y_log)
                            min_phi_y = min(phi_y_log)
                            std_phi_y = np.std(phi_y_log)
                            print('calulcated all the action logs')
                            
                        except:
                            print('failed to create action logs')
                            average_M_x = 0
                            max_M_x = 0
                            min_M_x = 0
                            std_M_x = 0
                            
                            average_M_y = 0
                            max_M_y = 0
                            min_M_y = 0
                            std_M_y = 0
                            
                            max_M_z = 0
                            min_M_z = 0
                            average_M_z = 0
                            std_M_z = 0
                            
                            min_freq = 0
                            max_freq = 0
                            average_freq = 0
                            std_freq = 0
                            
                            average_phi_x = 0
                            max_phi_x = 0
                            min_phi_x = 0
                            std_phi_x = 0
                            
                            average_phi_y = 0
                            max_phi_y = 0
                            min_phi_y = 0
                            std_phi_y = 0
                            
                            
                        print('made all the logs to send over to the master')
                        episode_metrics = (ep_reward, ep_success, frames, episode, ep_theta_dot, theta, average_M_x, average_M_y, max_M_x, max_M_y, min_M_x, min_M_y, ep_duration, evaluation, average_phi_x, average_phi_y, max_phi_x, max_phi_y, min_phi_x, min_phi_y, average_M_z, max_M_z, min_M_z, average_freq, max_freq, min_freq, std_M_x, std_M_y, std_M_z, std_freq, std_phi_x, std_phi_y, net_theta)
                        env_metrics_q.put(episode_metrics) #for the tensorboard
                        episode += 1
                    else:
                        print('paused training!!!')
                        pause_q.get()
                        env.pause()
                        while pause_q.empty():
                            pass
                        pause_q.get()
        except:
            abort_q.put(1)
            print('environment process raised an error')
            
    
    
def master_process(device_ID, begin_q, observation_q, env_metrics_q, training_logs_q, minibatch_q, done_q, pause_q, abort_q, env_reset_q, learner_reset_q, model_save_q):
    tf.random.set_seed(time.time())
    try:
        with tf.device('/device:cpu:0'):  
            start_time = time.time()
            print('starting master process')
            print('master process id:', os.getpid())
            
            loops = 0
            
            while abort_q.empty():
                loops += 1
                while loops > 1:
                    pass
                #set up the tensorboard log save location    
                #tensorboard = ModifiedTensorBoard(log_dir = '{}\logs\{}_{}'.format(tensorboard_save_path,MODEL_NAME, int(time.time()))) 
                tensorboard = ModifiedTensorBoard(log_dir = '{}\logs\{}_{}'.format(tensorboard_save_path,MODEL_NAME, loops))
                evaluation_tensorboard = ModifiedTensorBoard(log_dir = '{}\logs\{}_{}_eval'.format(tensorboard_save_path,MODEL_NAME, loops))
                   
                buffer = Buffer(buffer_capacity = BUFFER_CAPACITY,
                                           batch_size = BATCH_SIZE,
                                           state_shape = STATE_SHAPE,
                                           num_actions = NUM_ACTIONS)
                    
        
                #all metrics which I want to average over
                ep_rewards = []
                ep_successes = []
                ep_theta_dot = []
                theta_log = []
                average_M_x_log = []
                average_M_y_log = []
                max_M_x_log = []
                max_M_y_log = []
                min_M_x_log = []
                min_M_y_log = []
                ep_duration_log = []
                max_phi_y_log = []
                min_phi_x_log = []
                min_phi_y_log = []
                max_phi_x_log = []
                average_phi_x_log = []
                average_phi_y_log = []
                
                average_M_z_log = []
                max_M_z_log = []
                min_M_z_log = []
                average_freq_log = []
                max_freq_log = []
                min_freq_log = []
                std_M_x_log = []
                std_M_y_log = []
                std_M_z_log = []
                std_freq_log = []
                std_phi_x_log = []
                std_phi_y_log = []
    
        
                records = 0
                minibatches = 0
                frames = 0
                q_loss = 0
                pi_loss = 0
                alpha = 0
                episode = 0
                entropy = 0
                alpha_loss = 0
                evaluation = 0
                top_score = 0
                top_success = 0
                long_average_score = 0
                net_theta = 0
                
                print(f'master poised and ready for loop {loops}')
                time.sleep(5)
                print('master says go!')
                begin_q.put(1)
                time.sleep(0.1)
                begin_q.get()
                
                update_time = time.time()
                buffer_save_time = time.time()
                while time.time()-start_time < TRAINING_TIME and abort_q.empty() and frames <= MAX_FRAMES: #do forever
                
                    '''TODO: implement check for user input and pause, which will pause the environemnt process. this will allow for manual resets'''
                    #pause_q.put(1)
                    if minibatches % RESET == 0 and minibatches > 1 and PARAMETER_SEARCH:
                        print('resetting')
                        run_updates()
                        env_reset_q.put(1)
                        learner_reset_q.put(1)
                        break
                    
                    #check all of the queues for incoming data
                    if not observation_q.empty():
                        buffer.record(observation_q.get())
                        if buffer.buffer_counter % BUFFER_SAVE == 0:
                            print('___________________________saving buffers_______________________')
                            print(f'trying to save state buffer which is of shape {np.shape(buffer.state_buffer)}')
                            np.save(f'{buffer_save_path}/state_buffer',buffer.state_buffer)
                            print('successfully saved state buffer')
                            np.save(f'{buffer_save_path}/action_buffer',buffer.action_buffer)
                            np.save(f'{buffer_save_path}/next_state_buffer',buffer.next_state_buffer)
                            np.save(f'{buffer_save_path}/reward_buffer',buffer.reward_buffer)
                            np.save(f'{buffer_save_path}/done_buffer',buffer.done_buffer)
                            np.save(f'{buffer_save_path}/buffer_counter',buffer.buffer_counter)
                            print('___________________________done saving buffers_______________________')
                        
                        #print(f'records = {records}')
                    if not env_metrics_q.empty():
                        records += 1
                        #print('getting the env_metrics')
                        env_metrics = env_metrics_q.get()
                        print(f'here is the size of env metrics: {np.shape(env_metrics)}')
                        print(f'here are all the env metrics: {env_metrics}')
                        ep_rewards.append(env_metrics[0])
                        ep_successes.append(env_metrics[1])
                        frames = env_metrics[2]
                        episode = env_metrics[3]
                        ep_theta_dot.append(env_metrics[4])
                        theta = env_metrics[5]
                        theta_log.append(theta)
                        average_M_x = env_metrics[6]
                        average_M_x_log.append(average_M_x)
                        average_M_y = env_metrics[7]
                        average_M_y_log.append(average_M_y)
                        max_M_x = env_metrics[8]
                        max_M_x_log.append(max_M_x)
                        max_M_y = env_metrics[9]
                        max_M_y_log.append(max_M_y)
                        min_M_x = env_metrics[10]
                        min_M_x_log.append(min_M_x)
                        min_M_y = env_metrics[11]
                        min_M_y_log.append(min_M_y)
                        ep_duration = env_metrics[12]
                        ep_duration_log.append(ep_duration)
                        evaluation = env_metrics[13]
                        average_phi_x_log.append(env_metrics[14])
                        average_phi_y_log.append(env_metrics[15])
                        max_phi_x_log.append(env_metrics[16])
                        max_phi_y_log.append(env_metrics[17])
                        min_phi_x_log.append(env_metrics[18])
                        min_phi_y_log.append(env_metrics[19])
                        
                        average_M_z_log.append(env_metrics[20])
                        #print(f'average_M_z = {average_M_z_log[-1]}')
                        max_M_z_log.append(env_metrics[21])
                        #print(f'max_M_z = {max_M_z_log[-1]}')
                        min_M_z_log.append(env_metrics[22])
                        average_freq_log.append(env_metrics[23])
                        max_freq_log.append(env_metrics[24])
                        min_freq_log.append(env_metrics[25])
                        std_M_x_log.append(env_metrics[26])
                        std_M_y_log.append(env_metrics[27])
                        std_M_z_log.append(env_metrics[28])
                        std_freq_log.append(env_metrics[29])
                        #print(f'std_freq = {std_freq_log[-1]}')
                        std_phi_x_log.append(env_metrics[30])
                        std_phi_y_log.append(env_metrics[31])
                        net_theta = env_metrics[32]
                        #print(f'std_phi_y = {std_phi_y_log[-1]}')
                        print(f'***************frames = {frames}')
                        
                    
                        
                        if evaluation == 1:
                            try:
                                evaluation_tensorboard.update_stats(evaluation_success = ep_successes[-1],
                                                         evaluation_score = ep_rewards[-1],
                                                         evaluation_theta_dot = ep_theta_dot[-1],
                                                         evaluation_frames = frames,
                                                         evaluation_episode = episode,                                                
                                                         evaluation_theta = theta,
                                                         evaluation_ep_duration = ep_duration
                                                         )
                            except:
                                print('unable to record the evaluation run on the evaluation tensorboard')
                        
                    
                        
                    #create a minibatch, and send it to the learner over the minibatch queue
                    if frames > UPDATE_AFTER and minibatch_q.empty() and minibatches < MAX_UPDATES_PER_FRAME*frames: #buffer.buffer_counter: #this should be the number of frames
                        try:
                            #print('putting a minibatch')
                            minibatch = buffer.get_minibatch()
                            minibatch_q.put(minibatch)
                            minibatches += 1
                            print(f'*****************{minibatches} minibatches')
                        except:
                            print('no samples to get minibatch from')
                            pass
                    if not training_logs_q.empty():
                        #print('master getting training logs')
                        training_logs = training_logs_q.get()
                        try:
                           # print(f'here is the training log: {training_logs}')
                            q_loss = training_logs[0]
                            pi_loss = training_logs[1]
                            alpha = training_logs[2]
                            alpha_loss = training_logs[3]
                            entropy = training_logs[4]
                            #print(f'training logs: pi_loss = {pi_loss}, q_loss = {q_loss}, alpha = {alpha}, alpha_loss = {alpha_loss}, entropy = {entropy}')
                        except:
                            print('something went wrong in the training log')
                        
                    
                    
                    
                    if (not SIMULATION and records % TENSORBOARD_UPDATE_FREQUENCY == 0 and records > 0) or (SIMULATION and minibatches%SIM_UPDATE_EVERY == 0 and minibatches > 0 and records > 0):
                        print(f'******************updating tensorboard with {records} records*************************')
                        update_time = time.time()
                        try:
                            
                            average_success = sum(ep_successes[-records:])/records
                            if episode >= 100:
                                long_average_score = sum(ep_rewards[-100:])/100
                                if long_average_score >= top_score:
                                    print(f'new best performing model with score: {long_average_score} ')
                                    top_score = long_average_score
                                    model_save_q.put(top_score)
                            average_score = sum(ep_rewards[-records:])/records
                            max_score = max(ep_rewards[-records:])
                            min_score = min(ep_rewards[-records:])
                            average_success = sum(ep_successes[-records:])/records
                            average_theta_dot = sum(ep_theta_dot[-records:])/records
                            max_theta_dot = max(ep_theta_dot[-records:])
                            min_theta_dot = min(ep_theta_dot[-records:])
                            max_M_x = max(max_M_x_log[-records:])
                            max_M_y = max(max_M_y_log[-records:])
                            min_M_x = min(min_M_x_log[-records:])
                            min_M_y = min(min_M_y_log[-records:])
                            average_M_x = sum(average_M_x_log[-records:])/records
                            average_M_y = sum(average_M_y_log[-records:])/records
                            average_ep_duration = sum(ep_duration_log[-records:])/records
                            max_phi_x = max(max_phi_x_log[-records:])
                            max_phi_y = max(max_phi_y_log[-records:])
                            min_phi_x = min(min_phi_x_log[-records:])
                            min_phi_y = min(min_phi_y_log[-records:])
                            average_phi_x = sum(average_phi_x_log[-records:])/records
                            average_phi_y = sum(average_phi_y_log[-records:])/records
                            
                            
                            average_M_z = sum(average_M_z_log[-records:])/records
                            max_M_z = max(max_M_z_log[-records:])
                            min_M_z = min(min_M_z_log[-records:])
                            average_freq = sum(average_freq_log[-records:])/records
                            max_freq = max(max_freq_log[-records:])
                            min_freq = min(min_freq_log[-records:])
                            std_M_x = sum(std_M_x_log[-records:])/records
                            std_M_y = sum(std_M_y_log[-records:])/records
                            std_M_z = sum(std_M_z_log[-records:])/records
                            std_freq = sum(std_freq_log[-records:])/records
                            std_phi_x = sum(std_phi_x_log[-records:])/records
                            std_phi_y = sum(std_phi_y_log[-records:])/records
                            print(f'\n********************************************************\naverage_score: {average_score}, average_success = {average_success}, average_theta_dot = {average_theta_dot}')
                            
                            
                            
                                
                            tensorboard.update_stats(average_success = average_success,
                                                     average_score = average_score,
                                                     average_theta_dot = average_theta_dot,
                                                     frames = frames,
                                                     episode = episode,
                                                     q_loss = q_loss,
                                                     pi_loss = pi_loss,
                                                     alpha = alpha,
                                                     theta = theta,
                                                     minibatches = minibatches,
                                                     max_score = max_score,
                                                     min_score = min_score,
                                                     max_theta_dot = max_theta_dot,
                                                     min_theta_dot = min_theta_dot,
                                                     max_M_x = max_M_x,
                                                     max_M_y = max_M_y,
                                                     min_M_x = min_M_x,
                                                     min_M_y = min_M_y,
                                                     average_M_x = average_M_x,
                                                     average_M_y = average_M_y,
                                                     average_ep_duration = average_ep_duration,
                                                     records = records,
                                                     entropy = entropy,
                                                     alpha_loss = alpha_loss,
                                                     average_phi_x = average_phi_x,
                                                     average_phi_y = average_phi_y,
                                                     max_phi_x = max_phi_x,
                                                     max_phi_y = max_phi_y,
                                                     min_phi_x = min_phi_x,
                                                     min_phi_y = min_phi_y,
                                                     top_score = top_score,
                                                     long_average_score = long_average_score,
                                                     max_M_z = max_M_z,
                                                     min_M_z = min_M_z,
                                                     average_M_z = average_M_z,
                                                     average_freq = average_freq,
                                                     max_freq = max_freq,
                                                     min_freq = min_freq,
                                                     std_M_x = std_M_x,
                                                     std_M_y = std_M_y,
                                                     std_M_z = std_M_z,
                                                     std_freq = std_freq,
                                                     std_phi_x = std_phi_x,
                                                     std_phi_y = std_phi_y,
                                                     net_theta = net_theta
                                                     )
                            records = 0
                            print('                                                                   updated tensorboard')
                            
                            #plt.hist(theta_log, bins='auto')
                            #plt.title('theta histogram')
                            #plt.show()
            
                        except:
                            #abort_q.put(1)
                            print('Master process error: no data to update tensorboard with')
                            
                    
                #print('master process resetting')
                #done_q.put(1)
    except:
        abort_q.put(1)
        print('master process raised an error')
    
def learner_process(minibatch_q, training_logs_q, done_q, begin_q, learner_device_ID, new_pi_q, abort_q, reset_q, model_save_q):
    tf.random.set_seed(time.time())
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    
    with tf.device('/device:gpu:0'): 
        try:
            
            print(f'learner process beginning with GPU{learner_device_ID} and process ID: {os.getpid()}')
            
            while done_q.empty() and abort_q.empty():
                minibatches_gotten = 0
                if VERBOSE: print('generating models')
                if LOAD_PI == 1:
                    print("you are here")
                    print(f"this is the model to load: {load_pi_model_name}")
                    pi_model = tf.keras.models.load_model(f'{model_save_path}\{load_pi_model_name}.h5')
                    print("you loaded a model")
                else:
                    if CONVOLUTIONAL:
                        pi_model = get_conv_actor(STATE_SHAPE, NUM_ACTIONS)
                    else:
                        if VERBOSE: print('generating fresh pi model')
                        pi_model = get_state_actor(STATE_SHAPE, NUM_ACTIONS)
                        if VERBOSE: print('successfully generated fresh pi model')
                    
                if LOAD_Q == 1:
                    q1_model = tf.keras.models.load_model(f'{model_save_path}\{load_q1_model_name}.h5')
                    target_q1 = tf.keras.models.load_model(f'{model_save_path}\{load_q1_model_name}.h5')
                    q2_model = tf.keras.models.load_model(f'{model_save_path}\{load_q2_model_name}.h5')
                    target_q2 = tf.keras.models.load_model(f'{model_save_path}\{load_q2_model_name}.h5')
                    
                else:
                    if CONVOLUTIONAL:
                        q2_model = get_conv_critic(STATE_SHAPE, NUM_ACTIONS)
                        target_q2 = get_conv_critic(STATE_SHAPE, NUM_ACTIONS)
                        target_q2.set_weights(q2_model.get_weights())
                        
                        q1_model = get_conv_critic(STATE_SHAPE, NUM_ACTIONS)
                        target_q1 = get_conv_critic(STATE_SHAPE, NUM_ACTIONS)
                        target_q1.set_weights(q1_model.get_weights())
                    else:
                        q2_model = get_state_critic(STATE_SHAPE, NUM_ACTIONS)
                        target_q2 = get_state_critic(STATE_SHAPE, NUM_ACTIONS)
                        target_q2.set_weights(q2_model.get_weights())
                        
                        q1_model = get_state_critic(STATE_SHAPE, NUM_ACTIONS)
                        target_q1 = get_state_critic(STATE_SHAPE, NUM_ACTIONS)
                        target_q1.set_weights(q1_model.get_weights())
                
                #pi_model.summary()
                #q1_model.summary()
                #q2_model.summary()
                
                log_alpha = tf.Variable(INITIAL_LOG_ALPHA, trainable = True)
                
                
                print('learner waiting for start signal')
                while begin_q.empty():
                    pass
                print('learner starting')
                start_time = time.time()
                #print(f'minibatches modulo reset: {minibatches_gotten % RESET}')
                      
                while done_q.empty() and abort_q.empty(): 
                    
                    if not minibatch_q.empty():
                        #print('learner is requesting a minibatch')
                        state_batch, action_batch, reward_batch, next_state_batch, done_batch = minibatch_q.get() #retrieve the minibatch from the buffer
                        #print(f'\n\nlearner got a minibatch, and the input is of shape {np.shape(state_batch)}\n\n')
                        state_batch = tf.convert_to_tensor(state_batch)
                        action_batch = tf.convert_to_tensor(action_batch)
                        reward_batch = tf.convert_to_tensor(reward_batch)
                        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
                        next_state_batch = tf.convert_to_tensor(next_state_batch)
                        done_batch = tf.convert_to_tensor(done_batch)
                        done_batch = tf.cast(done_batch, dtype=tf.float32)
                        minibatches_gotten += 1
                        #print(f'learner converted the batch to tensors of type {type(done_batch)}')
                    
                        #q1_loss, q2_loss, pi_loss, alpha_loss, alpha = learn(state_batch, action_batch, reward_batch, next_state_batch,
                        #                                                         pi_model, q1_model, q2_model, target_q1, target_q2)
                        
                        q1_loss, q2_loss, pi_loss, alpha_loss, entropy  = learn(state_batch, action_batch, reward_batch, next_state_batch, done_batch,
                                                                                 pi_model, q1_model, q2_model, target_q1, target_q2, log_alpha, minibatches_gotten)
                        
                        #print('learner completed training')
                        training_logs = (q1_loss.numpy(), pi_loss.numpy(), alpha, alpha_loss, entropy)
                        #print('learner made the training logs')
                        training_logs_q.put(training_logs)
                        #print('learner sent the training logs')
                    
                    if not model_save_q.empty():
                        score = model_save_q.get()
                        print(f'saving the top models now with score = {score}')
                        pi_model.save(f'{model_save_path}\\top_{pi_model_name}_{score}_{minibatches_gotten}.h5')
                        q1_model.save(f'{model_save_path}\\top_{q1_model_name}_{score}_{minibatches_gotten}.h5')
                        q2_model.save(f'{model_save_path}\\top_{q2_model_name}_{score}_{minibatches_gotten}.h5')
                        print('saved the top model')
                    
                    if time.time() - start_time > 60:
                        start_time = time.time()
                        print(f'saving the models now at time {start_time}')
                        pi_model.save(f'{model_save_path}\{pi_model_name}.h5')
                        q1_model.save(f'{model_save_path}\{q1_model_name}.h5')
                        q2_model.save(f'{model_save_path}\{q2_model_name}.h5')
                        new_pi_q.put(1)
                    if not reset_q.empty():
                        reset_q.get()
                        run_updates()
                        break
            
                    
                print(f'learning process ending with {minibatches_gotten} minibatches')
        except:
            abort_q.put(1)
            print('learner process raised an error')

    
    
if __name__ == '__main__':
    #make the queues
    frame_q = mp.Queue()
    frame_request_q = mp.Queue()
    quit_q = mp.Queue()
    begin_q = mp.Queue()
    observation_q = mp.Queue()
    env_metrics_q = mp.Queue()
    new_pi_q = mp.Queue()
    pause_q = mp.Queue()
    training_logs_q = mp.Queue()
    minibatch_q = mp.Queue()
    done_q = mp.Queue()
    abort_q = mp.Queue()
    env_reset_q = mp.Queue()
    learner_reset_q = mp.Queue()
    model_save_q = mp.Queue()
    
    
    #make the processes
    vimba_process = mp.Process(target = camera_process, args = (frame_q, frame_request_q, abort_q))
    env_process = mp.Process(target = environment_process, args = (frame_q, frame_request_q, quit_q, begin_q, env_device_ID, env_metrics_q, observation_q, new_pi_q, pause_q, abort_q, env_reset_q))
    main_process = mp.Process(target = master_process, args = (master_device_ID, begin_q, observation_q, env_metrics_q, training_logs_q, minibatch_q, done_q, pause_q, abort_q, env_reset_q, learner_reset_q, model_save_q))
    learn_process = mp.Process(target = learner_process, args = (minibatch_q, training_logs_q, done_q, begin_q, learner_device_ID, new_pi_q, abort_q,learner_reset_q, model_save_q))
    GUI_process = mp.Process(target = pauser, args = (pause_q, abort_q, env_metrics_q, training_logs_q))
    
    GUI_process.start()
    learn_process.start()
    time.sleep(5)
    #start the processes
    env_process.start()
    time.sleep(5)
    if not SIMULATION:
        vimba_process.start()
        time.sleep(5)
    main_process.start()
    #time.sleep(5)
    #time.sleep(5)
    #time.sleep(5)
    
    #join the processes#
    main_process.join()
    if not SIMULATION: vimba_process.join()
    env_process.join()
    learn_process.join()
    GUI_process.join()
    
    print('Done')
    input()