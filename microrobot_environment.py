# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 10:23:30 2021

@author: Windows
"""

'''
An environment has a reset, and a step, and a render
reset returns: state, reward, done, duration
state is a 64x64x3 picture
reward is an integer
done is a 0 or 1
duration is an integer
step takes actions, which is a tuple of float32 values

environments have a parameter called action_space, which gives the size of the action space, and the min and max values of each action
environments have a parameter called observation_space, which gives the size of the observation (64x64x3)
environments should have a list or dictionary or something listing discriptive names for the action space (Mx, My, Mz, etc...)
'''
from math import pi, sin, cos, sqrt
import numpy as np
import cv2
import serial
import time
import tensorflow as tf



        
class Microrobot_Env:
    
    def __init__(self, frame_q,frame_request_q, state_size, N_steps, verbose, convolutional, num_actions, log_save_path):
        self.frame_q = frame_q
        self.frame_request_q = frame_request_q
        self.convolutional = convolutional
        self.duration = 0
        self.done = 0
        self.success = 0
        self.verbose = verbose
        self.episode_length = 100 #number of steps in an episode
        self.state = 0
        self.theta = 0
        self.net_theta = 0
        self.theta_last = 0
        self.goal_distance = 20
        self.THETA_MARGIN = 3 #how many degrees off before we say we hit the target (this has to do with image processing accuracy)
        self.goal = self.theta + self.goal_distance
        self.success_reward = 1000 #controls the reward scale
        self.STATE_SIZE = state_size
        self.abort = 0
        self.STEP_DURATION = 0.3 #time in seconds of the lenght of an step
        self.last_time = time.time()
        self.N_steps = N_steps
        self.action_last = np.zeros((num_actions,))
        self.M_x_log = []
        self.M_y_log = []
        self.M_z_log = []
        self.freq_log = []
        self.phi_x_log = []
        self.phi_y_log = []
        self.theta_dot_log = []
        self.theta_log = []
        self.goal_log = []
        self.duration_log = []
        self.log_save_path = log_save_path
        
        
        try:
            print('trying to connect to arduino')
            self.ser = serial.Serial('COM5', 9600, timeout=1)
            print('successfully connected to arduino')
        except:

            print('unable to open com port with arduino')
            input()
        
    def reset_arduino(self):
        print('resetting the arduino in environment')
        try:
            self.arduino_communication(1, 2, 3, 4, 5, 6)
        except:
            print('did not complete arduino communication')
        print('arduino recieved command')
        time.sleep(10)
        try:
            print('trying to connect to arduino')
            self.ser = serial.Serial('COM8', 9600, timeout=1)
            print('successfully connected to arduino')
        except:

            print('unable to open com port with arduino')
            input()
    
    def correct_for_wrap(self, angle):
        if angle > 360:
            angle -= 360
            
        if angle < 0:
            angle += 360
            
        return angle
    
    def take_observation(self):
        if self.verbose: print('called take_observation')
        state_img, time_frame = self.get_picture()
        if self.verbose: print('got a picture from self.get_picture')
        state_img = cv2.cvtColor(state_img, cv2.COLOR_BGR2GRAY)
        if self.verbose: print('converted the color of the picture')
        threshold_image = self.threshold(state_img)
        if self.verbose: print('thresholded the picture')
        cv2.imshow('threshold image', threshold_image)
        cv2.waitKey(20)
        if self.verbose: print('about to id the robot')
        state = self.id_robot(threshold_image, state_img)
        if self.verbose: print('found the robot position')
        state = cv2.resize(state,(self.STATE_SIZE,self.STATE_SIZE))  
        if self.verbose: print('resized the image')
        return state
    
    def record_logs(self, M_x, M_y, M_z, freq, phi_x, phi_y, step_theta_dot):
        self.M_x_log.append(M_x)
        self.M_y_log.append(M_y)
        self.M_z_log.append(M_z)
        self.freq_log.append(freq)
        self.phi_x_log.append(phi_x)
        self.phi_y_log.append(phi_y)
        self.theta_dot_log.append(step_theta_dot)
        self.theta_log.append(self.theta)
        self.goal_log.append(self.goal)
        self.duration_log.append(self.duration)
        
    def save_logs(self):
        print('___________________________saving logs_______________________')
        np.save(f'{self.log_save_path}/M_x_log', self.M_x_log)
        np.save(f'{self.log_save_path}/M_y_log', self.M_y_log)
        np.save(f'{self.log_save_path}/M_z_log', self.M_z_log)
        np.save(f'{self.log_save_path}/freq_log', self.freq_log)
        np.save(f'{self.log_save_path}/phi_y_log', self.phi_y_log)
        np.save(f'{self.log_save_path}/phi_x_log', self.phi_x_log)
        np.save(f'{self.log_save_path}/theta_dot_log', self.theta_dot_log)
        np.save(f'{self.log_save_path}/theta_log', self.theta_log)
        np.save(f'{self.log_save_path}/goal_log', self.goal_log)
        np.save(f'{self.log_save_path}/duration_log', self.duration_log)

        print('___________________________done saving logs_______________________')
        
    
    def step(self, action):
        states = []
        step_theta_dot = 0.0
        step_reward = 0.0
        self.abort = 0
        
        print(f'action is: {action}')
        
        '''unpack the action, and convert to proper range'''
        M_x = action[0]
        M_y= action[1]
        M_z = max(abs(M_x), abs(M_y))
        freq = 80
        phi_x = action[2]
        phi_x = (phi_x + 1)*pi*2 / 2 #convert to range (0,2pi)
        phi_y = action[3]
        phi_y = (phi_y + 1)*pi*2 / 2
            
        if tf.is_tensor(M_x):
            M_x = float(M_x.numpy())
        if tf.is_tensor(M_y):
            M_y = float(M_y.numpy())
        if tf.is_tensor(M_z):
            M_z = float(M_z.numpy())
        if tf.is_tensor(freq):
            freq = float(freq.numpy())
        if tf.is_tensor(phi_x):
            phi_x = float(phi_x.numpy())
        if tf.is_tensor(phi_y):
            phi_y = float(phi_y.numpy())
        
        '''define movement according to the actions by sending commands to the arduino''' 
        try:
            self.arduino_communication(M_x, M_y, M_z, freq, phi_x, phi_y)
        except:
            print('arduino communication failed')
        
        
        
        '''take N_steps observations'''
        
        for i in range(self.N_steps):
        
            step_start = time.time()
            
            '''do the action for step_duration seconds'''
            start = time.time()
            while time.time() - self.last_time < self.STEP_DURATION:
                pass
            self.last_time = time.time()
    
            '''take an observation. this takes the longest. immediately after this, then send the new command to the arduino'''
            observation = self.take_observation() #this updates self.theta
            cv2.imshow('state', observation)
            cv2.waitKey(20)
            
            '''taking an observation gave us a new theta, so we can caulate movement'''
            theta_dot = self.theta-self.theta_last
            if theta_dot < -300: theta_dot += 360 #for the edge case when theta is 0 and theta_last is 359
            if theta_dot > 300: theta_dot -= 360 #for the edge case when theta is 359 and theta_last is 0  
            step_theta_dot += theta_dot
            
            '''generate the state image based on theta and the goal'''
            if self.convolutional:
                self.state = observation/255
            else:  
                #print('generating states')
                if abs(self.goal-self.theta) > 200:
                    self.state = (self.theta/360,
                                  (self.goal+360-self.theta)/360,
                                  (self.episode_length-self.duration)/self.episode_length,
                                  self.action_last[0],
                                  self.action_last[1],
                                  self.action_last[2],
                                  self.action_last[3])
                else:
                    #print('here making a state')
                    self.state = (self.theta/360,
                                  (self.goal-self.theta)/360,
                                  (self.episode_length-self.duration)/self.episode_length,
                                  self.action_last[0],
                                  self.action_last[1],
                                  self.action_last[2],
                                  self.action_last[3])
                    #print(f'done making the state: {self.state}')
            states.append(self.state)

            self.duration += 1 #increment the step timer
            self.theta_last = self.theta
            
            
        #print(f'done generating states: {states}')
        self.record_logs(M_x, M_y, M_z, freq, phi_x, phi_y, step_theta_dot)
        self.action_last = action
        self.net_theta += step_theta_dot
        if self.duration >= self.episode_length: #check to see if done with episode
            self.done = 1
            
        step_reward = step_theta_dot
        #if abs(step_reward) > 200:
        #    step_reward = abs(step_reward) -  360 #edge case at circle wraparound
        
        '''if we have reached the goal give success reward and finish the episode'''
        if abs(self.goal - self.theta) < self.THETA_MARGIN:
            step_reward += self.success_reward
            self.success = 1
            self.done = 1
            
        
        if self.N_steps > 1:
            if self.convolutional:
                final_state = np.stack((states[-self.N_steps:]),2)
            else:
                try:
                    final_state = np.stack((states[-self.N_steps:]),1)
                    #print(f'final state shape = {np.shape(final_state)}')
                except:
                    #print('np.stack failed and state shape is {np.shape(states)}, giving last final state and aborting')
                    self.abort = 1
            
            if self.success: print('SUCCESS!!!')
            #print(f'step metrics: step reward: {step_reward}, step_theta_dot: {step_theta_dot}, goal_distance: {self.goal-self.theta}')
            #print(f'step metrics: theta: {self.theta}, goal: {self.goal}, net_theta: {self.net_theta}')
            #print(f'step metrics: state: {final_state}')
            return final_state, step_reward, self.done, self.duration, self.success, self.abort, step_theta_dot, self.theta, self.net_theta
        else:
            self.state = self.state.reshape(self.STATE_SIZE,self.STATE_SIZE,1)
            if self.success: print('SUCCESS!!!')
            return self.state, step_reward, self.done, self.duration, self.success, self.abort, step_theta_dot, self.theta, self.net_theta
    
    def reset(self, action):
        self.save_logs()
        self.duration = 0
        self.done = 0
        self.success = 0
        self.reward = 0
        if self.verbose: print('taking observation')
        self.take_observation()
        if self.verbose: print('successfully took observation')
        self.goal = self.theta + self.goal_distance
        self.goal = self.correct_for_wrap(self.goal)
        self.action_last = action
        print('did a reset, now doing a step')
        return self.step(action)
    
    def render(self):
        #state_RGB = cv.cvtColor(self.state, cv.COLOR_BGRA2BGR)
        cv2.imshow('state', self.state)
        cv2.waitKey(30)
        
    def arduino_communication(self, M_x, M_y, M_z, freq, phi_x, phi_y):

        action_string = f"Mydata={round(M_x,2)},{round(M_y,2)},{round(M_z,2)},{round(freq,2)},{round(phi_x,2)},{round(phi_y,2)}"
        print(action_string)
        action_string += "\n"
        action_string_bytes = action_string.encode('utf-8')
        self.ser.write(action_string_bytes)
        #time.sleep(.1)
        #print(i)
        Arduino_response = self.ser.readline()
        Arduino_response = Arduino_response.decode('utf-8')
        print(Arduino_response)
    
    def get_picture(self):
        #print(f'getting frame at time {time.time()}')
        self.frame_request_q.put(1)
        try:
            (frame, time_frame) = self.frame_q.get()
        except:
            print('frame_q was empty')
        
        #self.showInMovedWindow('frame',frame, 0, 400)
        #cv2.waitKey(20)
        return frame, time_frame
    
    def showInMovedWindow(self, winname, img, x, y):
        cv2.namedWindow(winname)        # Create a named window
        cv2.moveWindow(winname, x, y)   # Move it to (x,y)
        cv2.imshow(winname,img)
        
    def threshold(self, image):
        ret,thresh1 = cv2.threshold(image,200,255,cv2.THRESH_BINARY)
        cv2.imshow('threshold', thresh1)
        thresh1_inverse = cv2.bitwise_not(thresh1)
        
        kernel = np.ones((13,13),np.uint8)
        thresh1_dilate = cv2.dilate(thresh1_inverse,kernel,iterations = 1)
        cv2.imshow('dilated', thresh1_dilate)
        cv2.waitKey(20)
        state = thresh1_dilate
        return state
    
    def pause(self):
        self.arduino_communication(0, 0, 0, 0, 0, 0)
    
    def id_robot(self, image, state):

        contours, hierarchy = cv2.findContours(image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        cX = []
        cY = []
        contour_size = []
        angle = 0
        for c in contours:
            # calculate moments for each contour
            M = cv2.moments(c)
            # calculate x,y coordinate of center
            cX.append(int(M["m10"] / M["m00"]))
            cY.append(int(M["m01"] / M["m00"]))
            contour_size.append(int(M["m00"])) #find the size of each contour
           
        if len(cY) == 2:
            #find the larger of the two contours. this is the center.
            robot_index = contour_size.index(min(contour_size)) 
            center_index = contour_size.index(max(contour_size)) 
           
            angle = np.arctan((cY[robot_index]-cY[center_index])/(cX[robot_index]-cX[center_index]+.000001))
            distance = sqrt((cY[robot_index]-cY[center_index])**2+(cX[robot_index]-cX[center_index])**2)
            
            angle = angle*180/pi
            if cX[robot_index] > cX[center_index]:
                    angle = angle + 90
            elif cX[robot_index] < cX[center_index]:
                    angle = angle + 270
            elif cX[robot_index] == cX[center_index] and cY[robot_index] > cY[center_index]:
                angle = 180
            elif cX[robot_index] == cX[center_index] and cY[robot_index] < cY[center_index]:
                angle = 0
            if angle == -90:
                angle = 0
    
            
            color = (0, 0, 0) #BGR image space because cv2
            thickness = 2
            length = distance
            center_point = (cX[center_index], cY[center_index]) 
            #robot_point = (cX[robot_index], cY[robot_index])
            goal_point = (int(cX[center_index]+length*sin(self.goal*pi/180)), int(cY[center_index]-length*cos(self.goal*pi/180))) 
            image_with_goal = cv2.line(state, center_point, goal_point, color, thickness)  
            self.theta = angle
            
            return image_with_goal
            
        else:
            self.abort = 1
            
            print("ABORT:: more than two objects detected")
            return image
            
class Microrobot_Sim:
    
    def __init__(self, state_size, N_steps, verbose, convolutional, num_actions):
        self.convolutional = convolutional
        self.N_steps = N_steps
        self.verbose = verbose
        self.duration = 0
        self.done = 0
        self.success = 0
        self.episode_legnth = 100 #number of steps in an episode
        self.reward = 0
        self.state = 0
        self.theta = 180
        self.goal_distance = 10
        self.goal = self.theta + self.goal_distance
        self.target_direction = 1
        self.success_reward = 100
        self.theta_dot = 0
        self.freq_cutoff = 0.8
        self.theta_last = 0
        self.THETA_MARGIN = -3
        self.abort = 0
        
        self.action_last = np.zeros((num_actions,))
        
        #self.action_space = np.array([[-1,-1], [1,1]])
        self.action_space = np.array([[-1,-1,-1, -1, -1, -1], [1,1,1,1, 1, 1]])
        self.observation_space = [0,0]
        
        
    def step(self, action):
        
        states = []
        
        
        
        if self.verbose: print(f'action is: {action}')
        
        '''unpack the action, and convert to proper range'''
        M_x = action[0]
        if self.verbose: print(f'M_x is {M_x}')
        M_y= action[1]
        if self.verbose: print(f'M_y is {M_y}')
        M_z = max(abs(M_x), abs(M_y))
        freq = 0.8
        phi_x = 90
        phi_y = 90
        
        if np.size(action) > 2:
            M_z = action[2]
            M_z = (M_z + 1) / 2 
        if np.size(action) > 3:
            freq= action[3]
            freq = (freq + 1)*100 / 2
        if np.size(action) > 4:
            phi_x = action[4]
            phi_x = (phi_x + 1)*pi*2 / 2
            phi_y = action[5]
            phi_y = (phi_y + 1)*pi*2 / 2 #convert from range (-1,1) to range (0,2pi)
            
        if tf.is_tensor(M_x):
            M_x = float(M_x.numpy())
        if tf.is_tensor(M_y):
            M_y = float(M_y.numpy())
        if tf.is_tensor(M_z):
            M_z = float(M_z.numpy())
        if tf.is_tensor(freq):
            freq = float(freq.numpy())
        if tf.is_tensor(phi_x):
            phi_x = float(phi_x.numpy())
        if tf.is_tensor(phi_y):
            phi_y = float(phi_y.numpy())
        
        self.action_last = action
        
        
        for i in range(self.N_steps):
            if freq <= self.freq_cutoff:
                self.theta_dot = 1.25*freq*M_z*(M_y*sin(phi_y*pi/180)*sin(self.theta*pi/180-pi/2)+ M_x*sin(phi_x*pi/180)*sin(self.theta*pi/180))
            else:
                self.theta_dot = (-5*freq + 5)*freq*M_z*(M_y*sin(phi_y*pi/180)*sin(self.theta*pi/180-pi/2)+ M_x*sin(phi_x*pi/180)*sin(self.theta*pi/180))            
                     
                        
            self.theta_last = self.theta
            self.theta += self.theta_dot # update the state
            self.theta = self.correct_for_wrap(self.theta)
            #print(f'theta_dot: {self.theta_dot}')
            
            
            
            self.reward = -abs(self.goal-self.theta)
            if abs(self.goal - self.theta) > 300: #assume that theta is 359 and goal is 9, then the difference is 350
                self.reward = -abs(self.goal+360-self.theta)
                
            '''if we have reached the goal give success reward and finish the episode'''
            if self.reward > self.THETA_MARGIN:
                self.reward += self.success_reward
                self.success = 1
                #print('SUCCESS!!!')
                self.done = 1
            #print(f'reward = {self.reward}')
            
            
            
            
            
            
            
            self.state = (self.theta/360, self.action_last[0], self.action_last[1], (self.goal-self.theta)/360)
            #print(f'self.state = {self.state}')
            
            self.duration += 1 #increment the step timer
            if self.duration == self.episode_legnth: #check to see if done with episode
                self.done = 1
                
            states.append(self.state)
    
        if self.N_steps > 1:
            #print('stacking')
            if self.convolutional:
                final_state = np.stack((states[-self.N_steps:]),2)
            else:
                final_state = np.stack((states[-self.N_steps:]),1)
            if self.verbose: print('done stacking')
            #print(f'final state shape = {np.shape(final_state)}')
            #print(f'states shape = {np.shape(states)}')
            return final_state, self.reward, self.done, self.duration, self.success, self.abort, self.theta_dot, self.theta
        else:
            if self.convolutional:
                self.state = self.state.reshape(self.STATE_SIZE,self.STATE_SIZE,1) 
            #print(f'final state shape = {np.shape(final_state)}')
            return self.state, self.reward, self.done, self.duration, self.success, self.abort, self.theta_dot, self.theta
    
    def reset(self, action):
        self.duration = 0
        self.done = 0
        self.success = 0
        self.reward = 0
        self.theta = np.random.randint(0,359)

        self.goal = self.theta + self.goal_distance
        self.goal = self.correct_for_wrap(self.goal)
        self.action_last = action
        #print('resetting')
                   
        return self.step(action)
    
            
    def correct_for_wrap(self, angle):
        if angle > 360:
            angle -= 360
            
        if angle < 0:
            angle += 360
            
        return angle
                
    
    
    

    
