'''**********************************************setup parameters'''
#give the experiment a name. usually something to do with today's date
MODEL_NAME = 'april13_2022_3' #example: "jan10_2022"

#do we want to run a simulated microrobot (probably not)
SIMULATION = 0 #default is 0

# do we want a ton of cluttery debugging messages?
VERBOSE = 0 #default is 0

#do we want to run a convolutional neural net or a state neural net?
CONVOLUTIONAL = 0 #default is 0

#how many possible actions can the microrobot take?
NUM_ACTIONS = 4 #default is 4

#do we want to run multiple experiments in a row, doing a sweep of a certain parameter?
PARAMETER_SEARCH = 0 #default is 0

#do we want to automatically tune the temperature hyperparamater?
UPDATE_ALPHA = 1 #default is 1

#do we want to evaluate a model we have already trained? This runs the model deterministically. 
CONTINUOUS_EVALUATION = 0 #default is 0

#load a pretrained actor model?
LOAD_PI = 0 #default is 0
load_pi_model_name = 'top_pi_model_microrobot may 28_978' #default: "top_pi_model_microrobot may 28_978", a non-convolutional model
    
#load pretrained critic models?    
LOAD_Q = 0 #default is 0
load_q1_model_name = 'top_q1_model_microrobot may 28_978' #default: 'top_q1_model_microrobot may 28_978'
load_q2_model_name = 'top_q2_model_microrobot may 28_978' #default: 'top_q2_model_microrobot may 28_978'

#load a filled buffer from an earlier training session?
LOAD_BUFFER = 0 #default is 0
buffer_load_path = f'buffers/microrobot june 7 convolutional_buffers' #default: f'buffers/microrobot june 7 convolutional_buffers'

##how many frames to do at the beginning of training while randomly sampling actions, not getting from pi
IMPLEMENT_POLICY_AFTER = 500 #default is 1

#how many frames to run at the beginning before we start doing updates (fill the buffer)
UPDATE_AFTER = 1000 #default is 1000

#use these if you don't want to use the neural networks for the policy, but want to use a mathematical policy
#select only one at a time. for testing the integrity of the hardware (i.e. just seeing if the microrobot will go in a circle, chose sin_policy)
SQUARE_POLICY = 0 #default is 0
SIN_POLICY = 0 #default is 1
MIX_POLICY = 0 #default is 0
POLY_POLICY = 0 #default is 0

#update tensorboard every _ episodes if using robot
TENSORBOARD_UPDATE_FREQUENCY = 10 #default is 10

#if in simulation, update tensorboard every 50 episodes
SIM_UPDATE_EVERY = 50 #default is 50. 

#the characteristic dimension of the state image (NxN) if doing convolutional neural networks
STATE_SIZE = 64 #default is 64

#how many past episodes to keep track of in the state
N_STEPS = 3 #default is 3

#this is the size of the image that the camera sends over the frame_q. The size that the images are shown in the GUI
FRAME_SIZE = 300 #default is 300

#The size of the raw camera image in the GUI
ORIGINAL_FRAME_SIZE = 800 #default is 800

#time, in seconds, that we want to run this model for 
TRAINING_TIME = 50_000_000 #default is 50_000_000.

#How many frames do we want to run the experiment for?
MAX_FRAMES = 100_000 #default is 100_000

#how many (s,a,r,s*) experiences do we store in the FIFO buffer?
BUFFER_CAPACITY = 100_000 #default is 100_000. there is no reason for this to exceed MAX_FRAMES

#Minibatch size
BATCH_SIZE = 256 #default is 256

#critic learning rate
CRITIC_LR = 0.0001 #default is 0.0001

#actor learning rate
ACTOR_LR = 0.0001  #default is 0.0001

#temperature (alpha) learning rate
ALPHA_LR=0.0003 #default is 0.0003

#starting alpha value
alpha = 1 #default is 1

#starting log_alpha value
INITIAL_LOG_ALPHA = 0.0 #defualt is 0.0

#how many gradient steps to run between hyperparameter updates during a hyperparameter search session
RESET = 50_000 #default is 50_000

#the ratio of gradient steps to environment steps
MAX_UPDATES_PER_FRAME = 1 #default is 1

#run an evaluation episode with mu actions every n episodes
EVALUATION = 10 #default is 10

#save the buffers to the hard drive every n steps
BUFFER_SAVE = 10_000 #default is 10_000

#how many seconds to run the arduino before resetting (I think running too long is a problem)
ARDUINO_RESET_TIME = 60 #default is 60 (is this feature even doing anything right now?)