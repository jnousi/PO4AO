import torch
#torch.set_deterministic(True)
from torch import optim, save
import numpy as np
import cupy as cp
import time
import datetime
import os
from collections import deque
import matplotlib.pyplot as plt
import scipy.io
from astropy.io import fits
import torch.multiprocessing as mp
import json

from po4ao_config import config
from po4ao_models import ConvDynamicsFast, EnsembleDynamicsFast, ConvPolicyFastFast
from po4ao_util import get_n_params, EfficientExperienceReplay, SharedAdam

#cosmic imports
from Octopus.CacaoInterface import getInterface
import FpsInterfaceWrap as fiw
import CacaoInterfaceWrap as ciw

# read parameters for optimized performance
iters                = config['RL']['iterations']
episode_length       = config['RL']['episode_length']
initial_sigma        = config['RL']['max_sigma'] 
min_sigma            = config['RL']['min_sigma']
warmup_episodes      = config['RL']['warmup_episodes'] 
loss_penalty         = config['RL']['loss_function_penalty'] 

n_history            = config['MDP']['n_history']
planning_horizon     = config['MDP']['planning_horizon']
data_shape           = config['MDP']['data_shape'] 
control_delay         = config['MDP']['control_delay'] 

gain                 = config['integrator']['gain']
leak                 = config['integrator']['leak']
nmodes               = config['integrator']['n_modes']
integrator           = config['integrator']['integrator']

replay_size          = config['replay_buffers']['replay_size']
warmup_memory        = config['replay_buffers']['warmup_memory'] 
train_warmup_percent = config['replay_buffers']['train_warmup_percent']

batch_size           = config['NN_models']['training_batch']

device0              = 'cuda:0'
device1              = 'cuda:1'



def map_dtype(dtype):
    return {
        float     : ciw.CacaoInterfaceDouble,
        np.float64: ciw.CacaoInterfaceDouble,
        np.float32: ciw.CacaoInterfaceFloat,
        np.float16: ciw.CacaoInterfaceHalf,
        np.int16  : ciw.CacaoInterfaceInt16,
        int       : ciw.CacaoInterfaceInt32,
        np.uint32 : ciw.CacaoInterfaceInt32,
        np.int64  : ciw.CacaoInterfaceInt64,
        np.int8   : ciw.CacaoInterfaceInt8,
        np.uint16 : ciw.CacaoInterfaceUint16,
        np.uint32 : ciw.CacaoInterfaceUint32,
        np.uint64 : ciw.CacaoInterfaceUint64,
        np.uint8  : ciw.CacaoInterfaceUint8
    }[dtype]


# load the relevant SHMs
commands_from_cosmic = map_dtype(np.float32)('commands')
commands_from_cosmic_image = map_dtype(np.float32)('commands_image')
commands_to_cosmic = map_dtype(np.float32)('leak_command')
# commands_to_cosmic_image = map_dtype(np.float32)('commands_1_image')

# get the GPU buffers (cupy)
commands_from_cosmic_gpu = commands_from_cosmic.gpu_buffer()[:,0,0]
commands_from_cosmic_gpu_image = commands_from_cosmic_image.gpu_buffer()[:,0,0]
commands_to_cosmic_gpu = commands_to_cosmic.gpu_buffer()[:,0,0]

# Allocate some memory on the gpu using torch
prev_commands = torch.as_tensor(np.zeros((24,24)), device=device0)
commands = torch.as_tensor(commands_from_cosmic_gpu, device=device0)
dm = torch.as_tensor(np.zeros((24,24)), device=device0)


dm_coord = scipy.io.loadmat('dm_coord.mat')
dm_x = dm_coord['x']
dm_y = dm_coord['y']
dm_coords= (dm_x, dm_y)

dm_image_torch = torch.as_tensor(np.zeros((24,24)), device=device0, dtype= torch.float32)
dm_image_cp = cp.asarray(dm_image_torch)
command_vector = np.zeros(492, dtype = np.float32)
# dm_image[dm_coords] = command_vector
# command_vector = dm_image[dm_coords]

# Output to cosmic needs to be in cupy format. This is a view of the same memory in torch, just in cupy
dm_cp = cp.asarray(dm)
m2v_data = fits.open('KL_PTT_ifun_BMC492_cap6.fits')[0].data[:, 1:]
cupy_fifo = cp.zeros((1+control_delay, 492))

#control signal delay
def push_to_fifo(fifo, x):
    fifo[:-1,:] = fifo[1:,:]
    fifo[-1,:] = x

@torch.no_grad()
def sample_noise(sigma, flt, xvalid, yvalid):
    action_vec = torch.matmul(flt, sigma * torch.sign(torch.randn((492,)).to(device0)))
    action_im = torch.zeros((24,24)).to(device0)
    action_im[xvalid, yvalid] = action_vec
    return action_im

@torch.no_grad()
def step(action):
    """
    Pipeline specific function that sends new commands to dm, i.e., sets the action, and reads the 
    following WFS measurement projected to DM space trought a linear recontructor. The action and
    the WFS measurement have to be 2D images.

    :param action:             2D image of DM control voltages to be applied
    :return dm_image_torch:   2"D image of WFS measurement projected to DM voltages
    """

    # dm_image_cp[dm_coords] = commands_from_cosmic_gpu
    
    # dm[:] = prev_commands - (dm_image_torch * gain)
    dm[:] = (prev_commands * leak) + (action)
    prev_commands[:] = dm.clamp(-0.65,0.35)
    
    ## FIFO
    push_to_fifo(cupy_fifo, dm_cp[dm_coords])


    # Calculations are done, move results into the correct SHM buffer
    # commands_to_cosmic_gpu[:] = dm_cp[dm_coords]

    ## FIFO
    commands_to_cosmic_gpu[:] = cupy_fifo[0,:]


    # Tell COSMIC that there is new commands available
    commands_to_cosmic.notify()

    commands_from_cosmic_image.wait()
    # dm_image_cp[dm_coords] = commands_from_cosmic_gpu
    dm_image_cp[:,:] = commands_from_cosmic_gpu_image.reshape(24,24).T

    return -dm_image_torch


def flatten_dm():
    """
    Pipeline specific function to flatted the DM

    :return dm_image_torch:    2D image of the WFS measurement projected to DM voltages with flattened dm
    """
    commands_to_cosmic_gpu[:] = 0*dm_cp[dm_coords]
    commands_to_cosmic.notify()

    commands_from_cosmic_image.wait()
    # dm_image_cp[dm_coords] = commands_from_cosmic_gpu
    dm_image_cp[:,:] = commands_from_cosmic_gpu_image.reshape(24,24).T
    return -dm_image_torch



def train_dynamics(dynamics: EnsembleDynamicsFast, optimizer: SharedAdam, replay: EfficientExperienceReplay, replay_warmup: EfficientExperienceReplay, dyn_iters=5):
    """
    train_dynamics trains the dynamics model. It samples from the replay buffers (warm-up and new data), 
    forms the state variable by adding the past telemetry data, makes predictions and back-propagates 
    the loss value to optimize the dynamics model parameters.

    :param dynamics:      the dynamics model to be optimized
    :param optimizer:     optimizer for the gradient decent
    :param replay:        new data
    :param replay_warmup: warm-up data
    :prama dyn_iter:      number of gradient steps in training
    :return loss:         loss on the last forward pass
    """

    dynamics.train()

    for i in range(dyn_iters):
        optimizer.zero_grad()
        
        loss = 0

        for bs_model in dynamics.models:
            if torch.rand(1) > train_warmup_percent:
                sample = replay.sample_contiguous(n_history, episode_length, batch_size).to(device1)
            else:
                sample = replay_warmup.sample_contiguous(n_history, episode_length, batch_size).to(device1)

            states = sample.state()
            actions = sample.action()

            states = states.view(batch_size, n_history + 1, 1, *states.shape[1:])
            states_unfolded = states[:, :-1]
            
            actions_unfolded = actions.view(batch_size, n_history + 1, *actions.shape[1:])
            actions_unfolded = actions_unfolded[:, :-1].unsqueeze(2)
            
            next_states = states[:, -1]

            state = states_unfolded[:,-1].squeeze(2) 
            action = actions_unfolded[:,-1].squeeze(2)  

            history = torch.cat([states_unfolded[:,:-1].squeeze(2), actions_unfolded[:,:-1].squeeze(2)], dim=1)    

            pred = bs_model(torch.cat([history, state, action], dim=1))     

            assert pred.shape == next_states.shape
            pred_loss = (next_states - pred).pow(2).mean()
            
            loss += pred_loss

        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(dynamics.parameters(), 0.5)

        optimizer.step()

    return loss.item()

def train_policy(optimizer: SharedAdam, policy: ConvPolicyFastFast, dynamics: EnsembleDynamicsFast, replay: EfficientExperienceReplay, 
                    replay_warmup: EfficientExperienceReplay, pol_iters=5):
    """
    Train_policy trains the policy model, i.e., optimizes the policy parameters. 
    First, it samples initial states from the replay buffers and forms the state variable 
    by adding past telemetry. Next, the policy outputs (decides) the actions, and the 
    dynamics predicts the next state, the mean over the model in the ensemble.
    Then, it iterates the process over the planning horizon, collects the loss (-reward)
    at each time step, and backpropagate the cumulative loss to policy parameters.

    :param policy:        policy model to be optimized
    :param dynamics:      the dynamics model used for predicting the next states
    :param optimizer:     optimizer for the gradient decent
    :param replay:        new data
    :param replay_warmup: warm-up data
    :prama pol_iter:      number of gradient steps in training
    :return loss:         loss on the last forward pass
    """

    dynamics.train()
    policy.train()

    for p in dynamics.parameters():
        p.requires_grad_(False)

    for i in range(pol_iters):
        optimizer.zero_grad()
               
        if torch.rand(1) > train_warmup_percent:
            sample = replay.sample_contiguous(n_history, episode_length, batch_size).to(device1)
        else:
            sample = replay_warmup.sample_contiguous(n_history, episode_length, batch_size).to(device1)

        b = len(sample)
        
        states = sample.state()
        actions = sample.action()

        states_unfolded = states.view(batch_size, n_history + 1, 1, *states.shape[1:])
        states_unfolded = states_unfolded[:, :-1]
        
        actions_unfolded = actions.view(batch_size, n_history + 1, *actions.shape[1:])
        actions_unfolded = actions_unfolded[:, :-1].unsqueeze(2)        

        state = states_unfolded[:,-1].squeeze(2) 
        action = actions_unfolded[:,-1].squeeze(2)  

        # get past telemetry data
        past_obs = states_unfolded[:,:-1].squeeze(2)
        past_act = actions_unfolded[:,:-1].squeeze(2)  

        losses = torch.zeros(b, device=device1)

        for t in range(0, planning_horizon):

            history = torch.cat([past_obs, past_act], dim=1)

            action = policy(torch.cat([state, history], dim=1))

            next_state = dynamics(torch.cat([history, state, action], dim=1))           
            
            losses += loss_fn(next_state[:, 0], action)

            # roll history
            past_act = torch.cat([past_act[:,1:,:,:], action], dim = 1) 
            past_obs = torch.cat([past_obs[:,1:,:,:], state], dim = 1)

            next_state = torch.mean(next_state, dim = 1, keepdim = True)
            state = next_state

        loss = losses.mean()
        #print(loss.item())
        loss.backward()

        optimizer.step()

    for p in dynamics.parameters():
        p.requires_grad_(True)

    return loss.item()


def training_thread(start_q,dynamics_q, dynamics_optimizer_q, replay_q, replay_warmup_q, policy_optimizer_q, policy_q, finished_q, dyn_iters = 30, pol_iters = 10):
    """
    training_thread start the parallel training procedure, i.e., trains the dynamics and policy NNs in parallel to controller. 


    :param start_q:                   boolean for starting the training procedure. True when there is new data available
    :param dynamics_optimizer_q:      queue for the dynamics optimizer
    :param replay_q:                  queue for the data set to be trained on          

    """
    print("Training process started")
    while(1):
        start = start_q.get()
        if start:          
            start_time = time.time()
            dynamics = dynamics_q
            dynamics_optimizer = dynamics_optimizer_q#.get()
            replay = replay_q.get()
            replay_warmup = replay_warmup_q.get()
            policy_optimizer = policy_optimizer_q#.get()
            policy = policy_q

            dyn_loss = train_dynamics(dynamics, dynamics_optimizer, replay, replay_warmup, dyn_iters=config['training']['dynamics_grad_steps'])
            torch.cuda.synchronize(device="cuda:1")

            pol_loss = train_policy(policy_optimizer, policy, dynamics, replay, replay_warmup, pol_iters=config['training']['dynamics_grad_steps'])
            torch.cuda.synchronize(device="cuda:1")

            finished_q.put(True)
            start = False
            print(f'--------------------------------------------\n training ({time.time() - start_time:.2f}s). \n\t dyn:{1000*dyn_loss:.4f} pol:{1000*pol_loss:.4f} \n--------------------------------------------')



@torch.no_grad()
def run_episode_policy(past_obs, past_act, obs, replay, policy, sigma, episode_length): 
    """
    runs an episode on policy

    :param past_obs:      
    :param past_act:      
    :param obs:           
    :param replay:        
    :param policy:    
    :param sigma:
    :param episode_length:

    :return reward_sum:         
    :return past_act:        
    :return past_obs:         
    """

    policy.eval()
    reward_sum = 0 
       
    for t in range(episode_length):
           
        if integrator == True:
            action = gain*obs.unsqueeze(0).unsqueeze(0)
            time.sleep(0.0004)
        else:
            action = policy(torch.cat([obs.unsqueeze(0).unsqueeze(0), past_obs, past_act],dim = 1))           
        
        next_obs = step(action)
        
        # roll telemetry data with new data
        past_obs = torch.cat([past_obs[:,1:,:,:], obs.unsqueeze(0).unsqueeze(0)], dim = 1) 
        past_act = torch.cat([past_act[:,1:,:,:], action], dim = 1)
                
        reward_sum += torch.sum((obs.flatten()) ** 2)

        action_to_save = action.squeeze()
        replay.append(obs, action_to_save, next_obs)

        obs = next_obs        
    
    return reward_sum, past_obs, past_act, obs


def run_episode_warmup(replay, replay_warmup, sigma, episode_length, filter, xvalid, yvalid): 
    """
    runs an episode on integrator with added noise in control signals. Starts always with a flat mirror
         
    :param replay:        
    :param replay_warmup:    
    :param sigma:
    :param filter:
    :param xvalid:
    :param yvalid:

    :return reward_sum:         
    :return past_act:        
    :return past_obs:         
    """

    reward_sum = 0 
        
    obs = flatten_dm()
    past_obs = torch.zeros(1, (n_history-1), *obs.shape, device = device0).squeeze(2) # keep telemetry in memory for the next episode
    past_act = torch.zeros(1, (n_history-1), *obs.shape, device = device0).squeeze(2)
     
    for t in range(episode_length):
        
        action = gain * obs.unsqueeze(0).unsqueeze(0)
        action = action + sample_noise(sigma, filter, xvalid, yvalid)
               
        next_obs = step(action)

        past_obs = torch.cat([past_obs[:,1:,:,:], obs.unsqueeze(0).unsqueeze(0)], dim = 1) #roll telemetry
        past_act = torch.cat([past_act[:,1:,:,:], action], dim = 1)
               
        reward_sum += torch.sum((obs.flatten()) ** 2)

        action_to_save = action.squeeze()
        replay.append(obs, action_to_save, next_obs)
        
        if sigma >= min_sigma:
            replay_warmup.append(obs, action_to_save, next_obs)

        obs = next_obs
        
    return reward_sum, past_obs, past_act, obs

def loss_fn(state,action):
    "the loss function, i.e, negative reward, for policy training."

    return state.pow(2).mean() + loss_penalty*action.pow(2).mean()

def main():

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    savedir = f'/run/media/ghost/data/po4ao/{timestamp}'
    loaddir = f'/run/media/ghost/data/po4ao/calib/'   # copy the models and replay buffers you want use here!!
    
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    if not os.path.exists(loaddir):
        os.makedirs(loaddir)

    with open(os.path.join(savedir, f"config.txt"), 'w') as convert_file:
        convert_file.write(json.dumps(config))

    ctx = mp.get_context('spawn')
    start_q = ctx.Queue()
    replay_q = ctx.Queue()
    replay_warmup_q = ctx.Queue()
    finished_q = ctx.Queue()   
    start_q.put(False)
  
    KL_projection = m2v_data[:,:nmodes] @ np.linalg.pinv(m2v_data[:,:nmodes])
    KL_projection = torch.from_numpy(np.asarray(KL_projection)).float()

    mask = np.zeros((24,24))
    mask[dm_x, dm_y] = 1
    mask = np.array(mask, dtype = bool)

    xvalid0 = torch.from_numpy(dm_x).to(torch.int64).to(device0).squeeze()
    yvalid0 = torch.from_numpy(dm_y).to(torch.int64).to(device0).squeeze()

    xvalid1 = torch.from_numpy(dm_x).to(torch.int64).to(device1).squeeze()
    yvalid1 = torch.from_numpy(dm_y).to(torch.int64).to(device1).squeeze()

    replay = EfficientExperienceReplay((data_shape,data_shape), (data_shape,data_shape), replay_size* episode_length )
    replay_warmup = EfficientExperienceReplay((data_shape,data_shape), (data_shape,data_shape), warmup_memory* episode_length )
    
    dynamics = EnsembleDynamicsFast(mask, n_history).to(device1).share_memory()
    policy = ConvPolicyFastFast(xvalid1, yvalid1, KL_projection, n_history).to(device1).share_memory()
    policy_copy = ConvPolicyFastFast(xvalid0, yvalid0, KL_projection, n_history).to(device0).share_memory().eval()

    dynamics_optimizer = SharedAdam(dynamics.parameters())
    policy_optimizer = SharedAdam((policy.parameters()))

    dynamics_optimizer1 = optim.Adam(dynamics.parameters())
    policy_optimizer1 = optim.Adam(policy.parameters())

    sigma = initial_sigma
    rewards = torch.zeros(iters + warmup_episodes)

    training = False
    obs = flatten_dm()

    past_obs = torch.zeros(1, (n_history-1), *obs.shape, device = device0).squeeze(2) 
    past_act = torch.zeros(1, (n_history-1), *obs.shape, device = device0).squeeze(2)
    
    for i in range(warmup_episodes):
        start = time.time()
        reward_sum, past_obs, past_act, obs = run_episode_warmup(replay, replay_warmup, sigma, episode_length, KL_projection.to(device0), xvalid0, yvalid0)

        rewards[i] = reward_sum

        sigma -= (initial_sigma / (warmup_episodes/1))
        sigma = max(0, sigma)

        print(f'******************************************** \n Warm up {i} complete ({time.time() - start:.2f}s) \n\t reward:{reward_sum:.3f} \n********************************************')


    if config['save_and_load']['save_warmup_buffer']:
        torch.save(replay_warmup.states, os.path.join(savedir, f"states_warmup.pt"))
        torch.save(replay_warmup.next_states, os.path.join(savedir, f"next_states_warmup.pt"))
        torch.save(replay_warmup.actions, os.path.join(savedir, f"actions_warmup.pt")) 
        
        torch.save(replay.states, os.path.join(savedir, f"states.pt"))
        torch.save(replay.next_states, os.path.join(savedir, f"next_states.pt"))
        torch.save(replay.actions, os.path.join(savedir, f"actions.pt")) 
        
        print(f'--------------------------------------------\n warmup buffer saved! \n--------------------------------------------')    
    

    if config['save_and_load']['load_warmup_buffer']:
        replay_warmup.states = torch.load(os.path.join(loaddir, f"states_warmup.pt"))
        replay_warmup.next_states = torch.load(os.path.join(loaddir, f"next_states_warmup.pt"))
        replay_warmup.actions= torch.load(os.path.join(loaddir, f"actions_warmup.pt"))   

        replay_warmup.set_len(20* episode_length -1)

        replay.states = torch.load(os.path.join(loaddir, f"states.pt"))
        replay.next_states = torch.load(os.path.join(loaddir, f"next_states.pt"))
        replay.actions= torch.load(os.path.join(loaddir, f"actions.pt"))   

        replay.set_len(50* episode_length -1)
        print(f'--------------------------------------------\n warmup buffer loaded! \n--------------------------------------------')    
    
    if config['save_and_load']['load_models_pretrained']:
        dynamics.load_state_dict(torch.load( os.path.join(loaddir, f"dynamics_final.pt"),map_location=lambda storage, loc: storage))
        policy.load_state_dict(torch.load( os.path.join(loaddir, f"policy_final.pt"),map_location=lambda storage, loc: storage))
        policy_copy.load_state_dict(torch.load( os.path.join(loaddir, f"policy_final.pt"),map_location=lambda storage, loc: storage))
        
        print(f'--------------------------------------------\n Pretrained models loaded! \n--------------------------------------------')    
    
    elif replay_warmup.len > episode_length:
        # pretrain with warmup buffer
        start_time = time.time()
        dyn_loss = train_dynamics(dynamics, dynamics_optimizer1, replay_warmup, replay_warmup, dyn_iters=config['training']['dynamics_grad_steps_warmup'])
        torch.cuda.synchronize(device="cuda:1")

        pol_loss = train_policy(policy_optimizer1, policy, dynamics, replay_warmup, replay_warmup, pol_iters=config['training']['dynamics_grad_steps_warmup'])
        torch.cuda.synchronize(device="cuda:1")

        policy_copy.load_state_dict(policy.state_dict())

        print(f'--------------------------------------------\n Warmup training ({time.time() - start_time:.2f}s). \n\t dyn:{1000*dyn_loss:.4f} pol:{1000*pol_loss:.4f} \n--------------------------------------------')

    if config['save_and_load']['save_models_pretrained']:
        torch.save(dynamics.state_dict(), os.path.join(savedir, f"dynamics_pretrained.pt"))
        torch.save(policy.state_dict(), os.path.join(savedir, f"policy_pretrained.pt"))   

        print(f'--------------------------------------------\n Pretrained models saved! \n--------------------------------------------')



    replay_q.put(replay, False)
    replay_warmup_q.put(replay_warmup, False)

    if replay.len > episode_length and replay_warmup.len > episode_length:
        training_process = ctx.Process(target=training_thread, args=(start_q,dynamics, dynamics_optimizer, replay_q, replay_warmup_q, policy_optimizer, policy, finished_q, 50, 25,))
        training_process.start()
    else:
        print("Replay buffers empty --- training not started. Run warm up or load buffers")


    for p in policy_copy.parameters():
        p.grad = None

    obs = flatten_dm()
    for i in range(iters):

        start = time.time()
        
        reward_sum, past_obs, past_act, obs = run_episode_policy(past_obs, past_act, obs, replay, policy_copy, sigma, episode_length)
        
        rewards[i + warmup_episodes] = reward_sum
    
        try:
            training_finished = finished_q.get(False)
        except:
            training_finished = False

        if training_finished:
            training = False
            policy_copy.load_state_dict(policy.state_dict())
  
        if not training:
            replay_q.put(replay, False)
            replay_warmup_q.put(replay_warmup, False)
            start_q.put(True)
            training = True
            
        print(f'******************************************** \n Iteration {i} complete ({time.time() - start:.2f}s) \n\t reward:{reward_sum:.3f} \n********************************************')
        

    torch.save(rewards, os.path.join(savedir, f"rewards.pt"))
    torch.save(replay.states, os.path.join(savedir, f"states.pt"))
    torch.save(replay.actions, os.path.join(savedir, f"actions.pt")) 

    torch.save(dynamics.state_dict(), os.path.join(savedir, f"dynamics_final.pt"))
    torch.save(policy.state_dict(), os.path.join(savedir, f"policy_final.pt"))

    print("data saved!")
        
        
if __name__ ==  '__main__':
    main()
