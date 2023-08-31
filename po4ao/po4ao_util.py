import torch
import numpy as np 
from torch import nn

class EfficientExperienceReplay():

    def __init__(self, state_shape, action_shape, max_size=100000, warmup_memory = 0):
        self.max_size = max_size

        self.states = torch.empty(max_size, *state_shape).to("cuda:0")#.share_memory_()
        self.next_states = torch.empty(max_size, *state_shape).to("cuda:0")#.share_memory_()
        self.actions = torch.empty(max_size, *action_shape).to("cuda:0")#.share_memory_()

        self.len   = 0
        self.index_write = 0
        self.warmup_memory = warmup_memory

    def add(self, replay):
        cur_len = self.len
        new_len = self.len + len(replay)
        
        if isinstance(replay, EfficientExperienceReplay):
            replay = ReplaySample(replay.states[:len(replay)], replay.actions[:len(replay)], replay.rewards[:len(replay)], replay.next_states[:len(replay)])
        
        self.states[cur_len:new_len] = replay.state()
        self.next_states[cur_len:new_len] = replay.next_state()
        self.actions[cur_len:new_len] = replay.action()

        self.len = new_len

    def __add__(self, replay):
        self.add(replay)
        return self
    
    def append(self, obs, action, next_obs):

        if isinstance(obs, np.ndarray):
            raise 'should be torch'

        self.states[self.index_write] = obs
        self.next_states[self.index_write] = next_obs
        self.actions[self.index_write] = action

        self.index_write += 1

        if self.len < self.max_size:
            self.len += 1

        if self.index_write == self.max_size:
            print('Experience Replay Full')
            self.index_write = self.warmup_memory


    def sample_contiguous(self, horizon, max_ts, batch_size=32):    
        inds = torch.randint(0, max_ts - (horizon + 1), size=(batch_size, ))
        inds += torch.randint(0, len(self) // max_ts, size=(batch_size, )) * max_ts
        
        indices = torch.cat([torch.arange(ind, ind + horizon + 1) for ind in inds])
        # TODO check correct
        #indices = torch.from_numpy(vrange(inds.numpy(), np.ones_like(inds) * horizon + 1))
        
        return ReplaySample(self.states[indices], self.actions[indices], self.next_states[indices])

    
    def next_state(self):
        return self.next_states[:self.len]
    
    def state(self):
        return self.states[:self.len]

    def action(self):
        return self.actions[:self.len]

    def __len__(self):
        return self.len
        
    def set_len(self,index):
        self.len = index
        self.index_write = index

    def sample(self, size=512):
        inds = torch.randperm(self.len)[:size]
        return ReplaySample(self.states[inds], self.actions[inds], self.next_states[inds])  

    def clear(self):
        self.len = 0

class ReplaySample():
    def __init__(self, states, actions, next_states):
        self.states = states
        self.next_states = next_states
        self.actions = actions

    def state(self):
        return self.states
    
    def prev_action(self):
        return self.prev_actions

    def next_state(self):
        return self.next_states 

    def action(self):
        return self.actions

    def __len__(self):
        return len(self.states)

    def to(self, device):
        self.states = self.states.to(device)
        self.next_states = self.next_states.to(device)
        self.actions = self.actions.to(device)
        return self

import contextlib
import os

@contextlib.contextmanager
def stdchannel_redirected(stdchannel, dest_filename):
    """
    A context manager to temporarily redirect stdout or stderr

    e.g.:


    with stdchannel_redirected(sys.stderr, os.devnull):
        if compiler.has_function('clock_gettime', libraries=['rt']):
            libraries.append('rt')
    """

    try:
        oldstdchannel = os.dup(stdchannel.fileno())
        dest_file = open(dest_filename, 'w')
        os.dup2(dest_file.fileno(), stdchannel.fileno())

        yield
    finally:
        if oldstdchannel is not None:
            os.dup2(oldstdchannel, stdchannel.fileno())
        if dest_file is not None:
            dest_file.close()

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()