from copy import deepcopy
import argparse
import os
import random
import shutil
import time
import warnings
import sys
import numpy as np
import scipy as sp
import math

from time import sleep
            
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed as dist
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader # (testset, batch_size=4,shuffle=False, num_workers=4)
from torch.optim.lr_scheduler import ReduceLROnPlateau as RLRP
from torch.nn.init import xavier_normal
from torch.nn.parameter import Parameter

import pickle
import importlib
import itertools
import random
from datetime import datetime
from collections import OrderedDict
from copy import deepcopy

import CS_Simulation.src.DataStructure as DS
from CS_Simulation.src.utils import *
from CS_Simulation.src.system import *
from CS_Simulation.src.model import *

class CA():
    '''
    Cellular Automata dataset
    
    '''
    
    def __init__(self):
        self.name = 'CA'
        self.rule_name = 'Agent_'+str(self.name)
        self.state_num = 3 # x, y, c (cell state)

    def assign_pp(self, plugin_parameters):
        self.pp = plugin_parameters
        assert 'side_length' in plugin_parameters
        self.side_length = self.pp['side_length']
        self.input_length = self.side_length**2
        assert 'rule' in plugin_parameters
        self.rule = self.pp['rule']
        self.rule_name += '_a'+str(self.rule['alive'])+'_d'+str(self.rule['dead'])
    
    def neighbor_fn_data(self, data):
        size = np.prod(data.shape) 
        side = int(np.sqrt(size)) 
        M, N = side, side
        cells = list(range(size))
        idx, idy = np.unravel_index(cells, shape=(M, N))
    
        neigh_idx = np.vstack((idx-1, idx+1, idx, idx, idx-1, idx-1, idx+1, idx+1))
        neigh_idy = np.vstack((idy, idy, idy-1, idy+1, idy-1, idy+1, idy-1, idy+1)) # ←→↑↓↖↙↗↘

        neighbor = np.ravel_multi_index((neigh_idx, neigh_idy), mode = 'warp', dims=(M,N)).T
        return np.concatenate((neighbor, np.expand_dims(np.arange(size), 1)), axis = 1)

    def __next__(self):
        L = self.side_length
        while(True):
            system = np.random.randint(0, 2, L**2)
     
            system_now = deepcopy(system)
            answer = []
            neighbor = self.neighbor_fn_data(system)
            system_next = deepcopy(system_now)
            
            for i in range(L**2):
                n_list = system_now[neighbor][i]
                if system_now[i] == 1:
                    if np.sum(n_list[:-1]) not in self.rule['alive']:
                        system_next[i] = 0
                elif system_now[i] == 0 :
                    if np.sum(n_list[:-1]) in self.rule['dead']:
                        system_next[i] = 1
                else:
                    print('ERROR')
            
            cells = list(range(self.input_length))
            idx, idy = np.unravel_index(cells, shape=(self.side_length, self.side_length))
            system_now = np.stack((idx, idy, system_now), axis = 1)
            
            return np.array(system_now), np.array(system_next)
        
    def test_figure(self, fig, model, device):
        x = ["".join(seq) for seq in itertools.product("01", repeat=8)]
        t1, t2 = [], []
        for i in x:
            t1.append(i+'0')
            t2.append(i+'1')

        t = t1 + t2
        expected = []
        predicted = []

        for i in t1:
            if np.sum(list(map(int, list(i)))[:-1]) in system.rule['dead']:
                expected.append(1)
            else:
                expected.append(0)

        for i in t2:
            if np.sum(list(map(int, list(i)))[:-1]) in system.rule['alive']:
                expected.append(1)
            else:
                expected.append(0)

        for i in t:
            predicted.append(float(model(torch.Tensor(list(map(int, list(i)))).to(device), 1)))

        ax = fig.add_subplot(2,2,(3,4))
        ax.plot(expected, color = 'r', linewidth = 2, label = 'expected', alpha = 0.5)
        ax.scatter(list(range(512)), predicted, color = 'b', linewidth = 1, label = 'predicted')
        ax.set_xlabel('Case number', fontsize = 15)
        ax.set_ylabel('Result', fontsize = 15)
        ax.set_ylim(-1,2)
        return ax
    
    
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data["Image"])

    def __getitem__(self, idx):
        image = torch.tensor(self.data['Image'][idx]).to(torch.float32)  
        label = torch.tensor(self.data['Label'][idx]).to(torch.float32)  
        return image,  label
    
    
class DataGen():
    def __init__(self, system):
        self.system = system
        pass
    
    def run(self, total_size):

        train_image = []
        train_label = []
        
        for i in range(total_size):
            data, answer = next(self.system)
            
            train_image.append(data.astype(float))
            train_label.append(answer)

        train_output = {'Image':train_image, 'Label':train_label}
        
     
        
        return train_output
