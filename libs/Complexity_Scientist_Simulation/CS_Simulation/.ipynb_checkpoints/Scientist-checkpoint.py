#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author:   Munjung Kim
@Last Modified time: 2023-10-18 04:09:30

"""

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
from CS_Simulation.utils import DataGen
from CS_Simulation.utils import CustomDataset



class scientist:
    def __init__(self, experimentation_strategy,max_dimensions, att_dim, att_num=1,block_type = 'mlp',dropout = 0.1 ):
        
        
        self.data = {"Image" : [], "Label" : []}
        
        self.explanation = None
        
        self.local_rank = 6
        
        self.max_dimensions = max_dimensions # the number of node that the scientist can think 
        self.minimum_exploration = 0.1

        self.experimentation_strategy = experimentation_strategy
        
        
        
        self.D_in_enc = 3 # x, y, c
        self.D_hidden_enc = 128
        self.D_out_enc = int(att_dim * att_num)
        self.D_att = int(att_dim)
        self.D_att_num = int(att_num)
        self.D_in_dec = self.D_out_enc * (att_num+1)
        self.D_in_dec = self.D_out_enc * 2

        self.D_hidden_dec = 64
        self.D_out_dec = 1 # possibility of 1
        self.D_agent = max_dimensions
        
        
        self.block_type = block_type
        
        self.dropout = dropout
        
        
        
        self.batch_size = 16
        
    def make_observation(self, env, scientist2=None):
        
        # env = system
        generator = DataGen(env)
        
        # what to measure if there is no data yet
        if len(self.data['Image']) < 100 or np.random.random() < self.minimum_exploration or self.experimentation_strategy == "random": # exploration here!
            preferred_state = np.random.randint(0, 2, self.max_dimensions)
            env.set_initial_state(preferred_state)
            current_observation = generator.run( total_size = 1)
            self.data["Image"].extend(current_observation["Image"])
            self.data["Label"].extend(current_observation["Label"])
            
            
        elif self.experimentation_strategy == "safe": # exploring the safe area
            
            
            
            test_accuracy = np.array(self.evaluate_on_collected_data())
            target_observation_idx = np.argmax(test_accuracy)

            
            best_explained_data = self.data["Image"][target_observation_idx]
            
            
            
            
            random_change = np.random.randint(0,196,1)
         
            
            if best_explained_data[random_change][0][2] ==1:
                best_explained_data[random_change][0][2] = 0
                
            elif best_explained_data[random_change][0][2]==0:
                best_explained_data[random_change][0][2] = 1
                
        
                
            preferred_state = best_explained_data[:,2]
    

            env.set_initial_state(preferred_state)
            current_observation = generator.run( total_size = 1)
            self.data["Image"].extend(current_observation["Image"])
            self.data["Label"].extend(current_observation["Label"])
            
            
        elif self.experimentation_strategy == "risky": # exploring the safe area
            
            test_accuracy = np.array(self.evaluate_on_collected_data())
            target_observation_idx = np.argmin(test_accuracy)

            
            best_explained_data = self.data["Image"][target_observation_idx]
            
            
            
            
            random_change = np.random.randint(0,196,1)
         
            
            if best_explained_data[random_change][0][2] ==1:
                best_explained_data[random_change][0][2] = 0
                
            elif best_explained_data[random_change][0][2]==0:
                best_explained_data[random_change][0][2] = 1
                
        
                
            preferred_state = best_explained_data[:,2]
    

            env.set_initial_state(preferred_state)
            current_observation = generator.run( total_size = 1)
            self.data["Image"].extend(current_observation["Image"])
            self.data["Label"].extend(current_observation["Label"])
            
            
    
        return current_observation
    
    
    def initialize_explanation(self,att_type = 'gat'):
        
        
        if self.block_type == 'mlp':

            cfg_enc = [self.D_in_enc, self.D_hidden_enc, self.D_hidden_enc, self.D_out_enc]
            cfg_att = [self.D_att*2, 8, 8, 1]
            cfg_dec = [self.D_in_dec, self.D_hidden_dec, self.D_hidden_dec, self.D_out_dec]

        elif self.block_type == 'res':

            cfg_enc = [self.D_in_enc, self.D_hidden_enc, self.D_out_enc]
            cfg_att = [self.D_att * 2, 8, 1]
            cfg_dec = [self.D_in_dec, self.D_hidden_dec,self. D_out_dec]
            
        model = Module_GAT_DET(cfg_enc, cfg_att, cfg_dec, self.D_att, self.D_att_num, self.D_agent, self.block_type,  att_type, self.dropout).cuda()
        # torch.cuda.set_device(self.local_rank)
        # rank = int(os.environ['RANK'])
        # torch.distributed.init_process_group(backend='nccl')
        model = torch.nn.DataParallel(model, device_ids=[self.local_rank],output_device=self.local_rank)

            
        self.explanation = model
        
        
    def update_explanation(self,lr,weight_decay,epochs):
        train_dataset = CustomDataset(self.data)
        if len(self.data["Image"]) < 16:
            train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False, pin_memory=True,
                            num_workers=10)
        else:
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True,
                            num_workers=10)
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(
            self.explanation.parameters(), lr, weight_decay=weight_decay
        )
        scheduler = RLRP(optimizer, 'min', factor=0.7, patience=50, min_lr=5e-6, verbose=1)
        cudnn.benchmark = True
        
        with torch.autograd.detect_anomaly():
            for epoch in range(epochs):
                if epoch%10==0:
                    print("============== Epoch {} =============".format(epoch))
                    
                train_loss, train_acc = self.train(train_loader, self.explanation, criterion, optimizer, epoch, scheduler)

                print(self.explanation.parameters())
                print("[Epoch {}] Train Loss : {} / Train Acc : {}".format(str(epoch), str(train_loss), str(train_acc*100)))



    def evaluate_on_collected_data(self):
        train_dataset = CustomDataset(self.data)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, pin_memory=True,
                            num_workers=10)
        
        criterion = nn.BCEWithLogitsLoss()
        
        train_loss, train_acc = self.test(train_loader, self.explanation, criterion)
        
        
        
        return train_acc
    
    
    
    def train(self,train_loader, model, criterion, optimizer, epoch, scheduler):

        train_losses = AverageMeter('Loss', ':.4e')
        train_acc = AverageMeter('Accuracy', ':.4e')
        model.train()

        for data, labels in train_loader:
            
            data = data.cuda(self.local_rank)
            labels = labels.cuda(self.local_rank).squeeze()
           
            output = model(data).squeeze(-1)
            output = output.reshape(-1, 14, 14)[:, 1:-1, 1:-1].reshape(-1, 144)
            labels = labels.reshape(-1, 14, 14)[:, 1:-1, 1:-1].reshape(-1, 144) # cutting out the corners
            train_loss = criterion(output, labels) # reduction = sum
            train_losses.update(train_loss.item(), np.prod(output.shape))
            train_loss = train_loss/np.prod(output.shape) # reverting to mean

            x = DCN(torch.sigmoid(output))
            x = np.where(x>=0.5, 1, 0)
            answer_ratio = np.mean(np.where(x==DCN(labels), 1, 0))
            train_acc.update(answer_ratio, np.prod(output.shape))

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        scheduler.step(train_losses.avg, epoch)
        return train_losses.avg, train_acc.avg
    
    
    def test(self,test_loader, model, criterion):

        test_losses = []
        test_acc = []
        model.eval()

        with torch.no_grad():
            for data, labels in test_loader:
                data = data.cuda(self.local_rank)
                labels = labels.cuda(self.local_rank).squeeze()
                output = model(data).squeeze(-1)
                output = output.reshape(-1, 14, 14)[:, 1:-1, 1:-1].reshape(-1, 144)
                labels = labels.reshape(-1, 14, 14)[:, 1:-1, 1:-1].reshape(-1, 144) # cutting out the corners
                test_loss = criterion(output, labels) # reduction = sum
                test_losses.append(test_loss)

                x = DCN(torch.sigmoid(output))
                x = np.where(x>=0.5, 1, 0)
                answer_ratio = np.mean(np.where(x==DCN(labels), 1, 0))
                test_acc.append(answer_ratio)

        return test_losses, test_acc
    
    
