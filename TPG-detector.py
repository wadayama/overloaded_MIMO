# This code is an implementation of TPG-detector for overloaded MIMO in PyTorch.
# The details of the algorithm can be found in the paper:
# Satoshi Takabe, Masayuki Imanishi, Tadashi Wadayama, Kazunori Hayashi
# "Trainable Projected Gradient Detector for Massive Overloaded MIMO Channels: Data-driven Tuning Approach", arXiv:1806.10827.


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import math
import time
import os
import numpy as np
from numpy.random import *
import sys


# global variables

GPU = True
File = True

if GPU == True:
    cuda = torch.device('cuda')

args = sys.argv


# For a Real number system
N = 300  # length of a transmit signal vector
NM_ratio = 0.64 # M/N ratio
M = int(N * NM_ratio) # length of a receive signal vector
batch_size = 1250  # mini-batch size
num_batch = 1000  # number of mini-batches in a generation
num_layers = 2  # number of layers

snr = 15.0  # SNR per receive antenna [dB]
sigma2 = (N/math.pow(10,snr/10.0))/2.0
sigma_std = math.sqrt(sigma2) # SD for w

adam_lr = 0.025 # learning_rate for Adam
test_itr = 100  # number of iterator for evaluate

file_name = "N"+str(N)+"_snr"+str(snr) #File name for write parameter
file_name2 = "N"+str(N)

# channel_matrix generator
H_re = torch.normal(0.0, std=math.sqrt(0.5) * torch.ones((int)(M/2), (int)(N/2)))
H_im = torch.normal(0.0, std=math.sqrt(0.5) * torch.ones((int)(M/2), (int)(N/2)))
H = torch.cat((torch.cat((H_re,H_im),0),torch.cat((-1*H_im,H_re),0)),1)
Ht = H.t()
Ht = Ht.cuda()

# change global_H
def H_change():
    global H_re,H_im,H,Ht

    H_re = torch.normal(0.0, std=math.sqrt(0.5) * torch.ones((int)(M/2), (int)(N/2)))
    H_im = torch.normal(0.0, std=math.sqrt(0.5) * torch.ones((int)(M/2), (int)(N/2)))  # sensing matrix
    H = torch.cat((torch.cat((H_re,H_im),0),torch.cat((-1*H_im,H_re),0)),1)
    H = H.cuda()
    Ht = H.t()
    Ht = Ht.cuda()

def write_param():
    f = open(file_name,"a")
    f.write('\nParameter:\n')
    f.write('--------------------------------------------\n')
    f.write('(N,M)=:({0},{1})\n'.format(N,M))
    f.write('adam_lr:{0}\n'.format(adam_lr))
    f.write('num_layers:{0}\n'.format(num_layers))
    f.write('batch_size:{0}\n'.format(batch_size))
    f.write('num_batch:{0}\n'.format(num_batch))
    f.write('--------------------------------------------\n')
    f.close()

def write_file(BER,gamma,theta,alpha,last_print):
    f = open(file_name, "a")
    f2 = open(file_name2, "a")
    write_param()
    f.write("---------BER---------\n")
    f.write(str(BER))
    f.write("\n---------gamma---------\n")    
    f.write(str(gamma))
    f.write("\n---------theta---------\n")        
    f.write(str(theta))
    f.write("\n---------alpha---------\n")        
    f.write(str(alpha))
    f.write("\n---------last_print---------\n")        
    f.write(str(last_print))
    f.close()
    f2.write(str(BER)+",")
    f2.close()
    # f = open(file_name, "a")
    # f.write(output_str)
    # f.close()
    return 0

# detection for NaN
def isnan(x):
    return x != x

# mini-batch generator
def generate_batch():
    return 1.0 - 2.0*torch.bernoulli(0.5* torch.ones(batch_size, N))

# definition of TPG-detector network
class TPG_NET(nn.Module):
    def __init__(self):
        super(TPG_NET, self).__init__()
        self.gamma = nn.Parameter(torch.normal(1.0, 0.1 * torch.ones(num_layers)))
        self.theta = nn.Parameter(torch.normal(1.0, 0.1 * torch.ones(num_layers)))
        self.alpha = nn.Parameter(torch.abs(torch.normal(0.0, 0.01 * torch.ones(1))))


    def shrinkage_function(self, y, tau2):  # shrinkage_function
        return torch.tanh(y/tau2)

    def forward(self, x, s, max_itr):  # TPG-detector network
        alpha_I = self.alpha[0]*torch.eye(M).cuda()
        W = Ht.mm((H.mm(Ht) + alpha_I).inverse()) #LMMSE-like matrix
        Wt= W.t()
        y = x.mm(Ht) + torch.normal(0.0, sigma_std*torch.ones(batch_size, M)).cuda()
        for i in range(max_itr):
            t = y - s.mm(Ht)
            tau2 = torch.abs(self.theta[i])
            r = s + t.mm(Wt)*self.gamma[i]
            s = self.shrinkage_function(r, tau2)
        return s


def eval(network,t): #calculate BER
    s_zero = torch.zeros(batch_size, N).cuda()  # initial value
    accuracy,num_err = 0.0,0.0
    H_change()
    for i in range(test_itr):
        x = generate_batch().cuda()
        x_hat = network(x, s_zero, t+1).cuda()
        if isnan(x_hat).any():
            print("Nan")
            continue

        err = x * torch.sign(x_hat)
        num_err += torch.nonzero(F.relu(err)).size(0)

    accuracy = num_err/(test_itr*batch_size*N)
    BER = 1 - accuracy
    print('({0}) BER:{1:6.6f}'.format(t + 1, BER))

    return BER

def main():
    network = TPG_NET().cuda()  # generating an instance of TPG-detector
    s_zero = torch.zeros(batch_size, N).cuda()  # s_0 = 0

    torch.manual_seed(1)

    start = time.time()

    last_print = []
    last_print_BER = []
    last_print_gamma = ""
    last_print_theta = ""
    last_print_alpha = ""
    for t in range(num_layers):
        for i in range(num_batch):
            H_change()
            opt = optim.Adam(network.parameters(), lr=adam_lr)  # setting for optimizer
            x = generate_batch().cuda()

            opt.zero_grad()
            x_hat = network(x, s_zero, t+1).cuda()

            loss = F.mse_loss(x_hat, x) #squared_loss

            if i % 100 == 0:
                print('loss:{0}'.format(loss.data)) #print_loss

            loss.backward()

            grads = [param.grad for param in network.parameters()]

            grads_gamma = grads[0]
            grads_theta = grads[1]
            grads_alpha = grads[2]

            if isnan(grads_gamma).any() and isnan(grads_theta).any() and isnan(grads_alpha).any():  # avoiding NaN in gradients
                print("NaN_grad")
                continue
            opt.step()

        BER = eval(network,t)


        ## to save parameter
        param_set = [param for param in network.parameters()]
        gamma_set = param_set[0]
        theta_set = param_set[1]
        alpha_set = param_set[2]

        gamma_output = ""
        theta_output = ""
        alpha_output = ""
        for i in gamma_set.data.cpu().numpy():
            gamma_output = gamma_output + str(i) + ","
        for i in theta_set.data.cpu().numpy():
            theta_output = theta_output + str(i) + ","
        for i in alpha_set.data.cpu().numpy():
            alpha_output = alpha_output + str(i) + ","

        gamma_output = gamma_output[:-1]
        gamma_output = "[" + gamma_output + "]"

        theta_output = theta_output[:-1]
        theta_output = "[" + theta_output + "]"

        alpha_output = alpha_output[:-1]
        alpha_output = "[" + alpha_output + "]"


        last_print_gamma = last_print_gamma + gamma_output + ","
        last_print_theta = last_print_theta + theta_output + ","
        last_print_alpha = last_print_alpha + alpha_output + ","

        last_print.append('({0}) BER:{1:6.6f}\n'.format(t + 1, BER))
        last_print_BER.append(BER)


    last_print_gamma = last_print_gamma[:-1]
    last_print_theta = last_print_theta[:-1]
    last_print_alpha = last_print_alpha[:-1]
    if File == True:
        write_file(last_print_BER,last_print_gamma,last_print_theta,last_print_alpha,last_print)
    print(last_print_BER)

    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    print(last_print)


if __name__ == '__main__':
    main()
