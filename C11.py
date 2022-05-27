#!/usr/bin/python3

# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, Juraj Holas, Peter Gergel, Endre Hamerlik, Štefan Pócoš, Iveta Bečková 2017-2022

import numpy as np
import random

from hopfield import Hopfield
from util import *


## 1. Load data

# dataset = 'small.txt'
dataset = 'medium.txt'

patterns, dim = prepare_data(dataset)



# 2. Select a subset of patterns (optional)

# patterns = patterns[:]
# count = len(patterns)



## 3. Train the model

# Plot input patterns (comment out when it starts to annoy you)
plot_states(patterns, 'Training patterns')

model = Hopfield(dim)
model.train(patterns)



## 4. Generate an input

# a) random binary pattern
input = 2*(np.random.rand(dim) > 0.5)-1

# # b) corrupted input
# input = random.choice(patterns) # select random input
# ind = np.random.choice(dim, 7) # select some indices
# input[ind] *= -1 # flip signs



## 5. Run the model

plot_states([input], 'Random/corrupted input', block=False)




#  asynchronous stochastic vs. deterministic
model.run_async(input, eps=5, rows=2, row=1, trace=True)
model.run_async(input, eps=5, rows=2, row=2, beta_s=0.1, beta_f=10, trace=True)
