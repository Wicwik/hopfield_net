#!/usr/bin/python3

import numpy as np
import random

from hopfield import Hopfield
from util import *

from tqdm import tqdm

def overlap_percetages(states, patterns, dim):
    '''
    Calculate overlap percentages, how many neurons are equal to coresponding pattern
    If neurons are inverted to the pattern, it wont count as overlapping
    '''
    overlaps = []

    # for each pattern create an array of overlap percentages for each state
    for pattern in patterns:
        o_state = []
        for state in states:
            o_state.append((np.array(state) == np.array(pattern)).sum()/dim*100)

        overlaps.append(o_state)

    return overlaps

def noise_correcting_test(model, ks, patterns, dim, names, show_steps=5):
    '''
    Add k noise to each pattern to be reconstructed by hopfield network. 
    For each k and pattern plot energy and overlap percentages.
    '''
    for p_idx, pattern in enumerate(patterns):
        print('Corrupting pattern {}'.format(names[p_idx]))

        # construct nice plots
        fig = plt.figure(99, figsize=(16, 8))
        X = [ (1,2,1), (2,4,3), (2,4,4), (2,4,7), (2,4,8) ]
        axes = []

        for nrows, ncols, plot_number in X:
            axes.append(plt.subplot(nrows, ncols, plot_number))

        fig.suptitle('Letter {}'.format(names[p_idx]), fontsize=16)

        for k in ks:
            input_pattern = pattern.copy()
            ind = np.random.choice(dim, k) 
            input_pattern[ind] *= -1 

            S, E = model.run_sync(input_pattern)

            # to make our plots more pretty, we reapeat last value until show_steps
            if len(E) < show_steps:
                E += [E[-1]]*(show_steps - len(E))
            
            axes[0].plot(E, label='Noise {}'.format(k))
            axes[0].legend()

            axes[0].set_xlabel('Number of steps')
            axes[0].set_ylabel('Energy')

            overlaps = overlap_percetages(S, patterns, dim)

            for o_idx, overlap in enumerate(overlaps):
                if len(overlap) < show_steps:
                    overlap += [overlap[-1]]*(show_steps - len(overlap))

                axes[o_idx+1].plot(overlap)
                axes[o_idx+1].set_title('Overlap with {}'.format(names[o_idx]))
                axes[o_idx+1].set_ylim([-10,110])

        plt.show()


def network_dynamics_test(model, n_patterns, n_most_common=10, plot=True):
    '''
    Run network for n random patterns to reconstruct learned patterns
    '''
    model.reset_stats()

    input_patterns = 2*(np.random.rand(n_patterns, dim) > 0.5)-1

    for input_pattern in tqdm(input_patterns):
        model.run_sync(input_pattern)

    print('True attractors: {}, False attractors: {}, Limit cycles: {}'.format(model.true_attractors, model.false_attractors, model.cycles))

    sorted_counts = sorted(model.states_counter, key=lambda tup: tup[1], reverse=True)
    most_common = list(map(lambda x: x[0], sorted_counts[:n_most_common]))
    # print(sum(map(lambda x: x[1], sorted_counts))) # sanity check

    if plot:
        plot_states(most_common, 'Most common states')


def more_patterns_test(n_train_patterns, n_generated_patterns, dynamics_plot=True):
    '''
    Load 8 patterns dataset and run dynamics test for 1 to 8 different patterns
    '''
    recall_success_rates = []


    for idx, n in enumerate(n_train_patterns):
        dataset = 'letters_large.txt'
        patterns, dim = prepare_data_from_nums(dataset)

        if idx == 0:
            plot_states(patterns, 'Training patterns - large dataset')

        patterns = patterns[:n]

        print('Training on {} pattern/s:'.format(n))

        if dynamics_plot:
            plot_states(patterns, 'Current training patterns')

        model = Hopfield(dim)
        model.train(patterns)

        network_dynamics_test(model, n_generated_patterns, plot=dynamics_plot)
        recall_success_rates.append(model.true_attractors/n_generated_patterns*100)

    fig = plt.figure(98, figsize=(16, 8))

    plt.title('Effect of storing more 5x7 patterns')
    plt.plot(n_train_patterns, recall_success_rates)
    plt.xlabel('Number of patterns')
    plt.ylabel('Success rate')
    plt.show()


## Load data
dataset = 'letters.txt'
patterns, dim = prepare_data_from_nums(dataset)
names = ['X', 'H', 'O', 'Z']

# Plot input patterns (comment out when it starts to annoy you)
plot_states(patterns, 'Training patterns - normal dataset')


## Train the model
model = Hopfield(dim)
model.train(patterns)

# Test model
ks = [0, 7, 14, 21]
noise_correcting_test(model, ks, patterns, dim, names)

n_generated_patterns = 10000
network_dynamics_test(model, n_generated_patterns)

n_train_patterns = [1, 2, 3, 4, 5, 6, 7, 8]
more_patterns_test(n_train_patterns, n_generated_patterns, dynamics_plot=False)
