# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, Juraj Holas, Peter Gergel, Endre Hamerlik, Štefan Pócoš, Iveta Bečková 2017-2022
# Edited by Robert Belanec for third project purposes

import itertools
import numpy as np

from util import *

eps_hard_limit = 100


class Hopfield():

    def __init__(self, dim):
        self.dim  = dim
        self.beta = None  # if beta is None: deterministic run, else stochastic run
        
        # statistics counters
        self.true_attractors = 0
        self.false_attractors = 0
        self.cycles = 0

        # list of tuples (state, counter) to get 10 most commont final or cycle states
        self.states_counter = []


    def reset_stats(self):
        '''
        Reset counters
        '''
        self.true_attractors = 0
        self.false_attractors = 0
        self.cycles = 0
        self.states_counter = []

    def train(self, patterns):
        '''
        Compute weight matrix analytically
        '''
        self.patterns = patterns
        self.W  = np.zeros((self.dim, self.dim))
        for p in self.patterns:
            self.W += 1/len(self.patterns)*np.outer(p,p)
            np.fill_diagonal(self.W, 0)

    def energy(self, s):
        '''
        Compute energy for given state
        '''
        outer_s = np.outer(s,s)
        np.fill_diagonal(outer_s, 0)
        return -1/2*np.sum(self.W*outer_s)


    # asynchronous dynamics
    def forward_one_neuron(self, s, neuron_index):
        '''
        Perform forward pass in asynchronous dynamics - compute output of selected neuron
        '''
        net = self.W[neuron_index]@s

        if self.beta is None:
            # Deterministic transition
            return -1 if not np.sign(net) else np.sign(net)
        else:
            # Stochastic transition (Output for each neuron is either -1 or 1!)
            prob = 1/(1 + np.exp(-self.beta*net))
            return -1 if np.random.rand() >= prob else 1


    def forward_all_neurons(self, s):
        '''
        Perform forward pass in synchronous dynamics - compute output of all neurons
        '''
        net = self.W@s

        if self.beta is None:
            # Deterministic transition
            return np.sign(net) - (net == 0) # if net[i] == 0, be negative 
        else:
            # Stochastic transition (Output for each neuron is either -1 or 1!)
            prob = 1/(1 + np.exp(-self.beta*net))
            return 2*(prob >= np.random.rand(self.dim))-1

    def update_final_state_counter(self, s):
        '''
        Update state counter that succesfully converged to false or true attractor
        '''
        for idx, c in enumerate(self.states_counter):
            if np.array_equal(c[0], s) or np.array_equal(c[0], s*-1):
                tmp = list(self.states_counter[idx])
                tmp[1] += 1
                self.states_counter[idx] = tuple(tmp)
                break
        else:
            self.states_counter.append([s, 1])

    def update_cycle_state_counter(self, S, cycle_start):
        '''
        Update state counter that ended up stucked in a cycle
        '''
        for idx, c in enumerate(self.states_counter):
            for s_idx in range(cycle_start, len(S)):
                # check each state from begging to the end of the cycle
                if np.array_equal(c[0], S[s_idx]) or np.array_equal(c[0], S[s_idx]*-1):
                    tmp = list(self.states_counter[idx])
                    tmp[1] += 1
                    self.states_counter[idx] = tuple(tmp)
                    break
            else:
                continue
            break
        else:
            # if that state is not in our counter list, add it
            self.states_counter.append([S[cycle_start], 1])

    def is_true_attractor(self, s):
        '''
        Check if state or inverted state is equal to one of the patterns
        '''
        for p in self.patterns:
            if np.array_equal(p, s) or np.array_equal(p, s*-1):
                return True
                
        return False

    def contains_even_length_cycle(self, S):
        '''
        Check if sequence of states contains loop with even lenght
        '''
        last_idx = len(S)-1
        last_state = S[last_idx]

        for s_idx, state in enumerate(S):
            if np.array_equal(state, last_state) and s_idx != last_idx and ((last_idx - s_idx) % 2) == 0:
                return True, s_idx

        return False, -1

    def run_sync(self, x, beta_s=None, beta_f=None, plot=False, prints=False):
        '''
        Run model in synchronous dynamics. One input vector x will produce
        series of outputs (states) s_t.
        '''
        s = x.copy()
        e = self.energy(s)
        S = [s]
        E = [e]

        title = 'Running: synchronous {}'.format('stochastic' if beta_s is not None else 'deterministic')

        for ep in range(eps_hard_limit): # "enless" loop
            ## Set beta for this episode
            if beta_s is None:
                # Deterministic -> no beta
                self.beta = None
                if prints:
                    print('Ep {:2d}/{:2d}:  deterministic'.format(ep+1, eps_hard_limit))
            else:
                # Stochastic -> schedule for beta (or temperature)
                self.beta = beta_s * ( (beta_f/beta_s) ** (ep/(eps_hard_limit-1)))
                if prints:
                    print('Ep {:2d}/{:2d}:  stochastic, beta = 1/T = {:7.4f}'.format(ep+1, eps_hard_limit, self.beta))

            ## Compute new state for all neurons
            s = self.forward_all_neurons(s)
            e = self.energy(s)

            S.append(s)
            E.append(e)

            if self.beta is None:
                if np.all(self.forward_all_neurons(s) == s):
                    if prints:
                        print('Found fixed point. Reached local/global minimum after {} episode{}, terminating.'.format(ep+1, 's' if ep > 0 else ''))

                    if self.is_true_attractor(s):
                        self.true_attractors += 1
                    else:
                        self.false_attractors += 1

                    self.update_final_state_counter(s)

                    break

                contains, cycle_start = self.contains_even_length_cycle(S)   
                if contains:
                    if prints:
                        print('Found cycle. Reached local/global minimum after {} episode{}, terminating.'.format(ep+1, 's' if ep > 0 else ''))

                    self.cycles += 1

                    self.update_cycle_state_counter(S, cycle_start)

                    break

        if plot:
            plot_states([s], title=title, block=True)

        if prints:
            print('Final state energy = {:.2f}'.format(self.energy(s)))

        return S, E # if eps run out


    def run_async(self, x, eps=None, beta_s=None, beta_f=None, row=1, rows=1, trace=False):
        '''
        Run model in asynchronous dynamics. One input vector x will produce
        series of outputs (states) s_t.
        '''
        s = x.copy()
        e = self.energy(s)
        E = [e]

        title = 'Running: asynchronous {}'.format('stochastic' if beta_s is not None else 'deterministic')

        for ep in range(eps):
            ## Set beta for this episode
            if beta_s is None:
                # Deterministic -> no beta
                self.beta = None
                print('Ep {:2d}/{:2d}:  deterministic'.format(ep+1, eps))
            else:
                # Stochastic -> schedule for beta (or temperature)
                self.beta = beta_s * ( (beta_f/beta_s) ** (ep/(eps-1)))
                print('Ep {:2d}/{:2d}:  stochastic, beta = 1/T = {:7.4f}'.format(ep+1, eps, self.beta))

            ## Compute new state for each neuron individually
            for i in np.random.permutation(self.dim):
                s[i] = self.forward_one_neuron(s, neuron_index=i) # update state of selected neuron
                e = self.energy(s)
                E.append(e)

                # Plot
                if trace:
                    plot_state(s, energys=E, index=i, max_eps=eps*self.dim, row=row, rows=rows, title=title, block=False)
                    redraw()

            # Terminate deterministically when stuck in a local/global minimum (loops generally don't occur)
            if self.beta is None:
                if np.all(self.forward_all_neurons(s) == s):
                    print('Reached local/global minimum after {} episode{}, terminating.'.format(ep+1, 's' if ep > 0 else ''))
                    break

        # Plot
        if not trace:
            plot_state(s, energys=E, index=None, max_eps=eps*self.dim, row=row, rows=rows, title=title, block=True)

        print('Final state energy = {:.2f}'.format(self.energy(s)))
