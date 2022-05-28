# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, Juraj Holas, Peter Gergel, Endre Hamerlik, Štefan Pócoš, Iveta Bečková 2017-2022

import itertools
import numpy as np

from util import *

eps_hard_limit = 100


class Hopfield():

    def __init__(self, dim):
        self.dim  = dim
        self.beta = None  # if beta is None: deterministic run, else stochastic run


    def train(self, patterns):
        '''
        Compute weight matrix analytically
        '''

        self.W  = np.zeros((self.dim, self.dim))
        for p in patterns:
            self.W += 1/len(patterns)*np.outer(p,p) # FIXME compute weights - "store" patterns in weight matrix
            np.fill_diagonal(self.W, 0)

        print(self.W.shape)


    def energy(self, s):
        '''
        Compute energy for given state
        '''
        outer_s = np.outer(s,s)
        np.fill_diagonal(outer_s, 0)
        return -1/2*np.sum(self.W*outer_s)  # FIXME


    # asynchronous dynamics
    def forward_one_neuron(self, s, neuron_index):
        '''
        Perform forward pass in asynchronous dynamics - compute output of selected neuron
        '''
        net = self.W[neuron_index]@s  # FIXME

        if self.beta is None:
            # Deterministic transition
            return -1 if not np.sign(net) else np.sign(net)   # FIXME
        else:
            # Stochastic transition (Output for each neuron is either -1 or 1!)
            prob = 1/(1 + np.exp(-self.beta*net))                   # FIXME
            return -1 if np.random.rand() >= prob else 1          # FIXME


    # synchronous dynamics (not implemented, not part of the exercise)
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


    # not implemented properly, modify for correct functioning (not part of the exercise)
    def run_sync(self, x, beta_s=None, beta_f=None, row=1, rows=1, trace=False):
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
                print('Ep {:2d}/{:2d}:  deterministic'.format(ep+1, eps_hard_limit))
            else:
                # Stochastic -> schedule for beta (or temperature)
                self.beta = beta_s * ( (beta_f/beta_s) ** (ep/(eps_hard_limit-1)))
                print('Ep {:2d}/{:2d}:  stochastic, beta = 1/T = {:7.4f}'.format(ep+1, eps_hard_limit, self.beta))

            ## Compute new state for all neurons
            s = self.forward_all_neurons(s)
            e = self.energy(s)

            S.append(s)
            E.append(e)

            if trace:
                plot_state(s, energys=E, max_eps=eps_hard_limit*self.dim, row=row, rows=rows, title=title, block=False)
                redraw()


            ## Detect termination criterion
            # if [fixed point is reached] or [cycle is reached]:
            #     return S, E

        if not trace:
            plot_state(s, energys=E, index=None, max_eps=eps*self.dim, row=row, rows=rows, title=title, block=True)

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
