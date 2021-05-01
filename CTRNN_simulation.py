#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 16:55:30 2021

@author: marinadubova
"""

import numpy as np
#import matplotlib.pyplot as plt
    # importing the CTRNN class
from CTRNN import CTRNN
#from stochsearch import EvolSearch
from stochsearch import MicrobialSearch
from stochsearch import LamarckianSearch
from pathos.multiprocessing import ProcessPool
import time

import sys
if sys.version[0] == '3':
    import pickle
else:
    import cPickle as pickle

import sys

trial_index = int(sys.argv[1])
 

# new evolsearch class (author: Madhavun Candadai + edits by Joshua Nunley)
__evolsearch_process_pool = None


class EvolSearch:
    def __init__(self, evol_params, initial_pop, variable_mins, variable_maxes):
        """
        Initialize evolutionary search
        ARGS:
        evol_params: dict
            required keys -
                pop_size: int - population size,
                genotype_size: int - genotype_size,
                fitness_function: function - a user-defined function that takes a genotype as arg and returns a float fitness value
                elitist_fraction: float - fraction of top performing individuals to retain for next generation
                mutation_variance: float - variance of the gaussian distribution used for mutation noise
            optional keys -
                fitness_args: list-like - optional additional arguments to pass while calling fitness function
                                           list such that len(list) == 1 or len(list) == pop_size
                num_processes: int -  pool size for multiprocessing.pool.Pool - defaults to os.cpu_count()
        """
        # check for required keys
        required_keys = [
            "pop_size",
            "genotype_size",
            "fitness_function",
            "elitist_fraction",
            "mutation_variance",
        ]
        for key in required_keys:
            if key not in evol_params.keys():
                raise Exception(
                    "Argument evol_params does not contain the following required key: "
                    + key
                )

        # checked for all required keys
        self.pop_size = evol_params["pop_size"]
        self.genotype_size = evol_params["genotype_size"]
        self.fitness_function = evol_params["fitness_function"]
        self.elitist_fraction = int(
            np.ceil(evol_params["elitist_fraction"] * self.pop_size)
        )
        self.mutation_variance = evol_params["mutation_variance"]

        # validating fitness function
        assert self.fitness_function, "Invalid fitness_function"
        rand_genotype = np.random.rand(self.genotype_size)
        rand_genotype_fitness = self.fitness_function(rand_genotype)
        assert (
            type(rand_genotype_fitness) == type(0.0)
            or type(rand_genotype_fitness) in np.sctypes["float"]
        ), "Invalid return type for fitness_function. Should be float or np.dtype('np.float*')"

        # create other required data
        self.num_processes = evol_params.get("num_processes", None)
        self.pop = np.copy(initial_pop)  # TODO: Check if initial pop is the right size
        self.variable_mins = variable_mins
        self.variable_maxes = variable_maxes
        self.fitness = np.zeros(self.pop_size)
        self.num_batches = int(self.pop_size / self.num_processes)
        self.num_remainder = int(self.pop_size % self.num_processes)

        # check for fitness function kwargs
        if "fitness_args" in evol_params.keys():
            optional_args = evol_params["fitness_args"]
            assert (
                len(optional_args) == 1 or len(optional_args) == self.pop_size
            ), "fitness args should be length 1 or pop_size."
            self.optional_args = optional_args
        else:
            self.optional_args = None

        # creating the global process pool to be used across all generations
        global __evolsearch_process_pool
        __evolsearch_process_pool = ProcessPool(self.num_processes)
        time.sleep(0.5)

    def evaluate_fitness(self, individual_index):
        """
        Call user defined fitness function and pass genotype
        """
        if self.optional_args:
            if len(self.optional_args) == 1:
                return self.fitness_function(
                    self.pop[individual_index, :], self.optional_args[0]
                )
            else:
                return self.fitness_function(
                    self.pop[individual_index, :], self.optional_args[individual_index]
                )
        else:
            return self.fitness_function(self.pop[individual_index, :])

    def elitist_selection(self):
        """
        from fitness select top performing individuals based on elitist_fraction
        """
        self.pop = self.pop[np.argsort(self.fitness)[-self.elitist_fraction :], :]

    def mutation(self):
        """
        create new pop by repeating mutated copies of elitist individuals
        """
        # number of copies of elitists required
        num_reps = (
            int((self.pop_size - self.elitist_fraction) / self.elitist_fraction) + 1
        )

        # creating copies and adding noise
        mutated_elites = np.tile(self.pop, [num_reps, 1])
        mutated_elites += np.random.normal(
            loc=0.0,
            scale=self.mutation_variance,
            size=[num_reps * self.elitist_fraction, self.genotype_size],
        )

        # concatenating elites with their mutated versions
        self.pop = np.vstack((self.pop, mutated_elites))

        # clipping to pop_size
        self.pop = self.pop[: self.pop_size, :]

        # clipping to genotype range
        for i in range(self.pop_size):
            for j in range(self.genotype_size):
                self.pop[i, j] = np.clip(
                    self.pop[i, j], self.variable_mins[j], self.variable_maxes[j]
                )

    def step_generation(self):
        """
        evaluate fitness of pop, and create new pop after elitist_selection and mutation
        """
        global __evolsearch_process_pool

        if not np.all(self.fitness == 0):
            # elitist_selection
            self.elitist_selection()

            # mutation
            self.mutation()

        # estimate fitness using multiprocessing pool
        if __evolsearch_process_pool:
            # pool exists
            self.fitness = np.asarray(
                __evolsearch_process_pool.map(
                    self.evaluate_fitness, np.arange(self.pop_size)
                )
            )
        else:
            # re-create pool
            __evolsearch_process_pool = Pool(self.num_processes)
            self.fitness = np.asarray(
                __evolsearch_process_pool.map(
                    self.evaluate_fitness, np.arange(self.pop_size)
                )
            )

    def execute_search(self, num_gens):
        """
        runs the evolutionary algorithm for given number of generations, num_gens
        """
        # step generation num_gens times
        for gen in np.arange(num_gens):
            self.step_generation()

    def get_fitnesses(self):
        """
        simply return all fitness values of current population
        """
        return self.fitness

    def get_best_individual(self):
        """
        returns 1D array of the genotype that has max fitness
        """
        return self.pop[np.argmax(self.fitness), :]

    def get_best_individual_fitness(self):
        """
        return the fitness value of the best individual
        """
        return np.max(self.fitness)

    def get_mean_fitness(self):
        """
        returns the mean fitness of the population
        """
        return np.mean(self.fitness)

    def get_fitness_variance(self):
        """
        returns variance of the population's fitness
        """
        return np.std(self.fitness) ** 2


# function for running relational task + agent evaluation

# everything is within fitness function

def fitness_function(individual):
    def create_stimuli(n, shape, obj_types):
        dataset = []
        answers = []
        # all positive (same)
        for _ in range(n//2):
            stimulus = np.zeros(shape)
            object_1 = object_2 = np.random.choice(obj_types)
            while True:
                o1_x = np.random.choice(range(shape[0]))
                o1_y = np.random.choice(range(shape[1]))
                o2_x = np.random.choice(range(shape[0]))
                o2_y = np.random.choice(range(shape[1]))
                
                if [o1_x,o1_y] != [o2_x,o2_y]:
                    break
            stimulus[o1_x,o1_y] = object_1
            stimulus[o2_x,o2_y] = object_2
            dataset.append(stimulus)
            #dataset.append(np.array([[1,0],[0,1]]))
            answers.append(1)

        # all negative (different)
        for _ in range(n//2):
            stimulus = np.zeros(shape)
            while True:
                object_1 = np.random.choice(obj_types)        
                object_2 = np.random.choice(obj_types)
                if object_1 != object_2:
                    break
            while True:
                o1_x = np.random.choice(range(shape[0]))
                o1_y = np.random.choice(range(shape[1]))
                o2_x = np.random.choice(range(shape[0]))
                o2_y = np.random.choice(range(shape[1]))
                
                if [o1_x,o1_y] != [o2_x,o2_y]:
                    break
            stimulus[o1_x,o1_y] = object_1
            stimulus[o2_x,o2_y] = object_2
            dataset.append(stimulus)
            #dataset.append(np.zeros((2,2))+2)
            #dataset.append(np.array([[1,0],[0,-1]]))
            answers.append(0)
        return dataset, answers
    
    
    class environment:
        def __init__(self, data, answers, active_perception, net_size):
            self.stimuli = data
            self.answers = answers
            self.active_perception = active_perception
            self.net_size = net_size
            self.current_stimulus = None 

        def step(self,action,done=False):
            if done:#np.nanargmax(action) == self.net_size-2: ## wants to respond
                reward = 1-np.round(abs(self.answers[self.current_stimulus]-action[self.net_size-1]))#(-abs(self.answers[self.current_stimulus]-action[self.net_size-1]))+1 # from 0 to 1
                next_observation = self.stimuli[self.current_stimulus]
            else:
                reward = 0
                if not self.active_perception or np.nanargmax(action) == self.net_size-2 or np.nanargmax(action) == self.net_size-1:
                    next_observation = self.stimuli[self.current_stimulus] # show the whole picture again
               
                else:
                    zoom_index = np.nanargmax(action)
                    zoomed_observation = self.stimuli[self.current_stimulus].flatten()[zoom_index]
                    next_observation = np.zeros(self.stimuli[self.current_stimulus].shape)+zoomed_observation

            return reward, next_observation, done

        def choose_stimulus(self, n=None):
            if n:
                self.current_stimulus = n 
            else:
                self.current_stimulus = np.random.choice(range(len(self.stimuli)))
            # starting observation will be all zeros
            return self.stimuli[self.current_stimulus]
        
    def create_individual(genes, net_size):
    # Net size is fixed at 10
        step_size = 0.01
        network = CTRNN(size=net_size,step_size=step_size)
        network.taus = genes[0:net_size]
        network.biases = genes[net_size:net_size*2]
        network.weights = genes[net_size*2:].reshape((net_size,net_size))
        network.randomize_outputs(0.01,0.09)
        return network
    
    
    
    # choosing the condition for parallel simulations on supercomputer
    
    trial_index = int(sys.argv[1])
 
    cond_list = []
    samples = 1   
    active_perception = [True, False]
    stimuli = [[-1,1],[-1,-2,1,2],[1,2],[1,2,3,4],[1,2,3,4,5,6],[0.1,0.2],[-0.1,0.1],[0.1,0.2,0.3,0.4]]
    
    for ap in active_perception:
        for s in stimuli:
            for j in range(samples):
                cond_list.append([ap, s])
                
    act_perception = cond_list[trial_index][0]
    stim = cond_list[trial_index][1]

    # creating stimuli here
    data, answers = create_stimuli(100,(2,2),stim)
    
    sample_size = 100
    net_size = 6
    max_steps = 8 
    step_size = 0.01
    env = environment(data, answers, act_perception, net_size) # have to change this for the experiments with w/o active perception: True - active perception, False - no active perception
    agent = create_individual(individual,net_size)
    rewards = []
    
    
    # evaluate on the basis of sample_size randomly chosen stimuli
    for _ in range(sample_size):
        reward = 0
        observation = env.choose_stimulus(_)
        agent.randomize_states(0.01,0.99)
        #agent.randomize_outputs(0,1)
        # add empty inputs for 2 "motor" neurons
        observation = np.concatenate((observation.flatten(),np.zeros((2,))+0.001))
        #print(observation)
        done = False
        
        for j in range(max_steps):
            if j >= max_steps-2:
                done = True
            agent.euler_step(observation)
            for m in range(int(1//step_size)):
                agent.euler_step(observation) #([0.01]*net_size)
            action = agent.outputs
            if np.sum(np.isnan(action)) != 0:
                reward = -1000
                break
            rew, observation, done = env.step(action, done)
            # add empty inputs for 2 "motor" neurons
            observation = np.concatenate((observation.flatten(),np.zeros((2,))+0.01))
            reward += rew
            # no punishment if agent did not choose anything and reached max number of steps
            if done:
                break
        rewards.append(reward)
        # NN to settle back
        #agent.randomize_states(0,1)
    return np.float(np.mean(rewards)) 


# here is the actual simulation
net_size = 6
pop_size = 100
genotype_size=net_size*2 + net_size*net_size

evol_params = {
    'num_processes' : 4, # (optional) number of proccesses for multiprocessing.Pool
    'pop_size' : pop_size,    # population size
    'genotype_size': genotype_size, # dimensionality of solution
    'fitness_function': fitness_function, # custom function defined to evaluate fitness of a solution
    'elitist_fraction': 0.1,  # fraction of population retained as is between generations
    'mutation_variance': 0.05,  # mutation noise added to offspring.
}

# create evolutionary search object: here I adapted the code provided by Josh Nunley

initial_pop = np.zeros(shape=(pop_size, genotype_size))
variable_mins = [] #[None]*(net_size*2+net_size*net_size)
variable_maxes = [] #[None]*(net_size*2+net_size*net_size)

weight_lims = {"min": -1, "max": 1}
tau_lims = {"min": 0.01, "max": 1}
bias_lims = {"min": -1, "max": 1}


variable_mins[:net_size] = [tau_lims["min"]]*net_size
variable_mins[net_size : net_size * 2] = [bias_lims["min"]]*net_size
variable_mins[net_size * 2 : net_size ** 2 + net_size * 2] = [weight_lims["min"]]*(net_size**2)

variable_maxes[0:net_size] = [tau_lims["max"]]*net_size
variable_maxes[net_size : net_size * 2] = [bias_lims["max"]]*net_size
variable_maxes[net_size * 2 : net_size ** 2 + net_size * 2] = [weight_lims["max"]]*(net_size**2)


for i in range(pop_size):
    for j in range(genotype_size):
        initial_pop[i, j] = np.random.uniform(
            low=variable_mins[j],
            high=variable_maxes[j],
        )

    # center crossing: also adapted Josh Nunley's code
first_bias_ind = net_size 
last_bias_ind = net_size * 2
weights = initial_pop[:,net_size * 2 : net_size ** 2 + net_size * 2].reshape(
    (pop_size, net_size, net_size))

for i in range(pop_size):
    for j in range(first_bias_ind, last_bias_ind):
        bias_neuron_ind = j - net_size
        initial_pop[i, j] = -np.sum(weights[i, :, bias_neuron_ind]) / 2
            # initial_pop[i, j] = -np.sum(weights[i, bias_neuron_ind, :]) / 2

    # create evolutionary search object
es = EvolSearch(evol_params, initial_pop, variable_mins, variable_maxes)

# evolutionary simulation
    
best_fit = []
mean_fit = []
fit_variance = []
num_gen = 0
max_num_gens = 500
desired_fitness = 0.95
while es.get_best_individual_fitness() < desired_fitness and num_gen < max_num_gens:
    #print('Gen #'+str(num_gen)+' Best Fitness = '+str(es.get_best_individual_fitness())+' Mean fitness = ' + str(es.get_mean_fitness()))
    es.step_generation()
    best_fit.append(es.get_best_individual_fitness())
    mean_fit.append(es.get_mean_fitness())
    fit_variance.append(es.get_fitness_variance())
    num_gen += 1

best_agent = es.get_best_individual()

d = [best_fit, mean_fit, fit_variance, best_agent]

# saving the data

cond_list = []
samples = 1   
active_perception = [True, False]
stimuli = [[-1,1],[-1,-2,1,2],[1,2],[1,2,3,4],[1,2,3,4,5,6],[0.1,0.2],[-0.1,0.1],[0.1,0.2,0.3,0.4]]
    
for ap in active_perception:
    for s in stimuli:
        for j in range(samples):
            cond_list.append([ap, s])

d.append(cond_list[trial_index])

with open("relations/{}.pkl".format(trial_index+16), "wb") as fp:   #Pickling
#with open("game{}.pkl".format(trial_index), "wb") as fp:
    pickle.dump(d, fp, protocol=pickle.HIGHEST_PROTOCOL)
    

