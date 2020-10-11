#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import time
import os
import math as m
import numpy as np
import sys
import matplotlib.pyplot as plt

class MagicSquare(object):
    def __init__(self,n):
        
        self.n = n
        elements = range(1, n**2+1)
        self.M = np.array(elements,dtype = np.int32).reshape((n,n))
    def GenerateRandomSquare(self):
        
        self.M = np.random.permutation(range(1,self.n**2+1)).reshape(self.n,self.n)
    def GenerateNumberOfViolated(self):
        
        target_sum = self.n*(self.n**2 + 1)/2 
        num_of_violated = 0
        
        rows_sum = np.sum(self.M, axis=1) # get sum for each row
        num_of_violated += np.sum(rows_sum != target_sum)
        
        columns_sum = np.sum(self.M, axis=0) 
        num_of_violated += np.sum(columns_sum != target_sum)
        
        num_of_violated += (np.trace(self.M) != target_sum)
        
        num_of_violated += (np.trace(np.flip(self.M,axis=0)) != target_sum)
        return num_of_violated
    def GenerateSuccessors(self,mode,k=None):
        successors = []
        if k == None: 
            k = int(self.n**2*(self.n**2-1)/2)
            mode = 'top'
       
        if (mode == 'random'):
            i = 0
            while i < k:
                x1,x2,y1,y2 = np.random.randint(low=0,high=self.n,size=4)
                if (x1==x2) and (y1==y2):
                    continue
                successor = self.getSuccessor(x1,y1,x2,y2)
                exist = False
                for s in successors: 
                    if np.array_equal(s.M,successor.M):
                        exist = True
                if not exist:
                    successors.append(successor)
                    i += 1
                
        # if you want top k successors
        elif (mode == 'top'):
            for x1 in range(self.n):
                for y1 in range(self.n):
                    for x2 in range(x1,self.n):
                        for y2 in range(self.n):   
                            if (x1 == x2) and (y2<=y1):
                                continue
                            successor = self.getSuccessor(x1,y1,x2,y2)
                           
                            if not successors: 
                                successors.append(successor)
                            else:
                                isPlaced = False 
                                for i in range(len(successors)): 
                                    if (successor.GenerateNumberOfViolated() < successors[i].GenerateNumberOfViolated()):
                                        successors.insert(i,successor)
                                        isPlaced = True
                                        break
                                if (len(successors)>=k): 
                                    successors = successors[:k]
                                else: 
                                    if not isPlaced:
                                        successors.append(successor)
        else:
            raise ValueError("The mode argument is wrong! Take a look in the decription.")
        return successors
            
    def getSuccessor(self,x1,y1,x2,y2):
        successor = MagicSquare(self.n)
        successor.M = self.M.copy()
        # swap values
        successor.M[x1,y1],successor.M[x2,y2] = successor.M[x2,y2],successor.M[x1,y1]
        return successor
    def printSquare(self):
        print("Square: ")
        print(self.M)
        
# =============================================================================
class HillClimbing(object):
    def __init__(self, epochs):
        self.name = 'Hill Climbing'
        self.epochs = epochs
        # used to make a report
        self.current_num_of_violated = []
    def run(self, init_state):
        
        state = init_state
        i = 0
        while  i < self.epochs:
            
            self.current_num_of_violated.append(state.GenerateNumberOfViolated())
            
            if(state.GenerateNumberOfViolated() == 0): 
                break
            # get successor
            successor = state.GenerateSuccessors(mode='top',k=1)[0]
            if (successor.GenerateNumberOfViolated() >= state.GenerateNumberOfViolated()):
                self.current_num_of_violated.append(state.GenerateNumberOfViolated())
                break
            state = successor # make successor current
            i += 1
        return state, state.GenerateNumberOfViolated(), i
    def GenarateReport(self,output_dir):
        
        fig = plt.figure(figsize=(15, 6))
        plt.xlabel('Time (iteration)')
        plt.ylabel('Number of violated constraints')
        plt.plot(self.current_num_of_violated, figure=fig)
        if output_dir != None:
            plt.savefig(os.path.join(output_dir,'ViolatedConstraints-HC.png'))
        plt.close(fig)
            
# =============================================================================
class SimulatedAnnealing(object):
    def __init__(self, epochs, initial_temperature):
        
        self.name = 'Simulated Annealing'
        self.T0 = initial_temperature
        self.epochs = epochs
        self.temperature = []
        self.probability = []
        self.current_num_of_violated = []
    def run(self, init_state):
        
        state = init_state
        i = 0
        T = self.T0
        while  i < self.epochs:
            T = 0.995*T
            self.temperature.append(T) 
            self.current_num_of_violated.append(state.GenerateNumberOfViolated())
            if(state.GenerateNumberOfViolated() == 0) or (T == 0): 
                break
            successor = state.GenerateSuccessors(mode='random',k=1)[0] 
            delta_E = state.GenerateNumberOfViolated() - successor.GenerateNumberOfViolated()
            if (delta_E > 0): 
                state = successor
                self.probability.append(0)
            else:
                prob = m.exp(delta_E/T)
                self.probability.append(prob) 
                if (prob >= np.random.random()): 
                    state = successor
            i += 1
        return state, state.GenerateNumberOfViolated(), i
    def GenarateReport(self,output_dir):
        
        fig = plt.figure(figsize=(15, 6))
        plt.subplot(3,1,1)
        plt.xlabel('Time (iteration)')
        plt.ylabel('Temperature')
        plt.plot(self.temperature, figure=fig)
        plt.subplot(3,1,2)
        plt.xlabel('Time (iteration)')
        plt.ylabel('Probability')
        plt.stem(self.probability, use_line_collection = True)
        plt.subplot(3,1,3)
        plt.xlabel('Time (iteration)')
        plt.ylabel('Violated constraints')
        plt.stem(self.current_num_of_violated, use_line_collection = True)
        
        if output_dir != None:
            plt.savefig(os.path.join(output_dir,'ViolatedConstraints-SA.png'))
        plt.close(fig)
        
# =============================================================================

class GeneticAlgorithm():
    def __init__(self, population,mutation_probability,epochs=5000):
        
        self.name = 'Genetic Algorithm'
        self.epochs = epochs
        self.population = population
        self.mutation_probability = mutation_probability
        # used to make a report
        self.avg_num_of_violated = []
        self.min_num_of_violated = []
    def run(self, init_state):
       
        n = init_state.n
        population = init_state.GenerateSuccessors(mode='random',k=self.population)
        i = 0
        while  i < self.epochs:
            # make varaibles for report
            minimum = 2*n+2+1
            avg = 0
            for state in population:
                avg += state.GenerateNumberOfViolated()/len(population)
                if (state.GenerateNumberOfViolated() < minimum):
                    minimum = state.GenerateNumberOfViolated()
            self.avg_num_of_violated.append(avg)
            self.min_num_of_violated.append(minimum)
            # check if you have reached solution
            converged = False
            for s in population:
                if (s.GenerateNumberOfViolated() == 0):
                    state = s
                    converged = True
                    break
            if converged:
                break
            fitness = self.CheckFitness(population)
            population = self.GenerateSelection(population, fitness)
            children = self.GenerateCrossover(population)
            population = self.PerformMutation(children)
            i += 1
        if (i == self.epochs):
            state = min(population, key = lambda state:state.GenerateNumberOfViolated())
        return state, state.GenerateNumberOfViolated(), i
    
    def CheckFitness(self,population):
       
        n = population[0].n
        sum_of_all = sum(2*n+2-state.GenerateNumberOfViolated() for state in population)
        fitness = []
        if (sum_of_all == 0): 
            fitness = [1/len(population)]*len(population)
        else:
            for state in population:
                fitness.append((2*n+2-state.GenerateNumberOfViolated())/sum_of_all)
        return fitness
    def GenerateSelection(self,population,fitness):
       
        population = np.random.choice(population,size=self.population, p=fitness, replace=True)
        return population
    def GenerateCrossover(self,parents):
        
        n = parents[0].n
        children = []
        i = 0
        while i  < (self.population/2)*2 - 1:
            inversion1 = self.InversionHelper(parents[i].M.reshape(n**2))
            inversion2 = self.InversionHelper(parents[i+1].M.reshape(n**2))
            ind = np.random.randint(low=0, high=n**2+1)
            child1_inverted = self.CrossParentsHelper(inversion1,inversion2,ind)
            child2_inverted = self.CrossParentsHelper(inversion2,inversion1,ind)
            child = MagicSquare(n)
            child.M = self.PermutationHelper(child1_inverted).reshape(n,n)
            children.append(child)
            child = MagicSquare(n)
            child.M = self.PermutationHelper(child2_inverted).reshape(n,n)
            children.append(child)
            i += 2
        return children
    def CrossParentsHelper(self,parent1,parent2,ind):
        
        part1 = parent1[:ind]
        part2 = parent2[ind:]
        child = np.concatenate([part1,part2])
        return child
    def InversionHelper(self,permutation):
        
        inversion = []
        for i in range(len(permutation)):
            count = 0
            j = 0
            while (permutation[j] != i+1):
                if(permutation[j] > i+1):
                    count += 1
                j += 1
            inversion.append(count)
        return inversion               
    def PermutationHelper(self,inversion):
        permutation = [None]*len(inversion)
        positions = [None]*len(inversion)
        i = len(inversion)-1
        while i >= 0:
            positions[i] = int(inversion[i])
            for j in range(i+1,len(inversion)):
                if (positions[j] >= positions[i]):
                    positions[j] += 1
            i = i - 1
        for j,pos in enumerate(positions):
            permutation[pos] = j+1
        return np.array(permutation)
    def PerformMutation(self,children):
        
        n = children[0].n
        for i in range(len(children)):
            prob = np.random.random()
            if (self.mutation_probability >= prob):
                x1,x2,y1,y2 = -1,-1,-1,-1
                while(x1 == x2) and (y1==y2):
                    x1,y1,x2,y2 = np.random.randint(0,n,size=4)
                children[i] = children[i].getSuccessor(x1,y1,x2,y2)
        return children
                    
    def GenarateReport(self,output_dir):
       
        fig = plt.figure(figsize=(15, 6))
        plt.subplot(2,1,1)
        plt.xlabel('Time (generation)')
        plt.ylabel('AVG violated constraints')
        plt.plot(self.avg_num_of_violated, figure=fig)
        plt.subplot(2,1,2)
        plt.xlabel('Time (generation)')
        plt.ylabel('MIN violated constraints')
        plt.plot(self.min_num_of_violated, figure=fig)
        
        if output_dir != None:
            plt.savefig(os.path.join(output_dir,'ViolatedConstraints-GA.png'))
        plt.close(fig)            

def MonteCarloSimulation(algorithm,init_states,MC_epochs,plots_dir):
   
    number_of_violated = []
    needed_epochs = []
    for i in range(MC_epochs):
        print("Monte Carlo iteration: {}".format(i+1))
        state, num_of_violated, needed_iter = algorithm.run(init_states[i])
        if i == 0:
            algorithm.GenarateReport(os.path.join(plots_dir,algorithm.name))
        number_of_violated.append(num_of_violated)
        needed_epochs.append(needed_iter)
    avg_num_of_violated = sum(number_of_violated)/len(number_of_violated)
    avg_epochs = sum(needed_epochs)/len(needed_epochs)
    return avg_num_of_violated, avg_epochs

def MakeDirectoryTree(tree,output_dir):
    for d in tree:
        os.makedirs(os.path.join(output_dir,d),exist_ok = True)
if __name__=="__main__":
    plots_dir = os.path.join(os.getcwd(),'plots')
    MakeDirectoryTree(['plots'],os.getcwd())
    os.makedirs(plots_dir, exist_ok=True) 
    n = 3
    epochs = 1000
    MC_epochs = 100
    init_states = []
    for i in range(MC_epochs):
        init_state = MagicSquare(n)
        init_state.GenerateRandomSquare()
        init_states.append(init_state)
    alg1 = HillClimbing(epochs)
    alg2 = SimulatedAnnealing(epochs, initial_temperature=10000)
    alg3 = GeneticAlgorithm(10, 0.05, epochs)
    
    algorithms = [alg1,alg2,alg3]
    MakeDirectoryTree([alg.name for alg in algorithms], plots_dir)
    result_of_MC_simulation = pd.DataFrame(columns=["Algorithm", "Average Violated Constraints",
                                    "Average epochs",
                                    "Execution Time",
                                   ])
    for algorithm in algorithms:
         print('---'*10)
         print('Executing {} algorithm'.format(algorithm.name))
         tic = time.time()
         (avg_num_of_violated,
          avg_epochs) = MonteCarloSimulation(algorithm,init_states,MC_epochs,plots_dir)
         toc = time.time()
         print("Average number of violated constraints: {:.3f}".format(avg_num_of_violated))
         print("Average number of needed epochs: {:.3f}".format(avg_epochs))
         print("Time: {:.3f}s".format(toc-tic))
         result_of_MC_simulation = result_of_MC_simulation.append(
             {"Algorithm": algorithm.name,
              "Average Violated Constraints": avg_num_of_violated,
              "Average epochs": avg_epochs,
              "Execution Time": toc-tic}, 
             ignore_index = True
             )
    result_of_MC_simulation.to_csv(os.path.join(os.getcwd(), "result.csv"))


# In[ ]:




