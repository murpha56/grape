# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 15:21:08 2021

@author: allan
"""

import sys

import grape
import algorithms
from functions import add, sub, mul, pdiv, plog, exp, psqrt

from os import path
import pandas as pd
import numpy as np
from deap import creator, base, tools
import scipy.stats

import random

import warnings
warnings.filterwarnings("ignore")

problem = sys.argv[1]
fitfunc = sys.argv[2]
#seed = sys.argv[3]
#if problem == 'vladislavleva4':
#print("worked")


if problem == 'keijzer5':
    X_train_1 = np.random.uniform(-1.0, 1.0, (2, 1000))
    X_train_2 = np.random.uniform(1, 2, (1, 1000))
    #print("Train1:")
    #print(X_train_2.shape)
    X_train = np.concatenate((X_train_1, X_train_2), axis=0)
    #print("Together:")
    #print(X_train.shape)
    Y_train = np.zeros([1000,], dtype=float)
    #print(Y_train[0])

    for i in range(1000):
        Y_train[i] = (30*X_train[0,i]*X_train[1,i])/((X_train[0,i]-10)*(X_train[2,i]**2))

    #print(Y_train[0])

    X_test_1 = np.random.uniform(-1.0, 1.0, (2, 10000))
    X_test_2 = np.random.uniform(1, 2, (1, 10000))
    X_test = np.concatenate((X_test_1, X_test_2), axis=0)
    Y_test = np.zeros([10000,], dtype=float)

    for i in range(10000):
        Y_test[i] = (30*X_test[0,i]*X_test[1,i])/((X_test[0,i]-10)*(X_test[2,i]**2))

    GRAMMAR_FILE = 'Keijzer5.bnf'

if problem == 'keijzer13':
    #X_train = np.zeros([3,1000], dtype=float)
    X_train = np.random.uniform(-3.0, 3.0, (2, 20))
    Y_train = np.zeros([20,], dtype=float)
    #print(Y_train[0])

    for i in range(20):
        Y_train[i] = 6*np.sin(X_train[0,i])*np.cos(X_train[1,i])

    #print(Y_train[0])

    X_test = np.random.uniform(-3.0, 3.0, (2, 20))
    Y_test = np.zeros([20,], dtype=float)

    for i in range(20):
        Y_test[i] = 6*np.sin(X_test[0,i])*np.cos(X_test[1,i])

    GRAMMAR_FILE = 'Keijzer13.bnf'

if problem == 'nguyen7':
    #X_train = np.zeros([3,1000], dtype=float)
    X_train = np.random.uniform(0, 2.0, (1, 20))
    Y_train = np.zeros([20,], dtype=float)

    for i in range(20):
        Y_train[i] = log(X_train[0,i] + 1) + log(X_train[0,i]**2 + 1)

    X_test = np.random.uniform(0, 2.0, (1, 20))
    Y_test = np.zeros([20,], dtype=float)

    for i in range(20):
        Y_test[i] = log(X_test[0,i] + 1) + log(X_test[0,i]**2 + 1)

    GRAMMAR_FILE = 'Nguyen7.bnf'

if problem == 'nguyen5':
    #X_train = np.zeros([3,1000], dtype=float)
    X_train = np.random.uniform(-1.0, 1.0, (1, 20))
    Y_train = np.zeros([20,], dtype=float)

    for i in range(20):
        Y_train[i] = np.sin(X_train[0,i]**2)*np.cos(X_train[0,i]) - 1

    X_test = np.random.uniform(-1.0, 1.0, (1, 20))
    Y_test = np.zeros([20,], dtype=float)

    for i in range(20):
        Y_test[i] = np.sin(X_test[0,i]**2)*np.cos(X_test[0,i]) - 1

    GRAMMAR_FILE = 'Nguyen5.bnf'

if problem == 'korns12':
    #X_train = np.zeros([3,1000], dtype=float)
    X_train = np.random.uniform(-50.0, 50.0, (5, 10000))
    Y_train = np.zeros([10000,], dtype=float)
    for i in range(10000):
        Y_train[i] = 2 - 2.1*np.cos(9.8*X_train[0,i])*np.sin(1.3*X_train[1,i])
        print(Y_train[i])

    X_test = np.random.uniform(-50.0, 50.0, (5, 10000))
    Y_test = np.zeros([10000,], dtype=float)
    for i in range(10000):
        Y_test[i] = 2 - 2.1*np.cos(9.8*X_test[0,i])*np.sin(1.3*X_test[1,i])

    GRAMMAR_FILE = 'Korns12.bnf'

if problem == 'korns1':
    #X_train = np.zeros([3,1000], dtype=float)
    X_train = np.random.uniform(-50.0, 50.0, (5, 10000))
    Y_train = np.zeros([10000,], dtype=float)
    for i in range(10000):
        Y_train[i] = 1.57 + 24.3*X_train[3,i]
        print(Y_train[i])

    X_test = np.random.uniform(-50.0, 50.0, (5, 10000))
    Y_test = np.zeros([10000,], dtype=float)
    for i in range(10000):
        Y_test[i] = 1.57 + 24.3*X_test[3,i]

    GRAMMAR_FILE = 'Korns1.bnf'


if problem == 'korns5':
    #X_train = np.zeros([3,1000], dtype=float)
    X_train = np.random.uniform(-50.0, 50.0, (5, 10000))
    Y_train = np.zeros([10000,], dtype=float)
    for i in range(10000):
        Y_train[i] = 3.0 + (2.13*plog(X_train[4,i])
        print(Y_train[i])

    X_test = np.random.uniform(-50.0, 50.0, (5, 10000))
    Y_test = np.zeros([10000,], dtype=float)
    for i in range(10000):
        Y_test[i] = 3.0 + (2.13*plog(X_test[4,i])

    GRAMMAR_FILE = 'Korns5.bnf'

if problem == 'pagie1':
    X_train = np.zeros([2,676], dtype=float)
    Y_train = np.zeros([676,], dtype=float)

    data_train = pd.read_table(r"datasets/Pagie1_train.txt")
    for i in range(2):
        for j in range(676):
            X_train[i,j] = data_train['x'+ str(i)].iloc[j]
    for i in range(676):
        Y_train[i] = data_train['response'].iloc[i]

    X_test = np.zeros([2,10000], dtype=float)
    Y_test = np.zeros([10000,], dtype=float)

    data_test = pd.read_table(r"datasets/Pagie1_test.txt")
    for i in range(2):
        for j in range(10000):
            X_test[i,j] = data_test['x'+ str(i)].iloc[j]
    for i in range(10000):
        Y_test[i] = data_test['response'].iloc[i]

    GRAMMAR_FILE = 'Pagie1.bnf'

if problem == 'vladislavleva4':
    X_train = np.zeros([5,1024], dtype=float)
    Y_train = np.zeros([1024,], dtype=float)

    data_train = pd.read_table(r"datasets/Vladislavleva4_train.txt")
    for i in range(5):
        for j in range(1024):
              X_train[i,j] = data_train['x'+ str(i)].iloc[j]
    for i in range(1024):
        Y_train[i] = data_train['response'].iloc[i]

    X_test = np.zeros([5,5000], dtype=float)
    Y_test = np.zeros([5000,], dtype=float)

    data_test = pd.read_table(r"datasets/Vladislavleva4_test.txt")
    for i in range(5):
        for j in range(5000):
            X_test[i,j] = data_test['x'+ str(i)].iloc[j]
    for i in range(5000):
        Y_test[i] = data_test['response'].iloc[i]

    GRAMMAR_FILE = 'Vladislavleva4.bnf'

if problem == 'vladislavleva1':
    X_train = np.random.uniform(0.3, 4.0, (2, 100))
    Y_train = np.zeros([100,], dtype=float)

    for i in range(100):
        Y_train[i] = np.exp(-(X_train[0,i]-1)**2)/(1.2+(X_train[1,i]-2.5)**2)
        print(Y_train[i])

    X_test_1 = np.arange(-0.2, 4.2, 0.1)
    X_test_2 = np.arange(-0.2, 4.2, 0.1)
    X_test = np.concatenate((X_test_1, X_test_2), axis=0)
    Y_test = np.zeros([2026,], dtype=float)

    for i in range(2026):
        Y_test[i] = np.exp(-(X_test[0,i]-1)**2)/(1.2+(X_test[1,i]-2.5)**2)

    GRAMMAR_FILE = 'Vladislavleva1.bnf'

elif problem == 'Dow':
    X_train = np.zeros([57,747], dtype=float)
    Y_train = np.zeros([747,], dtype=float)

    data_train = pd.read_table(r"datasets/DowNorm_train.txt")
    for i in range(56):
        for j in range(747):
              X_train[i,j] = data_train['x'+ str(i+1)].iloc[j]
    for i in range(747):
        Y_train[i] = data_train['y'].iloc[i]

    X_test = np.zeros([57,319], dtype=float)
    Y_test = np.zeros([319,], dtype=float)

    data_test = pd.read_table(r"datasets/DowNorm_test.txt")
    for i in range(56):
        for j in range(319):
            X_test[i,j] = data_test['x'+ str(i+1)].iloc[j]
    for i in range(319):
        Y_test[i] = data_test['y'].iloc[i]

    GRAMMAR_FILE = 'Dow.bnf'

BNF_GRAMMAR = grape.Grammar(r"grammars/" + GRAMMAR_FILE)

def fitness_eval(individual, points):
    #points = [X, Y]
    x = points[0]
    y = points[1]

    if individual.invalid == True:
        return np.NaN,

    try:
        pred = eval(individual.phenotype)
    except (FloatingPointError, ZeroDivisionError, OverflowError,
            MemoryError, ValueError):
        return np.NaN,
    except Exception as err:
            # Other errors should not usually happen (unless we have
            # an unprotected operator) so user would prefer to see them.
            print("evaluation error", err)
            raise
    assert np.isrealobj(pred)

    #print(y)
    #print("Pred:")
    #print(pred)
    #slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(y, pred)

    try:
        if fitfunc == 'MSE':
            fitness = np.mean(np.square(y - pred))
        elif fitfunc == 'LS':
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(y, pred)
            scaled_output = intercept + slope*pred
            fitness = np.mean(np.square(y - scaled_output))
        else:
            corr_matrix = np.corrcoef(y, pred)
            fitness = 1 - (corr_matrix[0,1]**2)
    except (FloatingPointError, ZeroDivisionError, OverflowError,
            MemoryError, ValueError):
        fitness = np.NaN
    except Exception as err:
            # Other errors should not usually happen (unless we have
            # an unprotected operator) so user would prefer to see them.
            print("fitness error", err)
            raise

    if fitness == float("inf"):
        return np.NaN,

    #slope, intercept, r_value, p_value, std_err = stats.linregress(targets, outputs)

    return fitness,


def fitness_eval_MSE(individual, points):
    #points = [X, Y]
    x = points[0]
    y = points[1]

    if individual.invalid == True:
        return np.NaN,

    try:
        pred = eval(individual.phenotype)
    except (FloatingPointError, ZeroDivisionError, OverflowError,
            MemoryError, ValueError):
        return np.NaN,
    except Exception as err:
            # Other errors should not usually happen (unless we have
            # an unprotected operator) so user would prefer to see them.
            print("evaluation error", err)
            raise
    assert np.isrealobj(pred)

    try:
        fitness = np.mean(np.square(y - pred))
        print(fitness)
    except (FloatingPointError, ZeroDivisionError, OverflowError,
            MemoryError, ValueError):
        fitness = np.NaN
    except Exception as err:
            # Other errors should not usually happen (unless we have
            # an unprotected operator) so user would prefer to see them.
            print("fitness error", err)
            raise

    if fitness == float("inf"):
        return np.NaN,

    #slope, intercept, r_value, p_value, std_err = stats.linregress(targets, outputs)

    return fitness,


def fitness_eval_LS(individual, points):
    #points = [X, Y]
    x = points[0]
    y = points[1]
    slope = 0.0
    intercept = 0.0

    if individual.invalid == True:
        return np.NaN,

    try:
        pred = eval(individual.phenotype)
    except (FloatingPointError, ZeroDivisionError, OverflowError,
            MemoryError, ValueError):
        return np.NaN,
    except Exception as err:
            # Other errors should not usually happen (unless we have
            # an unprotected operator) so user would prefer to see them.
            print("evaluation error", err)
            raise
    assert np.isrealobj(pred)

    try:
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(y, pred)
        scaled_output = intercept + slope*pred
        fitness = np.mean(np.square(y - scaled_output))
    except (FloatingPointError, ZeroDivisionError, OverflowError,
            MemoryError, ValueError):
        fitness = np.NaN
    except Exception as err:
            # Other errors should not usually happen (unless we have
            # an unprotected operator) so user would prefer to see them.
            print("fitness error", err)
            raise

    if fitness == float("inf"):
        return np.NaN,

    individual.intercept = intercept
    individual.slope = slope

    return fitness,


def fitness_test_LS(individual, points):
    #points = [X, Y]
    x = points[0]
    y = points[1]

    if individual.invalid == True:
        return np.NaN,

    try:
        pred = eval(individual.phenotype)
    except (FloatingPointError, ZeroDivisionError, OverflowError,
            MemoryError, ValueError):
        return np.NaN,
    except Exception as err:
            # Other errors should not usually happen (unless we have
            # an unprotected operator) so user would prefer to see them.
            print("evaluation error", err)
            raise
    assert np.isrealobj(pred)

    try:
        scaled_output = individual.intercept + individual.slope*pred
        fitness = np.mean(np.square(y - scaled_output))
    except (FloatingPointError, ZeroDivisionError, OverflowError,
            MemoryError, ValueError):
        fitness = np.NaN
    except Exception as err:
            # Other errors should not usually happen (unless we have
            # an unprotected operator) so user would prefer to see them.
            print("fitness error", err)
            raise

    if fitness == float("inf"):
        return np.NaN,

    return fitness,



def fitness_eval_Corr(individual, points):
    #points = [X, Y]
    x = points[0]
    y = points[1]
    slope = 0.0
    intercept = 0.0

    if individual.invalid == True:
        return np.NaN,

    try:
        pred = eval(individual.phenotype)
    except (FloatingPointError, ZeroDivisionError, OverflowError,
            MemoryError, ValueError):
        return np.NaN,
    except Exception as err:
            # Other errors should not usually happen (unless we have
            # an unprotected operator) so user would prefer to see them.
            print("evaluation error", err)
            raise
    assert np.isrealobj(pred)

    try:
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(y, pred)
        corr_matrix = np.corrcoef(y, pred)
        fitness = 1 - (corr_matrix[0,1]**2)
    except (FloatingPointError, ZeroDivisionError, OverflowError,
            MemoryError, ValueError):
        fitness = np.NaN
    except Exception as err:
            # Other errors should not usually happen (unless we have
            # an unprotected operator) so user would prefer to see them.
            print("fitness error", err)
            raise

    if fitness == float("inf"):
        return np.NaN,

    individual.intercept = intercept
    individual.slope = slope

    return fitness,

def fitness_test_Corr(individual, points):
    #points = [X, Y]
    x = points[0]
    y = points[1]

    if individual.invalid == True:
        return np.NaN,

    try:
        pred = eval(individual.phenotype)
    except (FloatingPointError, ZeroDivisionError, OverflowError,
            MemoryError, ValueError):
        return np.NaN,
    except Exception as err:
            # Other errors should not usually happen (unless we have
            # an unprotected operator) so user would prefer to see them.
            print("evaluation error", err)
            raise
    assert np.isrealobj(pred)

    try:
        scaled_output = individual.intercept + individual.slope*pred
        fitness = np.mean(np.square(y - scaled_output))
    except (FloatingPointError, ZeroDivisionError, OverflowError,
            MemoryError, ValueError):
        fitness = np.NaN
    except Exception as err:
            # Other errors should not usually happen (unless we have
            # an unprotected operator) so user would prefer to see them.
            print("fitness error", err)
            raise

    if fitness == float("inf"):
        return np.NaN,

    return fitness,

POPULATION_SIZE = 500
MAX_GENERATIONS = 50
P_CROSSOVER = 0.9
P_MUTATION = 0.05
ELITE_SIZE = round(0.01*POPULATION_SIZE) #it should be smaller or equal to HALLOFFAME_SIZE
HALLOFFAME_SIZE = round(0.01*POPULATION_SIZE) #it should be at least 1

 #Pay attention that the seed is set up inside the loop of runs, so you are going to have similar runs

MIN_INIT_GENOME_LENGTH = 30 #used only for random initialisation
MAX_INIT_GENOME_LENGTH = 50
random_initilisation = False #put True if you use random initialisation

MAX_INIT_TREE_DEPTH = 13 #equivalent to 6 in GP with this grammar
MIN_INIT_TREE_DEPTH = 3
MAX_TREE_DEPTH = 35 #equivalent to 17 in GP with this grammar
MAX_WRAPS = 0
CODON_SIZE = 255

CODON_CONSUMPTION = 'lazy'
GENOME_REPRESENTATION = 'numpy'
MAX_GENOME_LENGTH = None

#Set the next two parameters with integer values, if you want to use the penalty approach
#PENALTY_DIVIDER = None
#PENALISE_GREATER_THAN = None

TOURNAMENT_SIZE = 5

toolbox = base.Toolbox()

# define a single objective, minimising fitness strategy:
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

creator.create('Individual', grape.Individual, fitness=creator.FitnessMin)

toolbox.register("populationCreator", grape.sensible_initialisation, creator.Individual)
#toolbox.register("populationCreator", grape.random_initialisation, creator.Individual)
#toolbox.register("populationCreator", grape.PI_Grow, creator.Individual)

#toolbox.register("evaluate", fitness_eval, penalty_divider=PENALTY_DIVIDER, penalise_greater_than=PENALISE_GREATER_THAN)
#toolbox.register("evaluate", fitness_eval)
toolbox.register("evaluate_MSE", fitness_eval_MSE)
toolbox.register("evaluate_LS", fitness_eval_LS)
toolbox.register("evaluate_test_LS", fitness_test_LS)
toolbox.register("evaluate_Corr", fitness_eval_Corr)
toolbox.register("evaluate_test_Corr", fitness_test_Corr)
# Tournament selection:
toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)

# Single-point crossover:
toolbox.register("mate", grape.crossover_onepoint)

# Flip-int mutation:
toolbox.register("mutate", grape.mutation_int_flip_per_codon)

REPORT_ITEMS = ['gen', 'invalid', 'min_train', 'min_test',
          'best_ind_length', 'avg_length',
          'best_ind_nodes', 'avg_nodes',
          'best_ind_depth', 'avg_depth',
          'best_ind_used_codons', 'avg_used_codons',
          'structural_diversity', 'fitness_diversity']

N_RUNS = 30

for i in range(N_RUNS):
    print()
    print()
    print("Run:", i+1)
    print()

    RANDOM_SEED = random.randint(i)
    random.seed(RANDOM_SEED) #Comment this line or set a different RANDOM_SEED each run if you want distinct results

    ##create data here,

    # create initial population (generation 0):
    if random_initilisation:
        population = toolbox.populationCreator(pop_size=POPULATION_SIZE,
                                           bnf_grammar=BNF_GRAMMAR,
                                           min_init_genome_length=MIN_INIT_GENOME_LENGTH,
                                           max_init_genome_length=MAX_INIT_GENOME_LENGTH,
                                           max_init_depth=MAX_TREE_DEPTH,
                                           codon_size=CODON_SIZE,
                                           codon_consumption=CODON_CONSUMPTION,
                                           genome_representation=GENOME_REPRESENTATION
                                           )
    else:
        population = toolbox.populationCreator(pop_size=POPULATION_SIZE,
                                           bnf_grammar=BNF_GRAMMAR,
                                           min_init_depth=MIN_INIT_TREE_DEPTH,
                                           max_init_depth=MAX_INIT_TREE_DEPTH,
                                           codon_size=CODON_SIZE,
                                           codon_consumption=CODON_CONSUMPTION,
                                           genome_representation=GENOME_REPRESENTATION
                                            )

    # define the hall-of-fame object:
    hof = tools.HallOfFame(HALLOFFAME_SIZE)

    # prepare the statistics object:
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.nanmean)
    stats.register("std", np.nanstd)
    stats.register("min", np.nanmin)
    stats.register("max", np.nanmax)

    # perform the Grammatical Evolution flow:
    population, logbook = algorithms.ge_eaSimpleWithElitism(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                              ngen=MAX_GENERATIONS, elite_size=ELITE_SIZE,
                                              bnf_grammar=BNF_GRAMMAR,
                                              codon_size=CODON_SIZE,
                                              max_tree_depth=MAX_TREE_DEPTH,
                                              max_genome_length=MAX_GENOME_LENGTH,
                                              points_train=[X_train, Y_train],
                                              points_test=[X_test, Y_test],
                                              codon_consumption=CODON_CONSUMPTION,
                                              report_items=REPORT_ITEMS,
                                              genome_representation=GENOME_REPRESENTATION,
                                              stats=stats, halloffame=hof, verbose=False)

    import textwrap
    best = hof.items[0].phenotype
    print("Best individual: \n","\n".join(textwrap.wrap(best,80)))
    print("\nTraining Fitness: ", hof.items[0].fitness.values[0])
    print("Depth: ", hof.items[0].depth)
    print("Length of the genome: ", len(hof.items[0].genome))
    print(f'Used portion of the genome: {hof.items[0].used_codons/len(hof.items[0].genome):.2f}')

    max_fitness_values, mean_fitness_values = logbook.select("max", "avg")
    min_fitness_values, std_fitness_values = logbook.select("min", "std")
    best_ind_length = logbook.select("best_ind_length")
    avg_length = logbook.select("avg_length")

    selection_time = logbook.select("selection_time")
    generation_time = logbook.select("generation_time")
    gen, invalid = logbook.select("gen", "invalid")
    avg_used_codons = logbook.select("avg_used_codons")
    best_ind_used_codons = logbook.select("best_ind_used_codons")

    best_ind_nodes = logbook.select("best_ind_nodes")
    avg_nodes = logbook.select("avg_nodes")

    best_ind_depth = logbook.select("best_ind_depth")
    avg_depth = logbook.select("avg_depth")

    behavioural_diversity = logbook.select("behavioural_diversity")
    structural_diversity = logbook.select("structural_diversity")
    fitness_diversity = logbook.select("fitness_diversity")

    fitness_test = logbook.select("fitness_test")

    import csv
    r = RANDOM_SEED

    header = REPORT_ITEMS
    with open("results/" + problem + "_FF_" + fitfunc + "_Run_" + str(i) + "_Seed_" + str(r) + ".csv", "w", encoding='UTF8', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(header)
        for value in range(len(max_fitness_values)):
            writer.writerow([gen[value], invalid[value], min_fitness_values[value], fitness_test[value],
                             best_ind_length[value],
                             avg_length[value],
                             best_ind_nodes[value],
                             avg_nodes[value],
                             best_ind_depth[value],
                             avg_depth[value],
                             best_ind_used_codons[value],
                             avg_used_codons[value],
                             structural_diversity[value],
                             fitness_diversity[value]])
