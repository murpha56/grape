# -*- coding: utf-8 -*-
"""
@author: Aidan, Allan & Mahsa
"""
import sys
sys.path.insert(0, './')
import src.grape as grape
import src.algorithms as algorithms
from src.functions import add, sub, mul, pdiv, neg, and_, or_, not_, less_than_or_equal, greater_than_or_equal


import os
import pandas as pd
import numpy as np
from deap import creator, base, tools
import random

from sklearn.model_selection import train_test_split

#BNF_GRAMMAR = grape.Grammar(r"grammars/UNET.bnf")

#genome = [1,1,2,3,4,5,6,6,7,3,4,6,4,2,2,1,5,6,8,9]

#indiv = grape.mapper_eager(genome,BNF_GRAMMAR,10)

#print(eval(indiv[0]))
#eval(indiv[0])
#exec(eval(indiv[0]), globals())
#unet()

import configparser
import collections
import tensorflow as tf
from deap import base, creator, tools
from keras.models import Model
from keras.layers import Input, concatenate, UpSampling2D, Dropout, AveragePooling2D, GlobalAveragePooling2D, \
    GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, \
    Conv2D, Add, Activation, Lambda, Conv1D, Layer, MaxPooling2D, AveragePooling2D, BatchNormalization, add, Conv2DTranspose

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras import backend as K
from matplotlib import pyplot as plt
import ast
import gc
from keras_drop_block import DropBlock2D
import time
import datetime
start_time = time.time()
print ("Start: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
import cv2
from skimage.io import imread
from keras.utils import io_utils
from tensorflow.python.platform import tf_logging as logging


problem = 'UNET'

if problem == 'UNET':

    BNF_GRAMMAR = grape.Grammar(r"grammars/UNET.bnf")

    x_train = "where you saved your training/../"
    y_train = "where you saved your training/../"
    x_validate = "where you saved your valid/../"
    y_validate = "where you saved your valid/../"

def eval_model_leval_model_loss_functionoss_function(

    start_model_time = time.time()
    print ("Start-model: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(f'GA------------Individual = {individual.phenotype})

    #create function UNet, this is the UNet creted by GE
    exec(eval(individual.phenotype), globals())

    model = unet()

    history=model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2, validation_data=(x_validate, y_validate), shuffle=True, callbacks=[custom_early_stopping])
    trainableParams=np.sum([np.prod(v.get_shape()) for v in model1.trainable_weights])
    nonTrainableParams=np.sum([np.prod(v.get_shape()) for v in model1.non_trainable_weights])
    totalParams=trainableParams + nonTrainableParams
    fitness1=max(history.history['val_accuracy'])
    fitness2= int(totalParams)
    print(f'Myresult')
    print(fitness2)
    print(fitness1)
    print ("End-model: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("---The End %s seconds ---" % (time.time() - start_model_time))
    del history
    del model
    gc.collect()
    return fitness1

toolbox = base.Toolbox()

# define a single objective, minimising fitness strategy:
creator.create("Fitness", base.Fitness, weights=(1.0,))

creator.create('Individual', grape.Individual, fitness=creator.Fitness)

toolbox.register("populationCreator", grape.sensible_initialisation, creator.Individual)
#toolbox.register("populationCreator", grape.random_initialisation, creator.Individual)
#toolbox.register("populationCreator", grape.PI_Grow, creator.Individual)

toolbox.register("evaluate", eval_model_loss_function)

# Tournament selection:
toolbox.register("select", tools.selTournament, tournsize=6)

# Single-point crossover:
toolbox.register("mate", grape.crossover_onepoint)

# Flip-int mutation:
toolbox.register("mutate", grape.mutation_int_flip_per_codon)

POPULATION_SIZE = 1000
MAX_GENERATIONS = 20
P_CROSSOVER = 0.8
P_MUTATION = 0.01
ELITE_SIZE = round(0.01*POPULATION_SIZE)

INIT_GENOME_LENGTH = 30 #used only for random initialisation
random_initilisation = False #put True if you use random initialisation

MAX_INIT_TREE_DEPTH = 10
MIN_INIT_TREE_DEPTH = 3
MAX_TREE_DEPTH = 90
MAX_WRAPS = 0
CODON_SIZE = 255

N_RUNS = 3

for i in range(N_RUNS):
    print()
    print()
    print("Run:", i+1)
    print()

    # create initial population (generation 0):
    if random_initilisation:
        population = toolbox.populationCreator(pop_size=POPULATION_SIZE,
                                           bnf_grammar=BNF_GRAMMAR,
                                           init_genome_length=INIT_GENOME_LENGTH,
                                           max_init_depth=MAX_TREE_DEPTH,
                                           codon_size=CODON_SIZE
                                           )
    else:
        population = toolbox.populationCreator(pop_size=POPULATION_SIZE,
                                           bnf_grammar=BNF_GRAMMAR,
                                           min_init_depth=MIN_INIT_TREE_DEPTH,
                                           max_init_depth=MAX_INIT_TREE_DEPTH,
                                           codon_size=CODON_SIZE
                                            )

    # define the hall-of-fame object:
    hof = tools.HallOfFame(ELITE_SIZE)

    # prepare the statistics object:
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.nanmean)
    stats.register("std", np.nanstd)
    stats.register("min", np.nanmin)
    stats.register("max", np.nanmax)

    # perform the Grammatical Evolution flow:
    population, logbook = algorithms.ge_eaSimpleWithElitism(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                              ngen=MAX_GENERATIONS, elite_size=ELITE_SIZE,
                                              bnf_grammar=BNF_GRAMMAR, codon_size=CODON_SIZE,
                                              max_tree_depth=MAX_TREE_DEPTH,
                                              points_train=[X_train, Y_train],
                                              points_test=[X_test, Y_test],
                                              stats=stats, halloffame=hof, verbose=False)

    import textwrap
    best = hof.items[0].phenotype
    print("Best individual: \n","\n".join(textwrap.wrap(best,80)))
    print("\nTraining Fitness: ", hof.items[0].fitness.values[0])
    print("Test Fitness: ", fitness_eval(hof.items[0], [X_test,Y_test])[0])
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

    fitness_test = logbook.select("fitness_test")

    best_ind_nodes = logbook.select("best_ind_nodes")
    avg_nodes = logbook.select("avg_nodes")

    best_ind_depth = logbook.select("best_ind_depth")
    avg_depth = logbook.select("avg_depth")

    structural_diversity = logbook.select("structural_diversity")

    import csv
    import random
    r = random.randint(1,1e10)

    header = ['gen', 'invalid', 'avg', 'std', 'min', 'max', 'fitness_test',
              'best_ind_length', 'avg_length',
              'best_ind_nodes', 'avg_nodes',
              'best_ind_depth', 'avg_depth',
              'avg_used_codons', 'best_ind_used_codons',
              'structural_diversity',
              'selection_time', 'generation_time']
    with open("results/" + str(r) + ".csv", "w", encoding='UTF8', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(header)
        for value in range(len(max_fitness_values)):
            writer.writerow([gen[value], invalid[value], mean_fitness_values[value],
                             std_fitness_values[value], min_fitness_values[value],
                             max_fitness_values[value],
                             fitness_test[value],
                             best_ind_length[value],
                             avg_length[value],
                             best_ind_nodes[value],
                             avg_nodes[value],
                             best_ind_depth[value],
                             avg_depth[value],
                             avg_used_codons[value],
                             best_ind_used_codons[value],
                             structural_diversity[value],
                             selection_time[value],
                             generation_time[value]])
