from deap import base, creator, algorithms, tools, gp
import random
import misc

toolbox = base.Toolbox()
def initialize_gp_environment(fitness_function, tone_count=3, tournament_size=4, cxpb=0.5, mutpb=0.5):
    # This is an optimization problem; individuals will attempt to minimize "error" as defined in the evaluate function
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # Adding all of the necessary components to the "toolbox"
    # This section sums up to:
    #    - Each individual has 4 floating point attributes for each tone
    #    - Defines the initialization bounds using uniformly randomly-generated values from "reasonable" ranges
    #      (frequency of tone, duration of tone, when to start the tone w/in the audio and how much to muzzle the tones (which are LOUD))
    #    - Defines the "population" as a list of "individuals"

    # Normalized version of attributes
    toolbox.register("attr_freq", random.uniform, 0, 1)
    toolbox.register("attr_duration", random.uniform, 0, 1)
    toolbox.register("attr_offset", random.uniform, 0, 1)
    toolbox.register("attr_muzzle", random.uniform, 0, 1)

    # Add them to individuals and set population to a list of individuals
    toolbox.register("individual", tools.initCycle, creator.Individual, 
                     (toolbox.attr_freq, toolbox.attr_duration, toolbox.attr_offset, toolbox.attr_muzzle), n=tone_count)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Define some parameters for how the population will evolve
    toolbox.register("mate", tools.cxUniform, indpb=cxpb)
    toolbox.register("mutate", tools.mutPolynomialBounded, indpb=mutpb, eta=1.0, low=0.0, up=1.0)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)
    toolbox.register("evaluate", fitness_function)
    return
    
def run_gp(starting_population=None, number_of_generations=40, population_size=20, cxpb=0.5, mutpb=0.5):
    misc.resetTimer()
    
    # Primary Return Datastructures
    population_history = [] # List of populations, initial population plus one per evolutionary step (generation)
    fitness_history = []
    
    # Initialize the population
    pop = None
    
    if starting_population is None:
        pop = toolbox.population(n=population_size)
    else:
        pop = starting_population

    population_history.append(pop)
    
    # Evaluate the fitnss of the entire population
    fitnesses = map(toolbox.evaluate, pop)
    fitness_history.append(fitnesses)
    
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    # Now, evolve the population...
    for g in range(number_of_generations):

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))

        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values
    
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Setup for next generation and save the results
        pop[:] = offspring
        population_history.append(offspring)
        fitness_history.append(fitnesses)

        print(f"Completed generation {g+1} of {number_of_generations}")
        misc.elapsedTime()
    
    return population_history, fitness_history