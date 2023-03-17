"""Setup and running of the optimize modular program."""

import logging
from random import Random

import multineat
import asyncio
from genotype import random as random_genotype
from optimizer import Optimizer
from revolve2.core.database import open_async_database_sqlite
from revolve2.core.optimization import DbId
import os
import time

# random number generator
rng = Random()
rng.seed(6)

#Simulation parameters
SIMULATION_TIME = 20
SAMPLING_FREQUENCY = 5
CONTROL_FREQUENCY = 60

#Settings for population management
POPULATION_SIZE = 100
OFFSPRING_SIZE = 100
CROSSOVER_PROBABILITY = 0.8
MUTATION_PROBABILITY = 0.8
MAX_POWER = 2000
THRESHOLD_DIST = 0.2
THRESHOLD_ENERGY = MAX_POWER - 2

#Time consuming parameters
NUMBER_OF_RUNS = 10
NUM_GENERATIONS = 50

#Mujoco parameters
NUM_INITIAL_MUTATIONS = 9
FORCE_RANGE = 1.0
MAX_PARTS = 17


async def experiment_force_range() -> None:
    #NOT CONSTANT FORCE RANGE!
    global SIMULATION_TIME                                                                                                          
    global SAMPLING_FREQUENCY
    global CONTROL_FREQUENCY
    global POPULATION_SIZE
    global OFFSPRING_SIZE
    global NUM_GENERATIONS
    global CROSSOVER_PROBABILITY
    global MUTATION_PROBABILITY
    global NUMBER_OF_RUNS
    global NUM_INITIAL_MUTATIONS
    global MAX_PARTS
    global MAX_POWER
    global THRESHOLD_DIST
    global THRESHOLD_ENERGY
    FORCE_RANGES = [0.75, 1.0, 1.25]

    for force_range in FORCE_RANGES:
        dir_name = "Experiment_force_range_" + str(force_range)
        if os.path.exists(dir_name) == False:
            os.mkdir(dir_name)

        os.chdir(dir_name)

        for i in range(0, NUMBER_OF_RUNS):
            if os.path.exists("Run " + str(i)) == False:
                os.mkdir("Run " + str(i))

            os.chdir("Run " + str(i))

            logging.info(f"START RUN " + dir_name + " " + str(i))

            # database
            database = open_async_database_sqlite("./database0", create=True)

            # unique database identifier for optimizer
            db_id = DbId.root("optmodular")

            # multineat innovation databases
            innov_db_body = multineat.InnovationDatabase()
            innov_db_brain = multineat.InnovationDatabase()

            initial_population = [
                random_genotype(innov_db_body, innov_db_brain, rng, NUM_INITIAL_MUTATIONS)
                for _ in range(POPULATION_SIZE)
            ]

            maybe_optimizer = await Optimizer.from_database(
                database=database,
                db_id=db_id,
                innov_db_body=innov_db_body,
                innov_db_brain=innov_db_brain,
                rng=rng,
            )
            if maybe_optimizer is not None:
                optimizer = maybe_optimizer
            else:
                optimizer = await Optimizer.new(
                    database=database,
                    db_id=db_id,
                    initial_population=initial_population,
                    rng=rng,
                    innov_db_body=innov_db_body,
                    innov_db_brain=innov_db_brain,
                    simulation_time=SIMULATION_TIME,
                    sampling_frequency=SAMPLING_FREQUENCY,
                    control_frequency=CONTROL_FREQUENCY,
                    num_generations=NUM_GENERATIONS,
                    offspring_size=OFFSPRING_SIZE,
                    crossover_prob=CROSSOVER_PROBABILITY,
                    mutation_prob=MUTATION_PROBABILITY,
                    force_range=force_range,
                    max_parts=MAX_PARTS,
                    max_power=MAX_POWER,
                    threshold_dist=THRESHOLD_DIST,
                    threshold_energy=THRESHOLD_ENERGY,
                )

            logging.info(f"Starting optimization process.. run {i}")

            await optimizer.run()

            logging.info(f"FINISHED run {i}")
            os.chdir("..")

        os.chdir("..")
        logging.info(f"FINISHED experiment " + dir_name)

async def experiment_max_parts_and_battery() -> None:
    #NOT CONSTANT MAX_PARTS, MAX_POWER, THRESHOLD_ENERGY
    global SIMULATION_TIME
    global SAMPLING_FREQUENCY
    global CONTROL_FREQUENCYF
    global POPULATION_SIZE
    global OFFSPRING_SIZE
    global NUM_GENERATIONS
    global CROSSOVER_PROBABILITY
    global MUTATION_PROBABILITY
    global NUMBER_OF_RUNS
    global NUM_INITIAL_MUTATIONS

    global THRESHOLD_DIST
    global FORCE_RANGE
    MAX_POWERS = [129]
    MAX_PARTS = [20]

    for max_power in MAX_POWERS:
        threshold_energy = max_power - 2
        for max_parts in MAX_PARTS:
            dir_name = "Experiment_max_power_" + str(max_power) + "_max_parts" + str(max_parts)
            if os.path.exists(dir_name) == False:
                os.mkdir(dir_name)

            os.chdir(dir_name)

            for i in range(0, NUMBER_OF_RUNS):
                if os.path.exists("Run " + str(i)) == False:
                    os.mkdir("Run " + str(i))

                os.chdir("Run " + str(i))

                logging.info(f"START RUN " + dir_name + " " + str(i))

                # database
                database = open_async_database_sqlite("./database0", create=True)

                # unique database identifier for optimizer
                db_id = DbId.root("optmodular")

                # multineat innovation databases
                innov_db_body = multineat.InnovationDatabase()
                innov_db_brain = multineat.InnovationDatabase()

                initial_population = [
                    random_genotype(innov_db_body, innov_db_brain, rng, NUM_INITIAL_MUTATIONS)
                    for _ in range(POPULATION_SIZE)
                ]

                maybe_optimizer = await Optimizer.from_database(
                    database=database,
                    db_id=db_id,
                    innov_db_body=innov_db_body,
                    innov_db_brain=innov_db_brain,
                    rng=rng,
                )
                if maybe_optimizer is not None:
                    optimizer = maybe_optimizer
                else:
                    optimizer = await Optimizer.new(
                        database=database,
                        db_id=db_id,
                        initial_population=initial_population,
                        rng=rng,
                        innov_db_body=innov_db_body,
                        innov_db_brain=innov_db_brain,
                        simulation_time=SIMULATION_TIME,
                        sampling_frequency=SAMPLING_FREQUENCY,
                        control_frequency=CONTROL_FREQUENCY,
                        num_generations=NUM_GENERATIONS,
                        offspring_size=OFFSPRING_SIZE,
                        crossover_prob=CROSSOVER_PROBABILITY,
                        mutation_prob=MUTATION_PROBABILITY,
                        force_range=FORCE_RANGE,
                        max_parts=max_parts,
                        max_power=max_power,
                        threshold_dist=THRESHOLD_DIST,
                        threshold_energy=threshold_energy,
                    )

                logging.info(f"Starting optimization process.. run {i}")

                await optimizer.run()

                logging.info(f"FINISHED run {i}")
                os.chdir("..")

            os.chdir("..")
            logging.info(f"FINISHED experiment " + dir_name)

async def test_one_generation() -> None:
    global SIMULATION_TIME
    global SAMPLING_FREQUENCY
    global CONTROL_FREQUENCY
    POPULATION_SIZE = 100
    OFFSPRING_SIZE = 100
    NUM_GENERATIONS = 5
    global CROSSOVER_PROBABILITY
    global MUTATION_PROBABILITY
    NUMBER_OF_RUNS = 1
    global NUM_INITIAL_MUTATIONS
    global MAX_PARTS
    MAX_POWER = 2000
    global THRESHOLD_DIST
    global THRESHOLD_ENERGY
    global FORCE_RANGE

    start = time.time()
    dir_name = "Experiment_one_generation"
    if os.path.exists(dir_name) == False:
        os.mkdir(dir_name)

    os.chdir(dir_name)

    for i in range(0, NUMBER_OF_RUNS):
        if os.path.exists("Run " + str(i)) == False:
            os.mkdir("Run " + str(i))

        os.chdir("Run " + str(i))

        logging.info(f"START RUN " + dir_name + " " + str(i))

        # database
        database = open_async_database_sqlite("./database0", create=True)

        # unique database identifier for optimizer
        db_id = DbId.root("optmodular")

        # multineat innovation databases
        innov_db_body = multineat.InnovationDatabase()
        innov_db_brain = multineat.InnovationDatabase()

        initial_population = [
            random_genotype(innov_db_body, innov_db_brain, rng, NUM_INITIAL_MUTATIONS)
            for _ in range(POPULATION_SIZE)
        ]

        maybe_optimizer = await Optimizer.from_database(
            database=database,
            db_id=db_id,
            innov_db_body=innov_db_body,
            innov_db_brain=innov_db_brain,
            rng=rng,
        )
        if maybe_optimizer is not None:
            optimizer = maybe_optimizer
        else:
            optimizer = await Optimizer.new(
                database=database,
                db_id=db_id,
                initial_population=initial_population,
                rng=rng,
                innov_db_body=innov_db_body,
                innov_db_brain=innov_db_brain,
                simulation_time=SIMULATION_TIME,
                sampling_frequency=SAMPLING_FREQUENCY,
                control_frequency=CONTROL_FREQUENCY,
                num_generations=NUM_GENERATIONS,
                offspring_size=OFFSPRING_SIZE,
                crossover_prob=CROSSOVER_PROBABILITY,
                mutation_prob=MUTATION_PROBABILITY,
                force_range=FORCE_RANGE,
                max_parts=MAX_PARTS,
                max_power=MAX_POWER,
                threshold_dist=THRESHOLD_DIST,
                threshold_energy=THRESHOLD_ENERGY,
            )

        logging.info(f"Starting optimization process.. run {i}")

        await optimizer.run()

        logging.info(f"FINISHED run {i}")
        os.chdir("..")

    logging.info(f"FINISHED experiment " + dir_name)
    stop = time.time()
    duration = stop - start
    logging.info(f"TIME IT TOOK FOR 5 GENERATIONS (POP=100, OFF=100, SIM_TIME=20: {duration}")

    f = open("time_taken.txt", "w")
    f.write(f"TIME IT TOOK FOR 5 GENERATIONS (POP=100, OFF=100, SIM_TIME=20: {duration}")
    f.close()

def main() -> None:
    """Run the optimization process."""
    # number of initial mutations for body and brain CPPNWIN networks
    import asyncio

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )

    # asyncio.run(test_one_generation())
    #asyncio.run(experiment_force_range())
    asyncio.run(experiment_max_parts_and_battery())
    # asyncio.run(experiment_max_parts())
    # asyncio.run(experiment_max_power())

    logging.info("Finished optimizing.")


if __name__ == "__main__":
    main()
