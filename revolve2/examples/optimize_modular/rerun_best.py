"""Visualize and simulate the best robot from the optimization process."""
import copy

from genotype                                 import GenotypeSerializer, develop
from revolve2.core.database                   import open_async_database_sqlite
from revolve2.core.database.serializers       import DbFloat
from revolve2.core.optimization.ea.generic_ea import DbEAOptimizerIndividual
from revolve2.core.optimization.ea.generic_ea import DbEAOptimizerGeneration
from revolve2.runners.mujoco import ModularRobotRerunner
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.future import select
from itertools import tee
from genotype import Genotype, GenotypeSerializer, crossover, develop, mutate
from pyrr import Quaternion, Vector3
from revolve2.core.physics.environment_actor_controller import (
    EnvironmentActorController,
)
from revolve2.core.physics.running import (
    ActorState,
    Batch,
    Environment,
    PosedActor,
    Runner,
)
from revolve2.runners.mujoco import LocalRunner

async def main() -> None:
    """Run the script."""
    db = open_async_database_sqlite("./database0")

    async with AsyncSession(db) as session:
        desired_generation_id = 20


        sorted_individuals = (
            await session.execute(
                select(DbEAOptimizerGeneration, DbEAOptimizerIndividual)
                .filter(DbEAOptimizerGeneration.generation_index == desired_generation_id,
                        DbEAOptimizerGeneration.individual_id == DbEAOptimizerIndividual.individual_id
                        )
            )
        )

        assert sorted_individuals is not None

        sorted_individuals, copy_sorted_individuals = tee(sorted_individuals)
        sorted_individuals, copy_copy_sorted_individuals = tee(sorted_individuals)

        dist_obj_ids = []
        for individual in sorted_individuals:
            dist_obj_ids.append(individual[1].distance_obj_id)

        sorted_individuals = copy_sorted_individuals

        remaining_power_ids = []
        for individual in sorted_individuals:
            remaining_power_ids.append(individual[1].remaining_power_id)

        sorted_individuals = copy_copy_sorted_individuals

        distances = []

        for distance_id in dist_obj_ids:
            distance_iterator = (
                await session.execute(
                    select(DbFloat)
                    .where(DbFloat.id == distance_id)
                )
            )

            assert distance_id is not None

            distances_temp = [distance[0].value for distance in distance_iterator]
            distances.extend(distances_temp)

        remaining_powers = []

        for remaining_power_id in remaining_power_ids:
            remaining_power_iterator = (
                await session.execute(
                    select(DbFloat)
                    .where(DbFloat.id == remaining_power_id)
                )
            )

            assert remaining_power_iterator is not None

            remaining_power_temp = [remaining_power[0].value for remaining_power in remaining_power_iterator]
            remaining_powers.extend(remaining_power_temp)

        # print(len(distances))
        # print(len(remaining_powers))

        # max_fitness = -1
        #
        # best_individuals = []
        #
        # for individual in sorted_individuals:
        #     print(individual)
        #     if max_fitness == -1:
        #         max_fitness = individual[1].value
        #
        #     if individual[1].value == max_fitness or individual[1].value == (max_fitness - 1):
        #         best_individuals.append(individual)

        robots_to_simulate = []
        counter = 0
        genotypes = []

        for best_individual in sorted_individuals:
            # if remaining_powers[counter] != 20 or remaining_powers[counter] == 11.6:
            #     counter += 1
            #     continue
            # print(f"fitness: {best_individual[2].value}")

            print(f"Generation = {best_individual[0].generation_index}, {distances[counter]}, {remaining_powers[counter]}")

            genotype = (
                await GenotypeSerializer.from_database(
                    session, [best_individual[1].genotype_id]
                )
            )[0]

            genotypes.append(genotype)
            robots_to_simulate.append(develop(genotype))
            counter += 1

        # for best_individual in sorted_individuals:
        #
        #     counter += 1
        #     if counter == 4:
        #         genotype = (
        #             await GenotypeSerializer.from_database(
        #                 session, [best_individual[1].genotype_id]
        #             )
        #         )[0]
        #
        #         genotypes.append(genotype)
        #         robots_to_simulate.append(develop(genotype))
        #         break

        batch = Batch(
            simulation_time=30,
            sampling_frequency=5,
            control_frequency=60,
        )

        for genotype in genotypes:
            actor, controller = develop(genotype).make_actor_and_controller()
            bounding_box = actor.calc_aabb()
            env = Environment(EnvironmentActorController(controller))
            env.actors.append(
                PosedActor(
                    actor,
                    Vector3(
                        [
                            0.0,
                            0.0,
                            bounding_box.size.z / 2.0 - bounding_box.offset.z,
                        ]
                    ),
                    Quaternion(),
                    [0.0 for _ in controller.get_dof_targets()],
                ),
            )

            batch.environments.append(env)

        runner = LocalRunner(headless=True, num_simulators=64)
        batch_results = await runner.run_batch(batch)
        # print(batch_results)

        rerunner = ModularRobotRerunner()
        await rerunner.rerun(robots_to_simulate, 20)

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
