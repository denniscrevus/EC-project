import os
from genotype                                 import GenotypeSerializer, develop
from revolve2.core.database                   import open_async_database_sqlite
from revolve2.core.database.serializers       import DbFloat
from revolve2.core.optimization.ea.generic_ea import DbEAOptimizerIndividual
from revolve2.core.optimization.ea.generic_ea import DbEAOptimizerGeneration
from revolve2.runners.mujoco import ModularRobotRerunner
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.future import select

async def main() -> None:
    """Run the script."""
    db = open_async_database_sqlite("./database0")

    async with AsyncSession(db) as session:
        desired_generation_id = 11

        sorted_individuals = (
            await session.execute(
                select(DbEAOptimizerGeneration, DbEAOptimizerIndividual)
                .filter(DbEAOptimizerGeneration.generation_index == desired_generation_id,
                        DbEAOptimizerGeneration.individual_id == DbEAOptimizerIndividual.individual_id
                        )
            )
        )

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())