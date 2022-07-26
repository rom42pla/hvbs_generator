import random
from pprint import pprint
import itertools
from typing import Optional

from fastapi import APIRouter, Query

from data.models.items import Pattern, Filament, Gem, Material, Ring, Rarity


def get_router(generator):
    router = APIRouter()

    @router.get("/ring")
    async def get_random_ring(
            rarity: Rarity
    ) -> Ring:
        rings_data = generator.db.get_data()["rings"]
        rings_data = {
            "filament": [Filament(**d) for d in rings_data["filaments"]],
            "gem": [Gem(**d) for d in rings_data["gems"]],
            "material": [Material(**d) for d in rings_data["materials"]],
            "pattern": [Pattern(**d) for d in rings_data["patterns"]]
        }
        # pprint(rings_data)
        components_order = list(rings_data.keys())
        best_ring: Optional[Ring] = None
        for components in itertools.product(*[random.sample(rings_data[k], k=len(rings_data[k])) for k in components_order]):
            pprint({k: v for k, v in zip(components_order, components)})
            ring = Ring(**{k: v for k, v in zip(components_order, components)})
            if ring.get_rarity() == rarity:
                return ring
            elif best_ring is None or ring.get_rarity() > best_ring.get_rarity():
                best_ring = ring
        return best_ring

    return router


