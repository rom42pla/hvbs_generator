import numpy as np
import pydantic
from typing import Union, Optional

from pydantic.validators import IntEnum


class Rarity(IntEnum):
    common = 0
    uncommon = 1
    rare = 2
    very_rare = 3
    legendary = 4


class Material(pydantic.BaseModel):
    name: str
    rarity: Rarity


class Filament(pydantic.BaseModel):
    name: str
    rarity: Rarity


class Pattern(pydantic.BaseModel):
    name: str
    rarity: Rarity


class Gem(pydantic.BaseModel):
    name: str
    rarity: Rarity


class Ring(pydantic.BaseModel):
    material: Material
    filament: Filament
    pattern: Pattern
    gem: Gem
    rarity: Optional[Rarity]

    def get_rarity(self):
        rarity = int(np.floor(
            np.mean([self.material.rarity, self.filament.rarity, self.pattern.rarity, self.gem.rarity])
        ))
        return Rarity(rarity)

    def __str__(self):
        return f"Anello di {self.material.name} con filamento {self.filament.name}, fantasia {self.pattern.name} e {self.gem.name}"
