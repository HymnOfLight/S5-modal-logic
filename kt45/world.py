"""World factory: bulk-generate a 100k-fact knowledge base.

We mix three flavours of facts so the demo is interesting and the
forward chainer has something to do:

* **Geography** — countries, capitals, continents, neighbours, oceans.
* **Mathematics** — primality, parity, divisibility, squares.
* **Commonsense** — animals, food, colours, weather facts.

The factory is deterministic given a seed, so the demo is fully
reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .facts import FactBase, Proposition
from .reasoning import Rule
from .truth import T


CONTINENTS: List[str] = [
    "Africa", "Antarctica", "Asia", "Europe",
    "North_America", "Oceania", "South_America",
]


# A small but real geography seed. The factory expands this with
# synthetic regions to reach the requested fact count.
SEED_COUNTRIES: List[Tuple[str, str, str]] = [
    # (country, capital, continent)
    ("China", "Beijing", "Asia"),
    ("Japan", "Tokyo", "Asia"),
    ("India", "New_Delhi", "Asia"),
    ("Korea", "Seoul", "Asia"),
    ("Vietnam", "Hanoi", "Asia"),
    ("Thailand", "Bangkok", "Asia"),
    ("Indonesia", "Jakarta", "Asia"),
    ("Iran", "Tehran", "Asia"),
    ("Turkey", "Ankara", "Asia"),
    ("Saudi_Arabia", "Riyadh", "Asia"),
    ("France", "Paris", "Europe"),
    ("Germany", "Berlin", "Europe"),
    ("Spain", "Madrid", "Europe"),
    ("Italy", "Rome", "Europe"),
    ("United_Kingdom", "London", "Europe"),
    ("Russia", "Moscow", "Europe"),
    ("Poland", "Warsaw", "Europe"),
    ("Sweden", "Stockholm", "Europe"),
    ("Netherlands", "Amsterdam", "Europe"),
    ("Greece", "Athens", "Europe"),
    ("USA", "Washington", "North_America"),
    ("Canada", "Ottawa", "North_America"),
    ("Mexico", "Mexico_City", "North_America"),
    ("Cuba", "Havana", "North_America"),
    ("Brazil", "Brasilia", "South_America"),
    ("Argentina", "Buenos_Aires", "South_America"),
    ("Chile", "Santiago", "South_America"),
    ("Peru", "Lima", "South_America"),
    ("Colombia", "Bogota", "South_America"),
    ("Egypt", "Cairo", "Africa"),
    ("Nigeria", "Abuja", "Africa"),
    ("South_Africa", "Pretoria", "Africa"),
    ("Kenya", "Nairobi", "Africa"),
    ("Morocco", "Rabat", "Africa"),
    ("Australia", "Canberra", "Oceania"),
    ("New_Zealand", "Wellington", "Oceania"),
]


SEED_NEIGHBOURS: List[Tuple[str, str]] = [
    ("China", "Russia"), ("China", "India"), ("China", "Vietnam"),
    ("France", "Germany"), ("France", "Spain"), ("France", "Italy"),
    ("Germany", "Poland"), ("Germany", "Netherlands"),
    ("USA", "Canada"), ("USA", "Mexico"),
    ("Brazil", "Argentina"), ("Brazil", "Peru"), ("Brazil", "Colombia"),
    ("Egypt", "Nigeria"),  # not real neighbours; stand-in for demo
]


COLOURS = ["Red", "Green", "Blue", "Yellow", "Purple", "Orange", "Black", "White"]
ANIMALS = [
    "Dog", "Cat", "Cow", "Horse", "Sheep", "Pig", "Chicken",
    "Lion", "Tiger", "Wolf", "Bear", "Eagle", "Sparrow",
    "Salmon", "Shark", "Octopus", "Bee", "Ant",
]
MAMMALS = {"Dog", "Cat", "Cow", "Horse", "Sheep", "Pig",
           "Lion", "Tiger", "Wolf", "Bear"}
FOODS = ["Rice", "Bread", "Apple", "Banana", "Carrot",
         "Tomato", "Potato", "Cheese", "Egg", "Milk"]


@dataclass
class WorldFactory:
    """Build a fact base of (approximately) ``target_size`` propositions."""

    target_size: int = 100_000
    seed: int = 42

    def build(self) -> Tuple[FactBase, List[Rule]]:
        fb = FactBase()
        self._geography(fb)
        self._math(fb)
        self._commonsense(fb)
        # Pad with synthetic-but-true facts to hit the target size.
        self._synthetic_pad(fb)
        rules = self._rules()
        return fb, rules

    # ------------------------------------------------------------------
    def _geography(self, fb: FactBase) -> None:
        # Continents
        for c in CONTINENTS:
            fb.assert_fact(Proposition("continent", (c,)), T)
        # Countries + capitals
        for country, capital, continent in SEED_COUNTRIES:
            fb.assert_fact(Proposition("country", (country,)), T)
            fb.assert_fact(Proposition("capital_of", (capital, country)), T)
            fb.assert_fact(Proposition("located_in", (country, continent)), T)
            fb.assert_fact(Proposition("city", (capital,)), T)
        # Neighbours (symmetric)
        for a, b in SEED_NEIGHBOURS:
            fb.assert_fact(Proposition("neighbour", (a, b)), T)
            fb.assert_fact(Proposition("neighbour", (b, a)), T)
        # Oceans
        for o in ["Pacific", "Atlantic", "Indian", "Arctic", "Southern"]:
            fb.assert_fact(Proposition("ocean", (o,)), T)

    def _math(self, fb: FactBase) -> None:
        # Numbers up to 5_000 — primes, parity, squares, divisibility.
        primes = self._sieve(5_000)
        for n in range(1, 5_001):
            num = str(n)
            fb.assert_fact(Proposition("number", (num,)), T)
            if n % 2 == 0:
                fb.assert_fact(Proposition("even", (num,)), T)
            else:
                fb.assert_fact(Proposition("odd", (num,)), T)
            if n in primes:
                fb.assert_fact(Proposition("prime", (num,)), T)
            r = int(n ** 0.5)
            if r * r == n:
                fb.assert_fact(Proposition("square", (num,)), T)
            if n % 5 == 0:
                fb.assert_fact(Proposition("divisible_by_5", (num,)), T)
            if n % 10 == 0:
                fb.assert_fact(Proposition("divisible_by_10", (num,)), T)

    @staticmethod
    def _sieve(n: int) -> set:
        if n < 2:
            return set()
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if sieve[i]:
                for j in range(i * i, n + 1, i):
                    sieve[j] = False
        return {i for i, ok in enumerate(sieve) if ok}

    def _commonsense(self, fb: FactBase) -> None:
        for a in ANIMALS:
            fb.assert_fact(Proposition("animal", (a,)), T)
            if a in MAMMALS:
                fb.assert_fact(Proposition("mammal", (a,)), T)
        for f in FOODS:
            fb.assert_fact(Proposition("food", (f,)), T)
        for c in COLOURS:
            fb.assert_fact(Proposition("colour", (c,)), T)
        # Some commonsense relations.
        fb.assert_fact(Proposition("eats", ("Cat", "Fish")), T)
        fb.assert_fact(Proposition("eats", ("Dog", "Bone")), T)
        fb.assert_fact(Proposition("eats", ("Lion", "Meat")), T)
        fb.assert_fact(Proposition("makes_sound", ("Dog", "Bark")), T)
        fb.assert_fact(Proposition("makes_sound", ("Cat", "Meow")), T)
        fb.assert_fact(Proposition("makes_sound", ("Cow", "Moo")), T)

    def _synthetic_pad(self, fb: FactBase) -> None:
        """Pad the world with synthetic but well-formed facts.

        We mint ``region_i`` / ``city_i`` / ``link(region_i, region_j)``
        facts until the size target is reached. This dominates the
        100k count and gives the chainer real work to do on the
        ``transitive_link`` rule (see :meth:`_rules`).
        """
        i = 0
        target = self.target_size
        while len(fb) < target:
            r = f"R{i:06d}"
            fb.assert_fact(Proposition("region", (r,)), T)
            if len(fb) >= target:
                break
            # Chain region links: R_i links to R_{i+1}
            if i > 0:
                prev = f"R{i-1:06d}"
                fb.assert_fact(Proposition("link", (prev, r)), T)
            i += 1
            # Tag a few regions to seed the chainer.
            if i % 1000 == 0:
                fb.assert_fact(Proposition("hub", (r,)), T)

    # ------------------------------------------------------------------
    def _rules(self) -> List[Rule]:
        """A modest rule set covering each fact flavour."""
        return [
            Rule(
                name="capital_implies_city",
                premises=(Proposition("capital_of", ("X", "Y")),),
                conclusion=Proposition("city", ("X",)),
            ),
            Rule(
                name="country_in_continent_is_in_world",
                premises=(Proposition("located_in", ("X", "Y")),),
                conclusion=Proposition("country", ("X",)),
            ),
            Rule(
                name="mammal_is_animal",
                premises=(Proposition("mammal", ("X",)),),
                conclusion=Proposition("animal", ("X",)),
            ),
            Rule(
                name="square_is_number",
                premises=(Proposition("square", ("X",)),),
                conclusion=Proposition("number", ("X",)),
            ),
            Rule(
                name="prime_is_number",
                premises=(Proposition("prime", ("X",)),),
                conclusion=Proposition("number", ("X",)),
            ),
            Rule(
                name="div10_is_div5",
                premises=(Proposition("divisible_by_10", ("X",)),),
                conclusion=Proposition("divisible_by_5", ("X",)),
            ),
            Rule(
                name="link_in_world",
                premises=(Proposition("link", ("X", "Y")),),
                conclusion=Proposition("region", ("Y",)),
            ),
        ]
