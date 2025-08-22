from __future__ import annotations
import random
from typing import Optional

ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"

def base36_to_int(s: str) -> int:
    s = s.strip().lower()
    if not s or any(ch not in ALPHABET for ch in s):
        raise ValueError("Seed inválida (solo base36).")
    val = 0
    for ch in s:
        val = val * 36 + ALPHABET.index(ch)
    return val

def int_to_base36(n: int) -> str:
    if n < 0:
        raise ValueError("n debe ser no negativo")
    if n == 0:
        return "0"
    out = []
    while n > 0:
        n, r = divmod(n, 36)
        out.append(ALPHABET[r])
    return "".join(reversed(out))

class RNG:
    """RNG determinista a partir de una seed base36 o un entero."""
    def __init__(self, seed: Optional[str|int]):
        if seed is None:
            seed_int = random.SystemRandom().randint(1, 2**63-1)
        elif isinstance(seed, int):
            seed_int = seed
        else:
            seed_int = base36_to_int(seed)
        self._seed_int = seed_int
        self._rand = random.Random(seed_int)

    @property
    def seed_int(self) -> int:
        return self._seed_int

    def randint(self, a: int, b: int) -> int:
        return self._rand.randint(a, b)

    def random(self) -> float:
        return self._rand.random()

    def choice(self, seq):
        return self._rand.choice(seq)

    def shuffle(self, seq):
        return self._rand.shuffle(seq)

    def gauss(self, mu: float, sigma: float) -> float:
        return self._rand.gauss(mu, sigma)
