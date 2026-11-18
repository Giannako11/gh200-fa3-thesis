"""Shared workload definitions and the benchmark routing policy."""

from dataclasses import dataclass
from itertools import product

EXPLORE = (1, 10, 50, 100)
SHAPES = tuple((m, k, m) for m, k in (
    (128, 128), (256, 256), (1024, 1024), (2048, 2048),
    (256, 1024), (1024, 256), (128, 2048), (2048, 128),
))
DEVICES = ("grace", "hopper")


@dataclass(frozen=True)
class Case:
    m: int
    k: int
    n: int
    a_pct: int
    b_pct: int

    @property
    def name(self):
        return f"{self.m}x{self.k}x{self.n}_a{self.a_pct:03d}_b{self.b_pct:03d}"

    @property
    def strategy(self):
        # Limit the generic CSR×CSR path, whose workspace can scale with
        # intermediate products. Both devices use the same representation.
        if (self.a_pct <= 10 and self.b_pct <= 10
                and self.m * self.k * self.n * self.a_pct * self.b_pct <= 500_000_000 * 10_000):
            return "csr_csr"
        if self.a_pct <= 10 or self.b_pct <= 10:
            return "csr_dense" if self.a_pct <= self.b_pct else "dense_csr"
        return "dense_dense"


def cases(shapes=SHAPES, densities=EXPLORE):
    # Visit the heavy corner first so the runtime estimate gets usable
    # anchors early and any resource problem is visible immediately.
    return [Case(*shape, a, b) for shape in sorted(shapes, key=lambda s: -s[0]*s[1]*s[2])
            for a, b in product(reversed(densities), repeat=2)]
