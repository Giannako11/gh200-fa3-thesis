"""Shared workload definitions and the benchmark routing policy."""

from dataclasses import dataclass

SHAPES = tuple((m * 1024, k * 1024, m * 1024) for m, k in (
    (2, 2), (4, 4), (16, 16), (32, 32), (4, 16), (16, 4), (2, 32), (32, 2),
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


def cases(shapes=SHAPES):
    ordered = sorted(shapes, key=lambda shape: -(shape[0] * shape[1] * shape[2]))
    return [Case(*shape, density, density) for density in (100, 0) for shape in ordered]
