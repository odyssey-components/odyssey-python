"""Half-open seed intervals used by training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SeedRange:
    start: int
    count: int

    @property
    def end(self) -> int:
        return self.start + self.count

    def __contains__(self, seed: int) -> bool:
        return self.start <= seed < self.end

    def overlaps(self, other: SeedRange) -> bool:
        return self.count > 0 and other.count > 0 and self.start < other.end and other.start < self.end
