from typing import List, Any, Dict, Optional
from collections import defaultdict

BASE = 8000


class FractalTier:
    """
    One hierarchical counter that realises the infinite nested-doll collapse:
        SUP_1  = floor(D0 / 8000)
        cache_2 = floor(SUP_1 / 8000)
        SUP_2  = floor(cache_2 / 8000)
        …
    """
    def __init__(self, base: int = BASE):
        self.base = base
        self.digits: List[int] = [0]          # digits[0] = lowest tier (raw)

    def add(self, n: int = 1) -> None:
        """Add n occurrences (or n identical tokens)."""
        if n <= 0:
            return
        self.digits[0] += n
        self._propagate(0)

    def _propagate(self, level: int) -> None:
        while self.digits[level] >= self.base:
            carry, rem = divmod(self.digits[level], self.base)
            self.digits[level] = rem
            if level + 1 == len(self.digits):
                self.digits.append(0)
            self.digits[level + 1] += carry
            level += 1                       # continue upward

    def total(self) -> int:
        """Reconstruct the exact original count (Python int is unlimited)."""
        total = 0
        power = 1
        for d in self.digits:
            total += d * power
            power *= self.base
        return total

    def markers(self) -> Dict[str, int]:
        """Human-readable view of the active higher-order markers."""
        names = []
        for i, d in enumerate(self.digits):
            if i == 0:
                names.append(("raw / D0", d))
            elif i % 2 == 1:
                names.append((f"SUP_{i//2 + 1}", d))
            else:
                names.append((f"cache_{i//2}", d))
        return {name: cnt for name, cnt in names if cnt}

    def __repr__(self) -> str:
        return f"FractalTier(total={self.total():_}, markers={self.markers()})"


class MultiCache:
    """
    Practical multi-token version: each distinct string / token gets its own
    fractal tier.  Identical items collapse independently.
    """
    def __init__(self, base: int = BASE):
        self.base = base
        self.tiers: Dict[Any, FractalTier] = defaultdict(lambda: FractalTier(base))

    def ingest(self, token: Any, count: int = 1) -> None:
        self.tiers[token].add(count)

    def ingest_stream(self, stream) -> None:
        for item in stream:
            self.ingest(item)

    def stats(self) -> Dict[Any, dict]:
        return {
            tok: {
                "total": tier.total(),
                "markers": tier.markers(),
                "storage_slots": len(tier.digits)
            }
            for tok, tier in self.tiers.items()
        }

    def grand_total(self) -> int:
        return sum(t.total() for t in self.tiers.values())


# ------------------------------------------------------------------
# Demo – reproduces the numbers in the original description
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Single-token extreme case
    s = FractalTier()
    s.add(8000)                 # first collapse → SUP_1 = 1
    print("After 8000:", s)

    s.add(8000 * 8000 - 8000)   # fill up to next collapse
    print("After 8000²:", s)

    # Multi-token stream
    cache = MultiCache()
    # simulate a long stream of duplicates
    cache.ingest("hello", 12_345_678)
    cache.ingest("world", 8000**3 + 42)
    cache.ingest("hello", 8000)          # another collapse for "hello"

    print("\nMulti-cache stats:")
    for tok, info in cache.stats().items():
        print(f"  {tok!r}: total={info['total']:_}  "
              f"slots={info['storage_slots']}  markers={info['markers']}")

    print("\nGrand total items represented:", f"{cache.grand_total():_}")
    print("Memory footprint of the counters themselves stays tiny "

          "(a few dozen integers even for astronomical N).")
