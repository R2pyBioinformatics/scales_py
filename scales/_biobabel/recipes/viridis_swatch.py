"""Recipe: sample pal_viridis at 8 positions and print hex codes.

Demonstrates the factory-then-call idiom (`scales.palette_factory_then_call`).
"""

from __future__ import annotations

from scales import pal_viridis


def main(n: int = 8, option: str = "D") -> list[str]:
    vir = pal_viridis(option=option)   # factory: pick style
    return list(vir(n))                # call: ask for n colors


if __name__ == "__main__":
    for hex_code in main():
        print(hex_code)
