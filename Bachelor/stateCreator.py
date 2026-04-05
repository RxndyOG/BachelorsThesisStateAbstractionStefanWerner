import csv
import os
from typing import List, Tuple, Dict, Set
import numpy as np


class RootStateGenerator:
    """
    Erzeugt Root-States für 2048-artige Grids nach festen Schemen.

    Ziel:
    - strukturell sinnvolle Basiszustände erzeugen
    - später kann ein echter State gegen diese Root-States gematcht werden
    - Root-States sind normalisiert (2,4,8,16,...) statt beliebiger absoluter Werte
    """

    def __init__(
        self,
        size: int = 4,
        max_non_zero: int = None,
        include_rotations: bool = True,
        include_mirrors: bool = True,
    ):
        self.size = size
        self.max_non_zero = max_non_zero if max_non_zero is not None else size * size
        self.include_rotations = include_rotations
        self.include_mirrors = include_mirrors

    # -------------------------------------------------
    # BASIC HELPERS
    # -------------------------------------------------
    def empty_grid(self) -> np.ndarray:
        return np.zeros((self.size, self.size), dtype=int)

    def state_to_key(self, state: np.ndarray) -> Tuple[int, ...]:
        return tuple(state.flatten())

    def key_to_state(self, key: Tuple[int, ...]) -> np.ndarray:
        return np.array(key, dtype=int).reshape(self.size, self.size)

    def normalized_values(self, count: int, reverse: bool = False) -> List[int]:
        """
        Erzeugt normalisierte 2048-Werte:
        2,4,8,16,...

        count=4 -> [2,4,8,16]
        """
        values = [2 ** (i + 1) for i in range(count)]
        if reverse:
            values.reverse()
        return values

    def place_values(self, coordinates: List[Tuple[int, int]], reverse: bool = False) -> np.ndarray:
        """
        Setzt entlang der gegebenen Koordinaten normalisierte Werte.
        """
        grid = self.empty_grid()
        values = self.normalized_values(len(coordinates), reverse=reverse)

        for (r, c), value in zip(coordinates, values):
            grid[r, c] = value

        return grid

    # -------------------------------------------------
    # TRANSFORMATIONS
    # -------------------------------------------------
    def all_variants(self, state: np.ndarray) -> List[np.ndarray]:
        """
        Erzeugt optionale Rotationen und Spiegelungen.
        Entfernt Dubletten.
        """
        variants = []

        rotations = [state]
        if self.include_rotations:
            rotations = [np.rot90(state, k=k) for k in range(4)]

        for rot in rotations:
            variants.append(rot)

            if self.include_mirrors:
                variants.append(np.fliplr(rot))
                variants.append(np.flipud(rot))

        # Deduplicate
        unique = []
        seen = set()
        for var in variants:
            key = self.state_to_key(var)
            if key not in seen:
                seen.add(key)
                unique.append(var)

        return unique

    # -------------------------------------------------
    # PATH / COORDINATE GENERATORS
    # -------------------------------------------------
    def corner_cluster_paths(self) -> List[List[Tuple[int, int]]]:
        """
        Erzeugt Cluster-Pfade aus einer Ecke heraus.
        Beispiel 4x4:
        (0,0), (0,1), (1,0), (1,1), ...
        """
        paths = []

        # Basis: nach Manhattan-Distanz vom Ursprung sortiert
        coords = [(r, c) for r in range(self.size) for c in range(self.size)]
        coords.sort(key=lambda x: (x[0] + x[1], x[0], x[1]))

        for length in range(1, min(self.max_non_zero, len(coords)) + 1):
            paths.append(coords[:length])

        return paths

    def snake_path(self) -> List[Tuple[int, int]]:
        """
        Snake/S-Schlange:
        Zeile 0 links->rechts
        Zeile 1 rechts->links
        usw.
        """
        coords = []
        for r in range(self.size):
            row = [(r, c) for c in range(self.size)]
            if r % 2 == 1:
                row.reverse()
            coords.extend(row)
        return coords

    def diagonal_paths(self) -> List[List[Tuple[int, int]]]:
        """
        Erzeugt diagonale/anti-diagonale Pfade unterschiedlicher Länge.
        """
        paths = []

        main_diag = [(i, i) for i in range(self.size)]
        anti_diag = [(i, self.size - 1 - i) for i in range(self.size)]

        for length in range(1, min(self.max_non_zero, len(main_diag)) + 1):
            paths.append(main_diag[:length])
            paths.append(anti_diag[:length])

        return paths

    def row_monotonic_paths(self) -> List[List[Tuple[int, int]]]:
        """
        Vollständige oder partielle monotone Zeilen.
        """
        paths = []

        for r in range(self.size):
            row = [(r, c) for c in range(self.size)]
            for length in range(1, min(self.max_non_zero, self.size) + 1):
                paths.append(row[:length])

        return paths

    def column_monotonic_paths(self) -> List[List[Tuple[int, int]]]:
        """
        Vollständige oder partielle monotone Spalten.
        """
        paths = []

        for c in range(self.size):
            col = [(r, c) for r in range(self.size)]
            for length in range(1, min(self.max_non_zero, self.size) + 1):
                paths.append(col[:length])

        return paths

    def staircase_paths(self) -> List[List[Tuple[int, int]]]:
        """
        Treppenförmige Muster.
        Beispiel 4x4:
        (0,0)
        (0,1),(1,1)
        (0,2),(1,2),(2,2)
        ...
        """
        paths = []

        coords = []
        for c in range(self.size):
            for r in range(c + 1):
                coords.append((r, c))

        for length in range(1, min(self.max_non_zero, len(coords)) + 1):
            paths.append(coords[:length])

        return paths

    # -------------------------------------------------
    # SCHEMA GENERATORS
    # -------------------------------------------------
    def generate_corner_cluster(self) -> List[Dict]:
        results = []
        for coords in self.corner_cluster_paths():
            for reverse in [False, True]:
                base = self.place_values(coords, reverse=reverse)
                for variant in self.all_variants(base):
                    results.append({
                        "schema": "corner_cluster",
                        "filled": len(coords),
                        "descending": int(reverse),
                        "state": variant
                    })
        return results

    def generate_snake(self) -> List[Dict]:
        results = []
        full_path = self.snake_path()

        for length in range(1, min(self.max_non_zero, len(full_path)) + 1):
            coords = full_path[:length]
            for reverse in [False, True]:
                base = self.place_values(coords, reverse=reverse)
                for variant in self.all_variants(base):
                    results.append({
                        "schema": "snake",
                        "filled": length,
                        "descending": int(reverse),
                        "state": variant
                    })
        return results

    def generate_diagonal(self) -> List[Dict]:
        results = []
        for coords in self.diagonal_paths():
            for reverse in [False, True]:
                base = self.place_values(coords, reverse=reverse)
                for variant in self.all_variants(base):
                    results.append({
                        "schema": "diagonal",
                        "filled": len(coords),
                        "descending": int(reverse),
                        "state": variant
                    })
        return results

    def generate_row_monotonic(self) -> List[Dict]:
        results = []
        for coords in self.row_monotonic_paths():
            for reverse in [False, True]:
                base = self.place_values(coords, reverse=reverse)
                for variant in self.all_variants(base):
                    results.append({
                        "schema": "row_monotonic",
                        "filled": len(coords),
                        "descending": int(reverse),
                        "state": variant
                    })
        return results

    def generate_column_monotonic(self) -> List[Dict]:
        results = []
        for coords in self.column_monotonic_paths():
            for reverse in [False, True]:
                base = self.place_values(coords, reverse=reverse)
                for variant in self.all_variants(base):
                    results.append({
                        "schema": "column_monotonic",
                        "filled": len(coords),
                        "descending": int(reverse),
                        "state": variant
                    })
        return results

    def generate_staircase(self) -> List[Dict]:
        results = []
        for coords in self.staircase_paths():
            for reverse in [False, True]:
                base = self.place_values(coords, reverse=reverse)
                for variant in self.all_variants(base):
                    results.append({
                        "schema": "staircase",
                        "filled": len(coords),
                        "descending": int(reverse),
                        "state": variant
                    })
        return results

    # -------------------------------------------------
    # MAIN GENERATION
    # -------------------------------------------------
    def generate_all(self) -> List[Dict]:
        raw = []
        raw.extend(self.generate_corner_cluster())
        raw.extend(self.generate_snake())
        raw.extend(self.generate_diagonal())
        raw.extend(self.generate_row_monotonic())
        raw.extend(self.generate_column_monotonic())
        raw.extend(self.generate_staircase())

        # globale Dubletten entfernen
        deduped = []
        seen = set()

        for item in raw:
            key = self.state_to_key(item["state"])
            if key not in seen:
                seen.add(key)
                deduped.append(item)

        # root_id vergeben
        for i, item in enumerate(deduped):
            item["root_id"] = i

        return deduped

    # -------------------------------------------------
    # SAVE / PRINT
    # -------------------------------------------------
    def save_to_csv(self, roots: List[Dict], filepath: str = "./Data/", filename: str = None):
        os.makedirs(filepath, exist_ok=True)

        if filename is None:
            filename = f"root_states_{self.size}x{self.size}.csv"

        full_path = os.path.join(filepath, filename)

        with open(full_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow([
                "root_id",
                "schema",
                "filled",
                "descending",
                "state"
            ])

            for item in roots:
                state_str = ",".join(map(str, self.state_to_key(item["state"])))
                writer.writerow([
                    item["root_id"],
                    item["schema"],
                    item["filled"],
                    item["descending"],
                    state_str
                ])

        print(f"Saved {len(roots)} root states to: {full_path}")

    def print_examples(self, roots: List[Dict], amount: int = 10):
        print(f"\nShowing {min(amount, len(roots))} example root states:\n")
        for item in roots[:amount]:
            print(
                f"root_id={item['root_id']} | "
                f"schema={item['schema']} | "
                f"filled={item['filled']} | "
                f"descending={item['descending']}"
            )
            print(item["state"])
            print("-" * 40)


def main():
    # Beispiele:
    # size=2  -> kleines Testsetup
    # size=4  -> klassisches 2048
    generator = RootStateGenerator(
        size=4,
        max_non_zero=8,          # z. B. nur bis 8 gefüllte Felder
        include_rotations=True,
        include_mirrors=True
    )

    roots = generator.generate_all()
    generator.print_examples(roots, amount=368)
    generator.save_to_csv(
        roots,
        filepath="./Data/",
        filename="generated_root_states_4x4.csv"
    )


if __name__ == "__main__":
    main()