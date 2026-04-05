import numpy as np
from collections import deque


def state_rotate(state):
    return np.rot90(state)   # 90° gegen den Uhrzeigersinn


def state_rotate_back(state):
    return np.rot90(state, k=-1)


def state_double(state):
    doubled = state.copy()
    doubled = doubled + doubled
    return doubled.astype(int)


def state_divide(state):
    divided = state.copy()
    divided = divided // 2
    return divided.astype(int)


def state_mirror_horizontal(state):
    """
    Spiegelt links <-> rechts
    Beispiel:
    [[2, 4],
     [8,16]]
    ->
    [[4, 2],
     [16,8]]
    """
    return np.fliplr(state).astype(int)


def state_mirror_vertical(state):
    """
    Spiegelt oben <-> unten
    Beispiel:
    [[2, 4],
     [8,16]]
    ->
    [[8,16],
     [2, 4]]
    """
    return np.flipud(state).astype(int)


def state_add_edge(state):
    """
    Edge-Operation:
    Jede 0, die orthogonal (oben, unten, links, rechts)
    an ein Nicht-Null-Feld angrenzt, wird zu 2.

    Beispiel:
    [[2, 0],
     [0, 0]]
    ->
    [[2, 2],
     [2, 0]]
    """
    rows, cols = state.shape
    new_state = state.copy()

    for r in range(rows):
        for c in range(cols):
            if state[r, c] != 0:
                continue

            neighbors = []

            if r > 0:
                neighbors.append(state[r - 1, c])  # oben
            if r < rows - 1:
                neighbors.append(state[r + 1, c])  # unten
            if c > 0:
                neighbors.append(state[r, c - 1])  # links
            if c < cols - 1:
                neighbors.append(state[r, c + 1])  # rechts

            if any(value != 0 for value in neighbors):
                new_state[r, c] = 2

    return new_state.astype(int)


def action_rotate(actions):
    """
    Rotiert die Aktionen passend zu np.rot90(state).

    Wenn der State 90° CCW rotiert wird, dann gilt:
    new["up"]    = old["right"]
    new["left"]  = old["up"]
    new["down"]  = old["left"]
    new["right"] = old["down"]
    """
    return {
        "up": actions["right"],
        "down": actions["left"],
        "left": actions["up"],
        "right": actions["down"],
    }


def action_double(actions):
    """
    Beim Verdoppeln des States ändern sich die Richtungen nicht.
    """
    return actions.copy()


def action_mirror_horizontal(actions):
    """
    Spiegelung links <-> rechts:
    up/down bleiben gleich, left/right werden getauscht.
    """
    return {
        "up": actions["up"],
        "down": actions["down"],
        "left": actions["right"],
        "right": actions["left"],
    }


def action_mirror_vertical(actions):
    """
    Spiegelung oben <-> unten:
    left/right bleiben gleich, up/down werden getauscht.
    """
    return {
        "up": actions["down"],
        "down": actions["up"],
        "left": actions["left"],
        "right": actions["right"],
    }


def action_edge(actions):
    """
    Bei Edge ändert sich die Orientierung nicht,
    daher bleiben die Actions gleich.
    """
    return actions.copy()


class PreCalculator:
    def __init__(self):
        self.operations = {
            "r": (state_rotate, action_rotate),
            "d": (state_double, action_double),
            "mh": (state_mirror_horizontal, action_mirror_horizontal),
            "mv": (state_mirror_vertical, action_mirror_vertical),
            #"e": (state_add_edge, action_edge),
        }

    def state_to_key(self, state):
        return tuple(state.flatten())

    def actions_to_key(self, actions):
        return (actions["up"], actions["down"], actions["left"], actions["right"])

    def dig_deeper(self, q_table, max_depth=5):
        """
        q_table = [state, actions]

        Gibt eine Liste zurück mit Einträgen:
        [
            {
                "state": ...,
                "actions": ...,
                "depth": ...,
                "path": ...
            },
            ...
        ]
        """
        results = []
        visited = set()

        start_state, start_actions = q_table

        queue = deque()
        queue.append((start_state, start_actions, 0, ""))

        while queue:
            current_state, current_actions, depth, path = queue.popleft()

            key = (self.state_to_key(current_state), self.actions_to_key(current_actions))
            if key in visited:
                continue
            visited.add(key)

            results.append({
                "state": current_state.copy(),
                "actions": current_actions.copy(),
                "depth": depth,
                "path": path
            })

            if depth >= max_depth:
                continue

            for op_name, (state_func, action_func) in self.operations.items():
                new_state = state_func(current_state)
                new_actions = action_func(current_actions)
                new_path = path + op_name

                queue.append((new_state, new_actions, depth + 1, new_path))

        return results


def main():
    parent_state = np.array([
        [2, 4],
        [0, 2]
    ])

    actions = {
        "up": 2.34,
        "down": 1.56,
        "left": 0.78,
        "right": 3.90
    }

    q_table = [parent_state, actions]

    precalc = PreCalculator()
    states = precalc.dig_deeper(q_table, max_depth=3)

    for i, item in enumerate(states):
        print(f"State {i+1}/{len(states)}")
        print(f"Depth: {item['depth']}")
        print(f"Path: {item['path']}")
        print(item["state"])
        print(item["actions"])
        print("-" * 40)


if __name__ == "__main__":
    main()