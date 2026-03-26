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


def action_rotate(actions):
    """
    Rotiert die Aktionen passend zu np.rot90(state).

    Wenn der State 90° CCW rotiert wird, dann gilt:
    original up    -> new left
    original left  -> new down
    original down  -> new right
    original right -> new up

    Also:
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
    Die Action-Namen bleiben also gleich.
    """
    return actions.copy()


class PreCalculator:
    def __init__(self):
        self.operations = {
            "r": (state_rotate, action_rotate),
            "d": (state_double, action_double),
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
        [2,4],
        [2,2]
    ])

    actions = {
        "up": 2.34,
        "down": 1.56,
        "left": 0.78,
        "right": 3.90
    }

    q_table = [parent_state, actions]

    precalc = PreCalculator()
    states = precalc.dig_deeper(q_table, max_depth=10)

    for i, item in enumerate(states):
        print(f"State {i+1}/{len(states)}")
        print(f"Depth: {item['depth']}")
        print(f"Path: {item['path']}")
        print(item["state"])
        print(item["actions"])
        print("-" * 40)


if __name__ == "__main__":
    main()