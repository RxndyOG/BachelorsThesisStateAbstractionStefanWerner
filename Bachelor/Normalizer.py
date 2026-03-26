import numpy as np


def state_rotate_ccw(state):
    return np.rot90(state, k=1, axes=(0, 1))


def state_rotate_cw(state):
    return np.rot90(state, k=-1, axes=(0, 1))


def state_double(state):
    return (state * 2).astype(int)


def state_divide(state):
    return (state // 2).astype(int)


class Normalizer:
    """
    Erzeugt für jeden State:
    - einen kanonischen Parent-State
    - die Operationen von Child -> Parent
    """

    def __init__(self):
        pass

    def state_to_key(self, state):
        return ",".join(map(str, state.flatten()))

    def key_to_state(self, state_key, size):
        values = list(map(int, state_key.split(",")))
        return np.array(values).reshape(size, size)

    def normalize_scale(self, state):
        """
        Teilt den State so lange durch 2,
        bis der kleinste non-zero Wert 2 ist
        (oder der State leer ist).
        
        Returns:
            normalized_state, divide_count
        """
        current = state.copy()
        divide_count = 0

        while True:
            non_zero = current[current > 0]

            if len(non_zero) == 0:
                break

            min_val = non_zero.min()

            if min_val <= 2:
                break

            current = state_divide(current)
            divide_count += 1

        return current, divide_count

    def normalize_state(self, state):
        """
        Normalisiert einen State:
        1. Skalierung
        2. Rotation
        
        Kanonische Form = lexikographisch kleinste Rotation
        der skalierten Version.

        Returns:
            parent_state,
            operations_child_to_parent (z.B. ['d','d','r'])
        """
        scaled_state, divide_count = self.normalize_scale(state)

        rotation_candidates = []
        current = scaled_state.copy()

        for rot_count in range(4):
            key = tuple(current.flatten())
            rotation_candidates.append((key, current.copy(), rot_count))
            current = state_rotate_ccw(current)

        # lexikographisch kleinste Darstellung wählen
        _, best_state, best_rot_count = min(rotation_candidates, key=lambda x: x[0])

        operations = []
        operations.extend(["d"] * divide_count)
        operations.extend(["r"] * best_rot_count)

        return best_state, operations


class Operator:
    """
    Rekonstruiert Child-State und Child-Actions aus Parent + gespeicherten Ops.
    
    Gespeicherte Ops bedeuten:
        Child -> Parent
    Für Rekonstruktion brauchen wir:
        Parent -> Child
    also inverse Operationen in umgekehrter Reihenfolge.
    """

    def __init__(self):
        pass

    def apply_parent_to_child_state(self, state, operations_child_to_parent):
        """
        Invertiert die gespeicherten Operationen.
        Inverse:
            d -> double
            r -> rotate_cw
        """
        current = state.copy()

        for op in reversed(operations_child_to_parent):
            if op == "d":
                current = state_double(current)
            elif op == "r":
                current = state_rotate_cw(current)

        return current

    def rotate_actions_cw(self, actions):
        """
        Parent -> Child bei clockwise Rotation des Boards.
        """
        return {
            "up": actions["left"],
            "right": actions["up"],
            "down": actions["right"],
            "left": actions["down"],
        }

    def rotate_actions_ccw(self, actions):
        """
        Parent -> Child bei counter-clockwise Rotation des Boards.
        """
        return {
            "up": actions["right"],
            "right": actions["down"],
            "down": actions["left"],
            "left": actions["up"],
        }

    def apply_parent_to_child_actions(self, actions, operations_child_to_parent):
        """
        Inverse der gespeicherten Child->Parent Ops auf Actions.
        
        d -> keine Änderung
        r -> inverse ist rotate_cw auf dem State,
             also müssen Actions ebenfalls rotate_cw transformiert werden
        """
        current = actions.copy()

        for op in reversed(operations_child_to_parent):
            if op == "d":
                continue
            elif op == "r":
                current = self.rotate_actions_cw(current)

        return current