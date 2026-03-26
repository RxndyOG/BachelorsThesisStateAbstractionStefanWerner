from collections import deque
import numpy as np


def state_rotate(state):
    return np.rot90(state, k=1, axes=(0, 1))

def state_rotate_back(state):
    return np.rot90(state, k=-1, axes=(0, 1))

def state_double(state):
    return np.add(state, state)

def state_divide(state):
    return np.divide(state, 2)

def state_edge(state):
    edged_state = state.copy()

    for i in range(edged_state.shape[0]):
        for j in range(edged_state.shape[1]):
            if edged_state[i, j] != 0:
                continue

            neighbors = []
            if j > 0:
                neighbors.append(edged_state[i, j - 1])
            if j < edged_state.shape[1] - 1:
                neighbors.append(edged_state[i, j + 1])
            if i > 0:
                neighbors.append(edged_state[i - 1, j])
            if i < edged_state.shape[0] - 1:
                neighbors.append(edged_state[i + 1, j])

            if any(v not in (0, 1) for v in neighbors):
                edged_state[i, j] = 2

    return edged_state

def state_edge_right(state):
    edged_state = state.copy()

    for i in range(edged_state.shape[0]):
        for j in range(1, edged_state.shape[1]):
            if edged_state[i, j] == 0 and edged_state[i, j - 1] not in (0, 1):
                edged_state[i, j] = 2

    return edged_state

def state_edge_left(state):
    edged_state = state.copy()

    for i in range(edged_state.shape[0]):
        for j in range(edged_state.shape[1] - 1):
            if edged_state[i, j] == 0 and edged_state[i, j + 1] not in (0, 1):
                edged_state[i, j] = 2

    return edged_state

def state_edge_down(state):
    edged_state = state.copy()

    for i in range(1, edged_state.shape[0]):
        for j in range(edged_state.shape[1]):
            if edged_state[i, j] == 0 and edged_state[i - 1, j] not in (0, 1):
                edged_state[i, j] = 2

    return edged_state

def state_edge_up(state):
    edged_state = state.copy()

    for i in range(edged_state.shape[0] - 1):
        for j in range(edged_state.shape[1]):
            if edged_state[i, j] == 0 and edged_state[i + 1, j] not in (0, 1):
                edged_state[i, j] = 2

    return edged_state

def state_mirror(state):
    mirrored_state = state_mirror_x(state)
    return np.fliplr(mirrored_state)

def state_mirror_x(state):
    return np.flipud(state)

def state_mirror_y(state):
    return np.fliplr(state)

def rotate_actions_ccw(q_values):
    return {
        "up": q_values["right"],
        "right": q_values["down"],
        "down": q_values["left"],
        "left": q_values["up"],
    }

def mirror_x_actions(q_values):
    return {
        "up": q_values["down"],
        "right": q_values["right"],
        "down": q_values["up"],
        "left": q_values["left"],
    }

def mirror_y_actions(q_values):
    return {
        "up": q_values["up"],
        "right": q_values["left"],
        "down": q_values["down"],
        "left": q_values["right"],
    }

def mirror_actions(q_values):
    q_values = mirror_x_actions(q_values)
    q_values = mirror_y_actions(q_values)
    return q_values

class Detector:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth

        self.operations = {
            "r": state_rotate,
            #"m": state_mirror,
            "x": state_mirror_x,
            "y": state_mirror_y,
            "d": state_double,
            #"e": state_edge,
        }

    def _state_to_key(self, state):
        return tuple(state.flatten())

    def generate_reachable_states(self, parent_state, max_depth=None):
        """
        Erzeugt ALLE erreichbaren Zustände von parent_state bis max_depth.
        Rückgabe:
            dict[state_key] = [op1, op2, ..., "n"]
        """
        depth_limit = max_depth if max_depth is not None else self.max_depth

        parent_state = np.asarray(parent_state, dtype=int)
        parent_key = self._state_to_key(parent_state)

        reachable = {
            parent_key: ["n"]
        }

        queue = deque()
        queue.append((parent_state, []))

        visited = {parent_key}

        while queue:
            current_state, operations = queue.popleft()

            if len(operations) >= depth_limit:
                continue

            for op_name, op_func in self.operations.items():
                new_state = op_func(current_state)

                if np.array_equal(new_state, current_state):
                    continue

                new_key = self._state_to_key(new_state)
                if new_key in visited:
                    continue

                new_operations = operations + [op_name]
                reachable[new_key] = new_operations + ["n"]

                visited.add(new_key)
                queue.append((new_state, new_operations))

        return reachable

    def detect_BFS(self, state, pState, max_depth=None):
        state = np.asarray(state, dtype=int)
        pState = np.asarray(pState, dtype=int)

        target_key = self._state_to_key(state)
        reachable = self.generate_reachable_states(pState, max_depth=max_depth)
        return reachable.get(target_key, None)

    def detect_DFS(self, state, pState, operations=None, visited=None, depth=0):
        if operations is None:
            operations = []
        if visited is None:
            visited = set()

        state = np.asarray(state, dtype=int)
        pState = np.asarray(pState, dtype=int)

        if np.array_equal(state, pState):
            return operations + ["n"]

        if depth >= self.max_depth:
            return None

        key = self._state_to_key(pState)
        if key in visited:
            return None
        visited.add(key)

        for op_name, op_func in self.operations.items():
            new_state = op_func(pState)

            if np.array_equal(new_state, pState):
                continue

            result = self.detect_DFS(
                state=state,
                pState=new_state,
                operations=operations + [op_name],
                visited=visited.copy(),
                depth=depth + 1
            )

            if result is not None:
                return result

        return None

    def detect(self, state, pState):
        bfs_result = self.detect_BFS(state, pState, max_depth=self.max_depth)
        if bfs_result is not None:
            return bfs_result

        dfs_result = self.detect_DFS(state, pState)
        if dfs_result is not None:
            return dfs_result

        return None

    def detect_as_string(self, state, pState):
        result = self.detect(state, pState)
        if result is None:
            return None
        return "".join(result)


class Operation:
    def __init__(self):
        self.operations = {
            "r": state_rotate,
            "m": state_mirror,
            "x": state_mirror_x,
            "y": state_mirror_y,
            "d": state_double,
            "e": state_edge,
            "n": lambda s: s.copy(),
        }

        self.action_operations = {
            "r": rotate_actions_ccw,
            "m": mirror_actions,
            "x": mirror_x_actions,
            "y": mirror_y_actions,
            "d": lambda q: q.copy(),
            "e": lambda q: q.copy(),
            "n": lambda q: q.copy(),
        }

    def print_operations(self):
        operations = []
        for op in self.operations:
            if op != "n":
                operations.append(f"{op}: {self.operations[op].__name__}")
        return "\n".join(operations)

    def apply_operations(self, parent_state, operation_list):
        current_state = np.asarray(parent_state, dtype=int).copy()

        for op_code in operation_list:
            if op_code not in self.operations:
                raise ValueError(f"Unbekannte Operation: {op_code}")

            if op_code == "n":
                break

            current_state = self.operations[op_code](current_state)

        return current_state

    def apply_operations_from_string(self, parent_state, operation_string):
        return self.apply_operations(parent_state, list(operation_string))

    def apply_action_operations(self, parent_q_values, operation_list):
        current_q = parent_q_values.copy()

        for op_code in operation_list:
            if op_code not in self.action_operations:
                raise ValueError(f"Unbekannte Action-Operation: {op_code}")

            if op_code == "n":
                break

            current_q = self.action_operations[op_code](current_q)

        return current_q

    def apply_action_operations_from_string(self, parent_q_values, operation_string):
        return self.apply_action_operations(parent_q_values, list(operation_string))