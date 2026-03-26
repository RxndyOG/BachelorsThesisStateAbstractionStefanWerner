from collections import deque
import numpy as np

class Detector:
    def __init__(self, max_depth=10, reverse_output=True):
        self.max_depth = max_depth
        self.reverse_output = reverse_output

        self.operations = {
            "r": state_rotate,
            #"m": state_mirror,
            "x": state_mirror_x,
            "y": state_mirror_y,
            "d": state_double,
        }

    def _state_to_key(self, state):
        return tuple(map(tuple, state))

    def _finalize_operations(self, operations):
        if operations is None:
            return None

        ops = operations.copy()

        if len(ops) > 0 and ops[-1] == "n":
            core = ops[:-1]
        else:
            core = ops

        if self.reverse_output:
            core = core[::-1]

        return core + ["n"]

    def detect(self, state, pState):
        bfs_result = self.detect_BFS(state, pState)
        if bfs_result is not None:
            return self._finalize_operations(bfs_result)

        dfs_result = self.detect_DFS(state, pState)
        if dfs_result is not None:
            return self._finalize_operations(dfs_result)

        return None

    def detect_BFS(self, state, pState, max_depth=None):
        if max_depth is None:
            max_depth = self.max_depth

        if np.array_equal(state, pState):
            return ["n"]

        queue = deque()
        visited = set()

        queue.append((pState, []))
        visited.add(self._state_to_key(pState))

        while queue:
            current_state, operations = queue.popleft()

            if len(operations) >= max_depth:
                continue

            for op_name, op_func in self.operations.items():
                new_state = op_func(current_state)

                if np.array_equal(new_state, current_state):
                    continue

                key = self._state_to_key(new_state)
                if key in visited:
                    continue

                new_operations = operations + [op_name]

                if np.array_equal(new_state, state):
                    return new_operations + ["n"]

                visited.add(key)
                queue.append((new_state, new_operations))

        return None

    def detect_DFS(self, state, pState, operations=None, visited=None, depth=0):
        if operations is None:
            operations = []

        if visited is None:
            visited = set()

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

    def apply_operations(self, parent_state, operation_list, stored_reversed=True):
        current_state = parent_state.copy()

        ops = operation_list.copy()

        if len(ops) > 0 and ops[-1] == "n":
            core = ops[:-1]
        else:
            core = ops

        # Wenn gespeichert als reversed, dann vor Anwendung wieder zurückdrehen
        if stored_reversed:
            core = core[::-1]

        for op_code in core:
            if op_code not in self.operations:
                raise ValueError(f"Unbekannte Operation: {op_code}")
            current_state = self.operations[op_code](current_state)

        return current_state

    def apply_operations_from_string(self, parent_state, operation_string, stored_reversed=True):
        return self.apply_operations(parent_state, list(operation_string), stored_reversed=stored_reversed)

    def apply_action_operations(self, parent_q_values, operation_list, stored_reversed=True):
        current_q = parent_q_values.copy()

        ops = operation_list.copy()

        if len(ops) > 0 and ops[-1] == "n":
            core = ops[:-1]
        else:
            core = ops

        if stored_reversed:
            core = core[::-1]

        for op_code in core:
            if op_code not in self.action_operations:
                raise ValueError(f"Unbekannte Action-Operation: {op_code}")
            current_q = self.action_operations[op_code](current_q)

        return current_q

    def apply_action_operations_from_string(self, parent_q_values, operation_string, stored_reversed=True):
        return self.apply_action_operations(parent_q_values, list(operation_string), stored_reversed=stored_reversed)
    
def state_rotate(state):
    return np.rot90(state, k=1, axes=(0, 1))

def state_rotate_back(state):
    return np.rot90(state, k=-1, axes=(0, 1))

def state_double(state):
    doubled_state = state.copy()
    for i in range(len(doubled_state)):
        for j in range(len(doubled_state[i])):
            if doubled_state[i][j] != 0:
                doubled_state[i][j] *= 2
    return doubled_state

def state_divide(state):
    divided = state.copy()
    divided = divided // 2
    return divided.astype(int)

def state_edge(state):
    
    edged_state = state.copy()

    for i in edged_state:
        for j in range(len(i)):
            if j > 0:
                if (i[j] == 0 and i[j-1] != 0 and i[j-1] != 1):
                    i[j] = 1
            if j < len(i) - 1:
                if (i[j] == 0 and i[j+1] != 0 and i[j+1] != 1):
                    i[j] = 1
    
    for j in range(len(edged_state[0])):
        for i in range(len(edged_state)):
            if i > 0:
                if (edged_state[i][j] == 0 and edged_state[i-1][j] != 0 and edged_state[i-1][j] != 1):
                    edged_state[i][j] = 1
            if i < len(edged_state) - 1:
                if (edged_state[i][j] == 0 and edged_state[i+1][j] != 0 and edged_state[i+1][j] != 1):
                    edged_state[i][j] = 1
    
    for i in edged_state:
        for j in range(len(i)):
            if i[j] == 1:
                i[j] = 2

    for j in range(len(edged_state[0])):
        for i in range(len(edged_state)):
            if edged_state[i][j] == 1:
                edged_state[i][j] = 2

    return edged_state

def state_edge_right(state):
    edged_state = state.copy()

    for i in edged_state:
        for j in range(len(i)):
            if j > 0:
                if (i[j] == 0 and i[j-1] != 0 and i[j-1] != 1):
                    i[j] = 1
    return edged_state

def state_edge_left(state):
    edged_state = state.copy()

    for i in edged_state:
        for j in range(len(i)):
            if j < len(i) - 1:
                if (i[j] == 0 and i[j+1] != 0 and i[j+1] != 1):
                    i[j] = 1

    for i in edged_state:
        for j in range(len(i)):
            if i[j] == 1:
                i[j] = 2

    return edged_state

def state_edge_down(state):
    edged_state = state.copy()

    for j in range(len(edged_state[0])):
        for i in range(len(edged_state)):
            if i > 0:
                if (edged_state[i][j] == 0 and edged_state[i-1][j] != 0 and edged_state[i-1][j] != 1):
                    edged_state[i][j] = 1

    for j in range(len(edged_state[0])):
        for i in range(len(edged_state)):
            if edged_state[i][j] == 1:
                edged_state[i][j] = 2

    return edged_state

def state_edge_up(state):
    edged_state = state.copy()

    for j in range(len(edged_state[0])):
        for i in range(len(edged_state)):
            if i < len(edged_state) - 1:
                if (edged_state[i][j] == 0 and edged_state[i+1][j] != 0 and edged_state[i+1][j] != 1):
                    edged_state[i][j] = 1

    for j in range(len(edged_state[0])):
        for i in range(len(edged_state)):
            if edged_state[i][j] == 1:
                edged_state[i][j] = 2

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
    # m = x dann y = 180° Rotation
    q_values = mirror_x_actions(q_values)
    q_values = mirror_y_actions(q_values)
    return q_values

def main():

    state = np.array([[0, 0, 4], 
                      [0, 4, 8], 
                      [4, 0, 8]])

    pState = np.array([[2, 0, 0], 
                       [4, 2, 0], 
                       [4, 0, 2]])

    #print("Original State:")
    #print(state)
    #rotated_state = state_rotate(state)
    #print("Rotated State:")
    #print(rotated_state)
    #rotated_back_state = state_rotate_back(rotated_state)
    #print("Rotated Back State:")
    #print(rotated_back_state)
    #double_state = state_double(state)
    #print("Doubled State:")
    #print(double_state)
    #print("Divided State:")
    #divided_state = state_divide(double_state)
    #print(divided_state)
    #print("Edge State:")
    #edge_state = state_edge(state)
    #print(edge_state)
    #print("Mirrored State:")
    #mirrored_state = state_mirror(state)
    #print(mirrored_state)

    det = Detector()
    operations = det.detect(state, pState)

    print("Detected Operations:")
    print(operations)

    op = Operation()
    transformed_state = op.apply_operations(pState, operations)

    print("Transformed State:")
    print(transformed_state)

    pass

if __name__ == "__main__":
    print("Recommendation: Run the 'main.py' file to execute the program.")
    main()