from collections import deque
import numpy as np

class Detector:
    def __init__(self, max_depth=10):
        self.max_depth = max_depth

        self.operations = {
            "r": state_rotate,
            "m": state_mirror,
            "x": state_mirror_x,
            "y": state_mirror_y,
            "d": state_double,
            "e": state_edge,
        }

    def _state_to_key(self, state):
        return tuple(map(tuple, state))

    def detect(self, state, pState):
        bfs_result = self.detect_BFS(state, pState)
        if bfs_result is not None:
            return bfs_result

        dfs_result = self.detect_DFS(state, pState)
        if dfs_result is not None:
            return dfs_result

        return None

    def detect_BFS(self, state, pState):
        if np.array_equal(state, pState):
            return ["n"]

        queue = deque()
        visited = set()

        queue.append((pState, []))
        visited.add(self._state_to_key(pState))

        while queue:
            current_state, operations = queue.popleft()

            if len(operations) >= self.max_depth:
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

    def print_operations(self):
        operations = []
        for op in self.operations:
            if op != "n":
                operations.append(f"{op}: {self.operations[op].__name__}")
        return "\n".join(operations)
            

    def apply_operations(self, parent_state, operation_list):

        current_state = parent_state.copy()

        for op_code in operation_list:
            if op_code not in self.operations:
                raise ValueError(f"Unbekannte Operation: {op_code}")

            if op_code == "n":
                break

            current_state = self.operations[op_code](current_state)

        return current_state

    def apply_operations_from_string(self, parent_state, operation_string):
        return self.apply_operations(parent_state, list(operation_string))

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