import numpy as np
from collections import deque

def state_rotate(state):
    return np.rot90(state, k=1, axes=(0, 1))

def state_rotate_back(state):
    return np.rot90(state, k=-1, axes=(0, 1))

def state_double(state):
    doubled = state.copy()
    doubled = doubled + doubled
    return doubled.astype(int)

def state_divide(state):
    divided = state.copy()
    divided = divided // 2
    return divided.astype(int)

class Detector:
    def __init__(self):
        self.operations = {
            #"r": state_rotate_back,
            "d": state_divide,
        }
        
    def state_to_key(self, state):
        return tuple(state.flatten())    
    
    def find_operation(self, state1, state2):
        for key, operation in self.operations.items():
            if np.array_equal(operation(state1), state2):
                return key
        return None
    
    def breadth_first_search(self, target_state, start_state, max_depth=3):
        if np.array_equal(start_state, target_state):
            return ["n"]

        queue = deque()
        visited = set()

        queue.append((start_state, []))
        visited.add(self.state_to_key(start_state))

        while queue:
            current_state, path = queue.popleft()

            if len(path) >= max_depth:
                continue

            for op_key, op_func in self.operations.items():
                new_state = op_func(current_state)
                new_path = path + [op_key]

                if np.array_equal(new_state, target_state):
                    return new_path + ["n"]

                key = self.state_to_key(new_state)
                if key not in visited:
                    visited.add(key)
                    queue.append((new_state, new_path))

        return None

class Operator:
    def __init__(self):
        self.operations = {
            #"r": state_rotate,
            "d": state_double,
        }

        self.action_operations = {
            #"r": self.rotate_actions,
            "d": self.double_actions,
        }

    def rotate_actions(self, actions):
        return {
            "up": actions["right"],
            "right": actions["down"],
            "down": actions["left"],
            "left": actions["up"],
        }

    def double_actions(self, actions):
        # Beim Verdoppeln ändern sich die Richtungen nicht
        return actions.copy()

    def generate_child_states(self, state, path=None):
        if path is None:
            path = []
            print("Error: Path should not be None. Initializing to empty list.")

        for i in path:
            if i == "n":
                continue

            operation = self.operations.get(i)
            if operation is not None:
                state = operation(state)

        return state

    def generate_child_actions(self, actions, path=None):
        if path is None:
            path = []
            print("Error: Path should not be None. Initializing to empty list.")

        current_actions = actions.copy()

        for i in path:
            if i == "n":
                continue

            operation = self.action_operations.get(i)
            if operation is not None:
                current_actions = operation(current_actions)

        return current_actions
        
def main():
    
    parent_state = np.array([[0, 2, 4], 
                             [8, 16, 32], 
                             [64, 128, 256]])
    child_state = np.array([[0, 4, 8],
                            [16, 32, 64], 
                            [128, 256, 512]])

    child_state = state_double(child_state)
    child_state = state_rotate(child_state)
    print("Child State:\n", child_state)

    detector = Detector()
    path = detector.breadth_first_search(target_state=parent_state, start_state=child_state, max_depth=5)

    print("Path from child to parent:", path)

    operator = Operator()
    reconstructed_state = operator.generate_child_states(parent_state, path)
    print("Child State:\n", reconstructed_state)


if __name__ == "__main__":
    print("Recommendation: Run the 'main.py' file to execute the program.")
    main()
