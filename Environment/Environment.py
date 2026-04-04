import numpy as np
import random

class Environment():
    def __init__(self, size = 4):
        self.grid = None
        self.actions = []
        self.size = size
        pass

    def reset(self):
        self.actions = []
        self.grid = np.zeros((self.size, self.size), dtype=int)
        self.spawn_tile()
        self.spawn_tile()
        return self.get_state()

    def step(self, action="up"):
        
        if action not in ["up", "down", "left", "right"]:
            raise ValueError(f"Invalid action: {action}")
        
        old_grid = self.grid.copy()
        self.grid, reward = self.merge(self.grid, action)

        moved = not np.array_equal(old_grid, self.grid)

        if moved:
            self.spawn_tile()

        done = self.is_done()
        next_state = self.get_state()
        info = {"moved": moved}

        return next_state, reward, done, info

    def ask_for_action(self):
        valid_actions = self.find_available_actions()
        if not valid_actions:
            return None
        action = input(f"Enter action ({', '.join(valid_actions)}): ")
        while action not in valid_actions:
            print("Invalid action. Please try again.")
            action = input(f"Enter action ({', '.join(valid_actions)}): ")
        return action

    def get_state(self):
        return self.grid.copy()

    def has_won(self, wanted_tile=2048):
        return np.any(self.grid == wanted_tile)

    def is_done(self):
        if np.any(self.grid == 0):
            return False
        if len(self.find_available_actions()) > 0:
            return False
        return True
    
    def spawn_tile(self):
        empty_cells = np.argwhere(self.grid == 0)
        if len(empty_cells) == 0:
            return
        y, x = empty_cells[np.random.randint(len(empty_cells))]
        self.grid[y, x] = random.choices([2, 4], weights=(90, 10), k=1)[0] 

    def find_available_actions_empty_check(self, grid, actions, rotated=False):
        for i in grid:
            x = 0
            while x < len(i):
                if i[x] != 0:
                    if x > 0 and i[x - 1] > 0 and i[x] == i[x - 1]:
                        if rotated:
                            if "up" not in actions:
                                actions.append("up")
                        else:
                            if "left" not in actions:
                                actions.append("left")
                    if x > 0 and i[x - 1] == 0:
                        if rotated:
                            if "up" not in actions:
                                actions.append("up")
                        else:
                            if "left" not in actions:
                                actions.append("left")
                    if x < len(i) - 1 and i[x + 1] > 0 and i[x] == i[x + 1]:
                        if rotated:
                            if "down" not in actions:
                                actions.append("down")
                        else:
                            if "right" not in actions:
                                actions.append("right")
                    if x < len(i) - 1 and i[x + 1] == 0:
                        if rotated:
                            if "down" not in actions:
                                actions.append("down")
                        else:
                            if "right" not in actions:
                                actions.append("right")
                x += 1

        return actions

    def find_available_actions(self):
        
        actions = []

        actions = self.find_available_actions_empty_check(grid=self.grid, actions=actions, rotated=False)

        grid = np.rot90(self.grid)

        actions = self.find_available_actions_empty_check(grid=grid, actions=actions, rotated=True)

        return actions

    def merge_row_right(self, row):
        original_len = len(row)
        row = row[row != 0]
        reward = 0
        new_row = []
        skip = False

        for i in range(len(row) - 1, -1, -1):
            if skip:
                skip = False
                continue

            if i > 0 and row[i] == row[i - 1]:
                merged_value = row[i] * 2
                new_row.append(merged_value)
                reward += merged_value
                skip = True
            else:
                new_row.append(row[i])

        new_row += [0] * (original_len - len(new_row))
        new_row.reverse()

        return np.array(new_row), reward

    def merge_row_left(self, row):
        original_len = len(row)
        row = row[row != 0]
        reward = 0
        new_row = []
        skip = False

        for i in range(len(row)):
            if skip:
                skip = False
                continue

            if i < len(row) - 1 and row[i] == row[i + 1]:
                merged_value = row[i] * 2
                new_row.append(merged_value)
                reward += merged_value
                skip = True
            else:
                new_row.append(row[i])

        new_row += [0] * (original_len - len(new_row))

        return np.array(new_row), reward

    def merge(self, grid, action):
        reward = 0
        grid = grid.copy()

        match action:
            case "left":
                for r in range(len(grid)):
                    grid[r], row_reward = self.merge_row_left(grid[r])
                    empty_tiles = np.sum(self.grid == 0)
                    reward = row_reward - 1 + 0.1 * empty_tiles

            case "right":
                for r in range(len(grid)):
                    grid[r], row_reward = self.merge_row_right(grid[r])
                    empty_tiles = np.sum(self.grid == 0)
                    reward = row_reward - 1 + 0.1 * empty_tiles

            case "up":
                grid = np.rot90(grid, k=-1, axes=(0, 1))
                for r in range(len(grid)):
                    grid[r], row_reward = self.merge_row_right(grid[r])
                    empty_tiles = np.sum(self.grid == 0)
                    reward = row_reward - 1 + 0.1 * empty_tiles
                grid = np.rot90(grid, k=1, axes=(0, 1))

            case "down":
                grid = np.rot90(grid, k=-1, axes=(0, 1))
                for r in range(len(grid)):
                    grid[r], row_reward = self.merge_row_left(grid[r])
                    empty_tiles = np.sum(self.grid == 0)
                    reward = row_reward - 1 + 0.1 * empty_tiles
                grid = np.rot90(grid, k=1, axes=(0, 1))

        self.grid = grid
        return grid, reward

    def test_import(self):
        print(self.grid)

    def get_score(self):
        return np.sum(self.grid)

def main():

    episodes = 100
    env = Environment(size=4)
    env.reset()
    total_reward = 0
    test = 0
    done = False
    while test == 0:
        test = test + 1
        for i in range(episodes):

            print(env.get_state())
            actions = env.find_available_actions()
            if not actions:
                print("No actions left")
                break
            #action = random.choice(actions)
            action = env.ask_for_action()

            state ,reward, done, info = env.step(action=action)

            print(f"Action: {action}, Reward: {reward}, Done: {done}")
            total_reward += reward

            #if env.has_won(16):
            #    done = True

            if done:
                break
            
        print("Total Score: ",env.get_score())
        print("Total Reward: ", total_reward)
        
if __name__ == "__main__":
    print("Recommendation: Run the 'main.py' file to execute the program.")
    main()