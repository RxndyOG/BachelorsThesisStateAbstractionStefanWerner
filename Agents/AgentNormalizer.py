import os
import numpy as np
import pandas as pd

from Bachelor.Normalizer import Normalizer, Operator
from Environment.Environment import Environment


class Agent:
    def __init__(
        self,
        environment,
        filepath="./",
        filename="q-table",
        max_depth=5,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        learning_rate=0.1,
        size = 2
    ):
        self.environment = environment

        # volle Q-Table fürs Learning
        self.q_table = pd.DataFrame(columns=["state", "up", "down", "left", "right"])

        self.q_table_reconstructed = pd.DataFrame(columns=["state", "up", "down", "left", "right"])

        # komprimierte Tabellen fürs separate Speichern
        self.q_table_parent = pd.DataFrame(columns=["Pstate", "up", "down", "left", "right"])
        self.q_table_child = pd.DataFrame(columns=["Pstate", "Cstate", "Operations"])


        self.filepath = filepath
        self.filename = filename

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate

        self.normalizer = Normalizer()
        self.operator = Operator()

        self.max_depth = max_depth
        
        self.size = size

    def state_to_key(self, state):
        state_array = np.array(state)
        return ",".join(map(str, state_array.flatten()))

    def key_to_state_array(self, state_key):
        values = list(map(int, state_key.split(",")))
        return np.array(values).reshape(self.size, self.size)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

    def choose_action(self, state, available_actions):
        if np.random.rand() < self.epsilon:
            return np.random.choice(available_actions)

        state_key = self.state_to_key(state)
        q_values = self.q_table[self.q_table["state"] == state_key]

        if q_values.empty:
            return np.random.choice(available_actions)

        q_values = q_values.iloc[0][available_actions]
        max_q_value = q_values.max()
        best_actions = q_values[q_values == max_q_value].index.tolist()

        return np.random.choice(best_actions)

    def learn(self, state, action, reward, next_state, available_actions):
        state_key = self.state_to_key(state)
        next_state_key = self.state_to_key(next_state)

        if self.q_table[self.q_table["state"] == state_key].empty:
            self.create_q_table_entry(state)

        if self.q_table[self.q_table["state"] == next_state_key].empty:
            self.create_q_table_entry(next_state)

        current_q_rows = self.q_table.loc[self.q_table["state"] == state_key, action]
        if current_q_rows.empty:
            return

        current_q_value = current_q_rows.values[0]

        if available_actions:
            next_rows = self.q_table.loc[
                self.q_table["state"] == next_state_key,
                available_actions
            ]

            if next_rows.empty:
                next_max_q_value = 0.0
            else:
                next_max_q_value = next_rows.max(axis=1).values[0]
        else:
            next_max_q_value = 0.0

        new_q_value = (
            (1 - self.learning_rate) * current_q_value
            + self.learning_rate * (reward + next_max_q_value)
        )

        self.q_table.loc[self.q_table["state"] == state_key, action] = new_q_value

    def create_q_table_entry(self, state):
        state_key = self.state_to_key(state)

        if state_key in self.q_table["state"].values:
            return

        new_entry = {
            "state": state_key,
            "up": 0.0,
            "down": 0.0,
            "left": 0.0,
            "right": 0.0
        }

        self.q_table = pd.concat(
            [self.q_table, pd.DataFrame([new_entry])],
            ignore_index=True
        )

    def divide_q_table(self):
        parent_map = {}
        child_rows = []

        if self.q_table.empty:
            self.q_table_parent = pd.DataFrame(columns=["Pstate", "up", "down", "left", "right"])
            self.q_table_child = pd.DataFrame(columns=["Pstate", "Cstate", "Operations"])
            return

        for _, row in self.q_table.iterrows():
            child_key = row["state"]
            child_state = self.key_to_state_array(child_key)

            parent_state, operations = self.normalizer.normalize_state(child_state)
            parent_key = self.state_to_key(parent_state)

            # Parent speichern, falls noch nicht vorhanden
            if parent_key not in parent_map:
                parent_map[parent_key] = {
                    "Pstate": parent_key,
                    "up": row["up"],
                    "down": row["down"],
                    "left": row["left"],
                    "right": row["right"]
                }

            # Wenn Child != Parent, als Child speichern
            if child_key != parent_key:
                child_rows.append({
                    "Pstate": parent_key,
                    "Cstate": child_key,
                    "Operations": ",".join(operations)
                })

        self.q_table_parent = pd.DataFrame(
            list(parent_map.values()),
            columns=["Pstate", "up", "down", "left", "right"]
        )

        self.q_table_child = pd.DataFrame(
            child_rows,
            columns=["Pstate", "Cstate", "Operations"]
        )
                
    def save_q_table_single(self, filepath=None, filename=None):
        if filepath is None:
            filepath = self.filepath
        if filename is None:
            filename = self.filename

        path = os.path.join(filepath, f"{filename}_single_{self.size}.csv")
        self.q_table.to_csv(path, index=False)    
           
    def save_q_table_parent_child(self, filepath=None, filename=None):
        if filepath is None:
            filepath = self.filepath
        if filename is None:
            filename = self.filename

        parent_path = os.path.join(filepath, f"{filename}_parent_{self.size}.csv")
        child_path = os.path.join(filepath, f"{filename}_child_{self.size}.csv")

        self.q_table_parent.to_csv(parent_path, index=False)
        self.q_table_child.to_csv(child_path, index=False)       
     
    def load_q_table_single(self, filepath=None, filename=None):
        if filepath is None:
            filepath = self.filepath
        if filename is None:
            filename = self.filename

        path = os.path.join(filepath, f"{filename}_single_{self.size}.csv")
        if os.path.exists(path):
            self.q_table = pd.read_csv(path)
        else:
            print(f"File {path} not found. Starting with empty Q-Table.")
            self.q_table = pd.DataFrame(columns=["state", "up", "down", "left", "right"])
     
    def load_q_table_reconstructed(self, filepath=None, filename=None):
        if filepath is None:
            filepath = self.filepath
        if filename is None:
            filename = self.filename

        parent_path = os.path.join(filepath, f"{filename}_parent_{self.size}.csv")
        child_path = os.path.join(filepath, f"{filename}_child_{self.size}.csv")

        if os.path.exists(parent_path) and os.path.exists(child_path):
            self.q_table_parent = pd.read_csv(parent_path)
            self.q_table_child = pd.read_csv(child_path)
            return self.reconstruct_q_table()

        print("Parent/Child files not found.")
        self.q_table_reconstructed = pd.DataFrame(columns=["state", "up", "down", "left", "right"])
        return self.q_table_reconstructed
    
    def save_q_table_reconstructed(self, filepath=None, filename=None):
        if filepath is None:
            filepath = self.filepath
        if filename is None:
            filename = self.filename

        os.makedirs(filepath, exist_ok=True)

        reconstructed_df = self.reconstruct_q_table()
        path = os.path.join(filepath, f"{filename}_reconstructed_{self.size}.csv")
        reconstructed_df.to_csv(path, index=False)
            
    def reconstruct_q_table(self):
        reconstructed_rows = []

        if self.q_table_parent.empty and self.q_table_child.empty:
            self.q_table_reconstructed = pd.DataFrame(columns=["state", "up", "down", "left", "right"])
            return self.q_table_reconstructed

        # Parent-Zeilen direkt übernehmen
        parent_lookup = {}

        for _, row in self.q_table_parent.iterrows():
            parent_lookup[row["Pstate"]] = {
                "up": row["up"],
                "down": row["down"],
                "left": row["left"],
                "right": row["right"]
            }

            reconstructed_rows.append({
                "state": row["Pstate"],
                "up": row["up"],
                "down": row["down"],
                "left": row["left"],
                "right": row["right"]
            })

        # Childs rekonstruieren
        for _, row in self.q_table_child.iterrows():
            parent_key = row["Pstate"]
            child_key_expected = row["Cstate"]
            operations_str = row["Operations"]

            if parent_key not in parent_lookup:
                continue

            parent_state = self.key_to_state_array(parent_key)

            parent_actions = parent_lookup[parent_key]

            if isinstance(operations_str, str) and operations_str.strip():
                operations = [op.strip() for op in operations_str.split(",") if op.strip()]
            else:
                operations = []

            child_state = self.operator.apply_parent_to_child_state(parent_state, operations)
            child_key_computed = self.state_to_key(child_state)

            child_actions = self.operator.apply_parent_to_child_actions(parent_actions, operations)

            if child_key_computed != child_key_expected:
                print("Reconstruction mismatch:")
                print("Parent:", parent_key)
                print("Stored child:", child_key_expected)
                print("Computed child:", child_key_computed)
                print("Operations:", operations)
                print("-" * 50)

            reconstructed_rows.append({
                "state": child_key_computed,
                "up": child_actions["up"],
                "down": child_actions["down"],
                "left": child_actions["left"],
                "right": child_actions["right"]
            })

        reconstructed_df = pd.DataFrame(
            reconstructed_rows,
            columns=["state", "up", "down", "left", "right"]
        )

        if not reconstructed_df.empty:
            reconstructed_df = reconstructed_df.drop_duplicates(
                subset=["state"],
                keep="first"
            ).reset_index(drop=True)

        self.q_table_reconstructed = reconstructed_df
        return self.q_table_reconstructed
    
def run_training_normalizer(env = Environment(size=2), filename="q_table", filepath="./Data/", episodes=1000, grid_size=2, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, learning_rate=0.1, max_depth=5, save_interval=10):
    filepath = filepath
    filename = filename
    
    grid_size = grid_size
    epsilon = epsilon
    epsilon_decay = epsilon_decay
    epsilon_min = epsilon_min
    learning_rate = learning_rate
    max_depth = max_depth
    
    env = Environment(size=grid_size)
    agent = Agent(environment=env, filepath=filepath, filename=filename, max_depth=max_depth, epsilon=epsilon, epsilon_decay=epsilon_decay, epsilon_min=epsilon_min, learning_rate=learning_rate, size=grid_size)

    
    agent.load_q_table_single(filepath=filepath, filename=filename)

    episodes = episodes

    for episode in range(episodes):
        state = env.reset()
        done = False

        while not done:
            actions = env.find_available_actions()
            if not actions:
                break

            action = agent.choose_action(state, actions)
            next_state, reward, done, info = env.step(action)
            next_actions = env.find_available_actions()

            agent.learn(state, action, reward, next_state, next_actions)
            state = next_state

        agent.decay_epsilon()

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1} completed.")

        save_interval = save_interval
        if (episode + 1) % save_interval == 0:
            agent.divide_q_table()
            agent.save_q_table_parent_child()
            agent.save_q_table_single()
            agent.save_q_table_reconstructed()


    agent.divide_q_table()
    agent.save_q_table_parent_child()
    agent.save_q_table_single()
    agent.save_q_table_reconstructed()

def main():
    filepath = "./Data/"
    filename = "q_table"

    grid_size = 2
    env = Environment(size=grid_size)
    agent = Agent(environment=env, filepath=filepath, filename=filename, max_depth=5, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, learning_rate=0.1, size=grid_size)

    agent.load_q_table_single(filepath=filepath, filename=filename)

    episodes = 1000

    for episode in range(episodes):
        state = env.reset()
        done = False

        while not done:
            actions = env.find_available_actions()
            if not actions:
                break

            action = agent.choose_action(state, actions)
            next_state, reward, done, info = env.step(action)
            next_actions = env.find_available_actions()

            agent.learn(state, action, reward, next_state, next_actions)
            state = next_state

        agent.decay_epsilon()

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1} completed.")
            agent.divide_q_table()
            agent.save_q_table_parent_child()
            agent.save_q_table_single()

    agent.divide_q_table()
    agent.save_q_table_parent_child()
    agent.save_q_table_single()

if __name__ == "__main__":
    print("Recommendation: Run the 'main.py' file to execute the program.")
    main()