import os
import numpy as np
import pandas as pd

from Bachelor.MatrixOperation import Detector, Operation
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

        # komprimierte Tabellen fürs separate Speichern
        self.q_table_parent = pd.DataFrame(columns=["Pstate", "up", "down", "left", "right"])
        self.q_table_child = pd.DataFrame(columns=["Pstate", "Cstate", "Operations"])

        self.filepath = filepath
        self.filename = filename

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate

        self.detector = Detector()
        self.operator = Operation()

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

    def detect_operations(self, parent_state, child_state, max_depth=5):
        return self.detector.detect_BFS(child_state, parent_state, max_depth=max_depth)

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

    def compress_q_table(self):
        parent_rows = []
        child_rows = []

        states = self.q_table["state"].tolist()
        used_as_child = set()

        for i, candidate_parent_key in enumerate(states):
            if candidate_parent_key in used_as_child:
                continue

            candidate_parent_row = self.q_table[self.q_table["state"] == candidate_parent_key].iloc[0]
            candidate_parent_state = self.key_to_state_array(candidate_parent_key)

            parent_rows.append({
                "Pstate": candidate_parent_key,
                "up": candidate_parent_row["up"],
                "down": candidate_parent_row["down"],
                "left": candidate_parent_row["left"],
                "right": candidate_parent_row["right"]
            })

            for j, candidate_child_key in enumerate(states):
                if i == j:
                    continue
                if candidate_child_key in used_as_child:
                    continue

                candidate_child_state = self.key_to_state_array(candidate_child_key)

                operations = self.detect_operations(
                    parent_state=candidate_parent_state,
                    child_state=candidate_child_state,
                    max_depth=self.max_depth
                )

                if operations is not None:
                    used_as_child.add(candidate_child_key)

                    child_rows.append({
                        "Pstate": candidate_parent_key,
                        "Cstate": candidate_child_key,
                        "Operations": ",".join(operations)
                    })

        self.q_table_parent = pd.DataFrame(parent_rows)
        self.q_table_child = pd.DataFrame(child_rows)

    def save_q_table_single(self, filepath=None, filename=None):
        filepath = filepath or self.filepath
        filename = filename or self.filename
        os.makedirs(filepath, exist_ok=True)
        self.q_table.to_csv(os.path.join(filepath, filename + "_single_" + str(self.size) + ".csv"), index=False)

    def save_q_table_parent(self, filepath=None, filename=None):
        filepath = filepath or self.filepath
        filename = filename or self.filename
        os.makedirs(filepath, exist_ok=True)
        self.q_table_parent.to_csv(os.path.join(filepath, filename + "_parent_" + str(self.size) + ".csv"), index=False)

    def save_q_table_child(self, filepath=None, filename=None):
        filepath = filepath or self.filepath
        filename = filename or self.filename
        os.makedirs(filepath, exist_ok=True)
        self.q_table_child.to_csv(os.path.join(filepath, filename + "_child_" + str(self.size) + ".csv"), index=False)

    def compress_and_save_tables(self):
        self.compress_q_table()
        self.save_q_table_parent()
        self.save_q_table_child()

    def load_q_table_single(self, filepath=None, filename=None):
        filepath = filepath or self.filepath
        filename = filename or self.filename
        full_path = os.path.join(filepath, filename + "_single_" + str(self.size) + ".csv")
        if os.path.exists(full_path):
            self.q_table = pd.read_csv(full_path)
            
    def load_q_table_parent(self, filepath=None, filename=None):
        filepath = filepath or self.filepath
        filename = filename or self.filename
        full_path = os.path.join(filepath, filename + "_parent_" + str(self.size) + ".csv")
        if os.path.exists(full_path):
            self.q_table_parent = pd.read_csv(full_path)
            
    def load_q_table_child(self, filepath=None, filename=None):
        filepath = filepath or self.filepath
        filename = filename or self.filename
        full_path = os.path.join(filepath, filename + "_child_" + str(self.size) + ".csv")
        if os.path.exists(full_path):
            self.q_table_child = pd.read_csv(full_path)
            
    def load_q_table_merge(self, filepath=None, filename=None):
        self.load_q_table_parent(filepath, filename)
        self.load_q_table_child(filepath, filename)
        return self.reconstruct_q_table_from_parent_child()
        
    def load_q_table_merge(self, filepath=None, filename=None):
        self.load_q_table_parent(filepath, filename)
        self.load_q_table_child(filepath, filename)

        if self.q_table_parent.empty and self.q_table_child.empty:
            self.q_table = pd.DataFrame(columns=["state", "up", "down", "left", "right"])
            return self.q_table

        return self.reconstruct_q_table_from_parent_child()   
     
    def save_q_table_reconstructed(self, filepath=None, filename=None):
        reconstructed_q_table = self.reconstruct_q_table_from_parent_child()
        filepath = filepath or self.filepath
        filename = filename or self.filename
        os.makedirs(filepath, exist_ok=True)
        reconstructed_q_table.to_csv(os.path.join(filepath, filename + "_reconstructed_" + str(self.size) + ".csv"), index=False)
        
    def reconstruct_q_table_from_parent_child(self):
        
        reconstructed_rows = []

        # 1. Alle Parent-States direkt übernehmen
        for _, row in self.q_table_parent.iterrows():
            reconstructed_rows.append({
                "state": row["Pstate"],
                "up": row["up"],
                "down": row["down"],
                "left": row["left"],
                "right": row["right"]
            })

        # 2. Alle Child-States aus Parent + Operationen rekonstruieren
        for _, row in self.q_table_child.iterrows():
            parent_key = row["Pstate"]
            operations_str = row["Operations"]

            parent_match = self.q_table_parent[self.q_table_parent["Pstate"] == parent_key]
            if parent_match.empty:
                continue

            parent_row = parent_match.iloc[0]
            parent_state = self.key_to_state_array(parent_key)

            if isinstance(operations_str, str):
                operation_list = [op_code.strip() for op_code in operations_str.split(",") if op_code.strip()]
            else:
                operation_list = ["n"]

            child_state = self.operator.apply_operations(parent_state, operation_list)
            child_key = self.state_to_key(child_state)

            parent_q_values = {
                "up": parent_row["up"],
                "down": parent_row["down"],
                "left": parent_row["left"],
                "right": parent_row["right"],
            }

            child_q_values = self.operator.apply_action_operations(parent_q_values, operation_list)

            reconstructed_rows.append({
                "state": child_key,
                "up": child_q_values["up"],
                "down": child_q_values["down"],
                "left": child_q_values["left"],
                "right": child_q_values["right"]
            })

        # Immer mit festen Spalten erzeugen
        reconstructed_df = pd.DataFrame(
            reconstructed_rows,
            columns=["state", "up", "down", "left", "right"]
        )

        if not reconstructed_df.empty:
            reconstructed_df = reconstructed_df.drop_duplicates(
                subset=["state"],
                keep="first"
            ).reset_index(drop=True)

        return reconstructed_df 

def run_training(env = Environment(size=2), filename="q_table", filepath="./Data/", episodes=1000, grid_size=2, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, learning_rate=0.1, max_depth=5, save_interval=10):
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
            agent.compress_and_save_tables()
            agent.save_q_table_reconstructed(filepath=filepath, filename=filename)
            agent.save_q_table_single()

    agent.compress_and_save_tables()
    agent.save_q_table_reconstructed(filepath=filepath, filename=filename)
    agent.save_q_table_single()

def main():
    filepath = "./Data/"
    filename = "q_table"

    grid_size = 2
    env = Environment(size=grid_size)
    agent = Agent(environment=env, filepath=filepath, filename=filename, max_depth=5, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, learning_rate=0.1, size=grid_size)

    agent.load_q_table_merge(filepath=filepath, filename=filename)
    agent.save_q_table_reconstructed(filepath=filepath, filename=filename)

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
            agent.compress_and_save_tables()
            agent.save_q_table_single()

    # am Ende sicherheitshalber nochmal speichern
    agent.compress_and_save_tables()
    agent.save_q_table_single()

if __name__ == "__main__":
    print("Recommendation: Run the 'main.py' file to execute the program.")
    main()