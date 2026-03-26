import os
import numpy as np
import pandas as pd

from Bachelor.PreCalculator import PreCalculator
from Environment.Environment import Environment


class Agent:
    ACTION_COLUMNS = ["up", "down", "left", "right"]

    def __init__(
        self,
        environment,
        filepath="./",
        filename="q-table",
        max_depth=10,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        learning_rate=0.1,
        size=2,
        use_normal_q_table=True,
        use_compressed_q_table=True
    ):
        self.environment = environment

        self.filepath = filepath
        self.filename = filename

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.size = size

        self.use_normal_q_table = use_normal_q_table
        self.use_compressed_q_table = use_compressed_q_table

        self.calc = PreCalculator()

        # normale Full-Q-Table
        self.q_table_normal = pd.DataFrame(columns=["state", "up", "down", "left", "right"])

        # komprimierte Tabellen
        self.q_table_parent = pd.DataFrame(columns=["Pstate", "up", "down", "left", "right"])
        self.q_table_child = pd.DataFrame(columns=["Pstate", "Cstate", "Operations"])

        self.q_table_reconstructed = pd.DataFrame(columns=["state", "up", "down", "left", "right"])

        self.precalculated_map = {}

    def normal_state_exists(self, state_key):
        return state_key in self.q_table_normal["state"].values

    def ensure_normal_state_exists(self, state):
        state_key = self.state_to_key(state)

        if self.normal_state_exists(state_key):
            return

        new_entry = {
            "state": state_key,
            "up": 0.0,
            "down": 0.0,
            "left": 0.0,
            "right": 0.0
        }

        self.q_table_normal = pd.concat(
            [self.q_table_normal, pd.DataFrame([new_entry])],
            ignore_index=True
        )

    def get_normal_actions_for_state(self, state_key):
        match = self.q_table_normal[self.q_table_normal["state"] == state_key]
        if match.empty:
            return self.empty_actions()

        row = match.iloc[0]
        return {
            "up": float(row["up"]),
            "down": float(row["down"]),
            "left": float(row["left"]),
            "right": float(row["right"])
        }

    def state_to_key(self, state):
        state_array = np.array(state)
        return ",".join(map(str, state_array.flatten()))

    def key_to_state_array(self, state_key):
        values = list(map(int, state_key.split(",")))
        return np.array(values).reshape(self.size, self.size)

    def empty_actions(self):
        return {
            "up": 0.0,
            "down": 0.0,
            "left": 0.0,
            "right": 0.0
        }

    def row_to_actions(self, row, prefix=""):
        return {
            "up": float(row[f"{prefix}up"]),
            "down": float(row[f"{prefix}down"]),
            "left": float(row[f"{prefix}left"]),
            "right": float(row[f"{prefix}right"]),
        }

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

    # --------------------------------------------------
    # Parent / Child / Precalculated Handling
    # --------------------------------------------------

    def apply_operations_to_state(self, state, operations):
        current_state = state.copy()

        for op in operations:
            if op not in self.calc.operations:
                continue

            state_func, _ = self.calc.operations[op]
            current_state = state_func(current_state)

        return current_state

    def apply_operations_to_actions(self, actions, operations):
        current_actions = actions.copy()

        for op in operations:
            if op not in self.calc.operations:
                continue

            _, action_func = self.calc.operations[op]
            current_actions = action_func(current_actions)

        return current_actions

    def parent_exists(self, state_key):
        return state_key in self.q_table_parent["Pstate"].values

    def child_exists(self, state_key):
        return state_key in self.q_table_child["Cstate"].values

    def get_parent_row(self, parent_key):
        match = self.q_table_parent[self.q_table_parent["Pstate"] == parent_key]
        if match.empty:
            return None
        return match.iloc[0]

    def get_oriented_actions_for_state(self, state_key):
        """
        Gibt die Actions für einen State zurück.
        - Wenn Parent: direkt aus Parent-Tabelle
        - Wenn Child: aus Parent holen und via gespeicherten Operationen transformieren
        """
        if self.parent_exists(state_key):
            row = self.get_parent_row(state_key)
            return self.row_to_actions(row)

        if state_key in self.precalculated_map:
            parent_key = self.precalculated_map[state_key]["Pstate"]
            operations = self.precalculated_map[state_key]["Operations"]

            parent_row = self.get_parent_row(parent_key)
            if parent_row is None:
                return self.empty_actions()

            parent_actions = self.row_to_actions(parent_row)
            return self.apply_operations_to_actions(parent_actions, operations)

        return self.empty_actions()

    def apply_operations_to_actions(self, actions, operations):
        """
        operations ist z.B. 'rdr'
        Wendet nur die Action-Transformationen in derselben Reihenfolge an.
        """
        current_actions = actions.copy()

        for op in operations:
            if op not in self.calc.operations:
                continue

            _, action_func = self.calc.operations[op]
            current_actions = action_func(current_actions)

        return current_actions

    def ensure_state_exists(self, state):
        state_key = self.state_to_key(state)

        # Bereits Parent
        if self.parent_exists(state_key):
            return

        # Bereits als möglicher Child bekannt
        if state_key in self.precalculated_map:
            info = self.precalculated_map[state_key]

            # Erst JETZT persistent als benutzter Child speichern
            if not self.child_exists(state_key):
                new_child = {
                    "Pstate": info["Pstate"],
                    "Cstate": state_key,
                    "Operations": info["Operations"]
                }

                self.q_table_child = pd.concat(
                    [self.q_table_child, pd.DataFrame([new_child])],
                    ignore_index=True
                )
            return

        # Sonst neuer Parent
        new_parent = {
            "Pstate": state_key,
            "up": 0.0,
            "down": 0.0,
            "left": 0.0,
            "right": 0.0
        }

        self.q_table_parent = pd.concat(
            [self.q_table_parent, pd.DataFrame([new_parent])],
            ignore_index=True
        )

        self.precalculate_from_parent(state)

    def precalculate_from_parent(self, parent_state):
        parent_key = self.state_to_key(parent_state)

        parent_row = self.get_parent_row(parent_key)
        if parent_row is None:
            return

        parent_actions = self.row_to_actions(parent_row)

        results = self.calc.dig_deeper(
            [parent_state.copy(), parent_actions.copy()],
            max_depth=self.max_depth
        )

        for item in results:
            child_state = item["state"]
            child_key = self.state_to_key(child_state)
            operations = item["path"]

            # Root selbst nicht als Child merken
            if child_key == parent_key:
                continue

            # echte Parents nie überschreiben
            if self.parent_exists(child_key):
                continue

            # nur im Cache merken, NICHT persistent speichern
            if child_key not in self.precalculated_map:
                self.precalculated_map[child_key] = {
                    "Pstate": parent_key,
                    "Operations": operations
                }

    # --------------------------------------------------
    # RL
    # --------------------------------------------------

    def choose_action(self, state, available_actions):
        if np.random.rand() < self.epsilon:
            return np.random.choice(available_actions)

        state_key = self.state_to_key(state)

        if self.policy_source == "normal":
            q_values_dict = self.get_normal_actions_for_state(state_key)
        else:
            q_values_dict = self.get_oriented_actions_for_state(state_key)

        if not q_values_dict:
            return np.random.choice(available_actions)

        q_values_filtered = {a: q_values_dict[a] for a in available_actions}
        max_q_value = max(q_values_filtered.values())
        best_actions = [a for a, v in q_values_filtered.items() if v == max_q_value]

        return np.random.choice(best_actions)

    def learn_compressed(self, state, action, reward, next_state, available_actions):
        state_key = self.state_to_key(state)
        next_state_key = self.state_to_key(next_state)

        self.ensure_state_exists(state)
        self.ensure_state_exists(next_state)

        if state_key in self.precalculated_map:
            parent_key = self.precalculated_map[state_key]["Pstate"]
            operations = self.precalculated_map[state_key]["Operations"]
            parent_action = self.reverse_action_by_operations(action, operations)
        else:
            parent_key = state_key
            parent_action = action

        current_parent_row = self.q_table_parent[self.q_table_parent["Pstate"] == parent_key]
        if current_parent_row.empty:
            return

        current_q_value = current_parent_row.iloc[0][parent_action]

        if available_actions:
            next_q_values = self.get_oriented_actions_for_state(next_state_key)
            next_available_qs = [next_q_values[a] for a in available_actions if a in next_q_values]
            next_max_q_value = max(next_available_qs) if next_available_qs else 0.0
        else:
            next_max_q_value = 0.0

        new_q_value = (
            (1 - self.learning_rate) * current_q_value
            + self.learning_rate * (reward + next_max_q_value)
        )

        self.q_table_parent.loc[
            self.q_table_parent["Pstate"] == parent_key, parent_action
        ] = new_q_value

    def learn(self, state, action, reward, next_state, available_actions):
        if self.use_normal_q_table:
            self.learn_normal(state, action, reward, next_state, available_actions)

        if self.use_compressed_q_table:
            self.learn_compressed(state, action, reward, next_state, available_actions)

    def learn_normal(self, state, action, reward, next_state, available_actions):
        state_key = self.state_to_key(state)
        next_state_key = self.state_to_key(next_state)

        self.ensure_normal_state_exists(state)
        self.ensure_normal_state_exists(next_state)

        current_rows = self.q_table_normal.loc[self.q_table_normal["state"] == state_key, action]
        if current_rows.empty:
            return

        current_q_value = current_rows.values[0]

        if available_actions:
            next_rows = self.q_table_normal.loc[
                self.q_table_normal["state"] == next_state_key,
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

        self.q_table_normal.loc[
            self.q_table_normal["state"] == state_key, action
        ] = new_q_value

    def learn_compressed(self, state, action, reward, next_state, available_actions):
        state_key = self.state_to_key(state)
        next_state_key = self.state_to_key(next_state)

        self.ensure_state_exists(state)
        self.ensure_state_exists(next_state)

        if state_key in self.precalculated_map:
            parent_key = self.precalculated_map[state_key]["Pstate"]
            operations = self.precalculated_map[state_key]["Operations"]
            parent_action = self.reverse_action_by_operations(action, operations)
        else:
            parent_key = state_key
            parent_action = action

        current_parent_row = self.q_table_parent[self.q_table_parent["Pstate"] == parent_key]
        if current_parent_row.empty:
            return

        current_q_value = current_parent_row.iloc[0][parent_action]

        if available_actions:
            next_q_values = self.get_oriented_actions_for_state(next_state_key)
            next_available_qs = [next_q_values[a] for a in available_actions if a in next_q_values]
            next_max_q_value = max(next_available_qs) if next_available_qs else 0.0
        else:
            next_max_q_value = 0.0

        new_q_value = (
            (1 - self.learning_rate) * current_q_value
            + self.learning_rate * (reward + next_max_q_value)
        )

        self.q_table_parent.loc[
            self.q_table_parent["Pstate"] == parent_key, parent_action
        ] = new_q_value
    
    def reverse_action_by_operations(self, action, operations):
        """
        Child-action -> Parent-action

        Wir wenden die inverse Richtungsabbildung in umgekehrter Reihenfolge an.
        Aktuell für:
        - r  (Rotation)
        - d  (Double, ändert Action nicht)
        """
        current_action = action

        for op in reversed(operations):
            if op == "r":
                current_action = self.inverse_single_rotate_action(current_action)
            elif op == "d":
                current_action = current_action

        return current_action

    def inverse_single_rotate_action(self, action):
        """
        Vorwärts war:
            new_up    = old_right
            new_down  = old_left
            new_left  = old_up
            new_right = old_down

        Invers:
            old_up    <- new_left
            old_down  <- new_right
            old_left  <- new_down
            old_right <- new_up
        """
        mapping = {
            "up": "left",
            "down": "right",
            "left": "down",
            "right": "up"
        }
        return mapping[action]

    # --------------------------------------------------
    # Save / Load
    # --------------------------------------------------

    def save_q_table_normal(self, filepath=None, filename=None):
        if filepath is None:
            filepath = self.filepath
        if filename is None:
            filename = self.filename

        os.makedirs(filepath, exist_ok=True)

        normal_path = os.path.join(filepath, f"{filename}_normal_{self.size}.csv")
        self.q_table_normal.to_csv(normal_path, index=False)

    def load_q_table_normal(self, filepath=None, filename=None):
        if filepath is None:
            filepath = self.filepath
        if filename is None:
            filename = self.filename

        normal_path = os.path.join(filepath, f"{filename}_normal_{self.size}.csv")

        if os.path.exists(normal_path):
            self.q_table_normal = pd.read_csv(normal_path)
        else:
            self.q_table_normal = pd.DataFrame(columns=["state", "up", "down", "left", "right"])

    def save_q_table_parent_child(self, filepath=None, filename=None):
        if filepath is None:
            filepath = self.filepath
        if filename is None:
            filename = self.filename

        os.makedirs(filepath, exist_ok=True)

        parent_path = os.path.join(filepath, f"{filename}_parent_{self.size}.csv")
        child_path = os.path.join(filepath, f"{filename}_child_{self.size}.csv")

        self.q_table_parent.to_csv(parent_path, index=False)
        self.q_table_child.to_csv(child_path, index=False)

    def load_q_table_parent_child(self, filepath=None, filename=None):
        if filepath is None:
            filepath = self.filepath
        if filename is None:
            filename = self.filename

        parent_path = os.path.join(filepath, f"{filename}_parent_{self.size}.csv")
        child_path = os.path.join(filepath, f"{filename}_child_{self.size}.csv")

        if os.path.exists(parent_path):
            self.q_table_parent = pd.read_csv(parent_path)
        else:
            self.q_table_parent = pd.DataFrame(columns=["Pstate", "up", "down", "left", "right"])

        if os.path.exists(child_path):
            self.q_table_child = pd.read_csv(child_path)
        else:
            self.q_table_child = pd.DataFrame(columns=["Pstate", "Cstate", "Operations"])

        self.rebuild_precalculated_map_from_child_table()

    def rebuild_precalculated_map_from_child_table(self):
        self.precalculated_map = {}

        if self.q_table_child.empty:
            return

        for _, row in self.q_table_child.iterrows():
            cstate = row["Cstate"]
            self.precalculated_map[cstate] = {
                "Pstate": row["Pstate"],
                "Operations": row["Operations"] if isinstance(row["Operations"], str) else ""
            }

    # --------------------------------------------------
    # Optional erstmal deaktiviert
    # --------------------------------------------------

    def reconstruct_q_table(self):
        reconstructed_rows = []

        if self.q_table_parent.empty:
            self.q_table_reconstructed = pd.DataFrame(
                columns=["state", "up", "down", "left", "right"]
            )
            return self.q_table_reconstructed

        # 1. Parents direkt übernehmen
        for _, row in self.q_table_parent.iterrows():
            reconstructed_rows.append({
                "state": row["Pstate"],
                "up": row["up"],
                "down": row["down"],
                "left": row["left"],
                "right": row["right"]
            })

        # 2. Childs aus Parent + Operations erzeugen
        for _, row in self.q_table_child.iterrows():
            parent_key = row["Pstate"]
            child_key_stored = row["Cstate"]
            operations = row["Operations"] if isinstance(row["Operations"], str) else ""

            parent_match = self.q_table_parent[self.q_table_parent["Pstate"] == parent_key]
            if parent_match.empty:
                continue

            parent_row = parent_match.iloc[0]
            parent_state = self.key_to_state_array(parent_key)

            parent_actions = {
                "up": float(parent_row["up"]),
                "down": float(parent_row["down"]),
                "left": float(parent_row["left"]),
                "right": float(parent_row["right"])
            }

            child_state = self.apply_operations_to_state(parent_state, operations)
            child_actions = self.apply_operations_to_actions(parent_actions, operations)

            child_key_computed = self.state_to_key(child_state)

            # Optional Debug
            if child_key_computed != child_key_stored:
                print("Reconstruction mismatch")
                print("Parent:", parent_key)
                print("Stored child:", child_key_stored)
                print("Computed child:", child_key_computed)
                print("Operations:", operations)
                print("-" * 40)

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

    # def reconstruct_q_table(self):
    #     pass

    def save_q_table_reconstructed(self, filepath=None, filename=None):
        if filepath is None:
            filepath = self.filepath
        if filename is None:
            filename = self.filename

        os.makedirs(filepath, exist_ok=True)

        reconstructed_path = os.path.join(filepath, f"{filename}_reconstructed_{self.size}.csv")
        self.q_table_reconstructed.to_csv(reconstructed_path, index=False)
    
def run_training_reworked2(
    env=Environment(size=2),
    filename="q_table",
    filepath="./Data/",
    episodes=1000,
    grid_size=2,
    epsilon=1.0,
    epsilon_decay=0.995,
    epsilon_min=0.01,
    learning_rate=0.1,
    max_depth=10,
    save_interval=10,
    use_normal_q_table=True,
    use_compressed_q_table=True,
    policy_source="normal"
):
    env = Environment(size=grid_size)

    agent = Agent(
        environment=env,
        filepath=filepath,
        filename=filename,
        max_depth=max_depth,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        learning_rate=learning_rate,
        size=grid_size,
        use_normal_q_table=use_normal_q_table,
        use_compressed_q_table=use_compressed_q_table
    )

    agent.policy_source = policy_source

    if use_normal_q_table:
        agent.load_q_table_normal(filepath=filepath, filename=filename)

    if use_compressed_q_table:
        agent.load_q_table_parent_child(filepath=filepath, filename=filename)

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

        if (episode + 1) % save_interval == 0:
            if use_normal_q_table:
                agent.save_q_table_normal(filepath=filepath, filename=filename)

            if use_compressed_q_table:
                agent.save_q_table_parent_child(filepath=filepath, filename=filename)
                agent.reconstruct_q_table()
                agent.save_q_table_reconstructed(filepath=filepath, filename=filename)

    if use_normal_q_table:
        agent.save_q_table_normal(filepath=filepath, filename=filename)

    if use_compressed_q_table:
        agent.save_q_table_parent_child(filepath=filepath, filename=filename)
        agent.reconstruct_q_table()
        agent.save_q_table_reconstructed(filepath=filepath, filename=filename)