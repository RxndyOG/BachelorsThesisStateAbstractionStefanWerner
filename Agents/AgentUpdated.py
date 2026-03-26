import os
import numpy as np
import pandas as pd

from Bachelor.MatrixOperationUpdated import Detector, Operation
from Environment.Environment import Environment


class Agent:
    ACTIONS = ["up", "down", "left", "right"]

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
        size=2
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

        # -----------------------------
        # Schnelle interne Datenstrukturen
        # -----------------------------
        # volle Q-Table:
        # q_table_dict[state_key] = {"up":..., "down":..., "left":..., "right":...}
        self.q_table_dict = {}

        # exakte Einfügereihenfolge wie früher im DataFrame
        self.state_order = []

        # komprimierte Tabellen
        self.parent_rows = []   # [{"Pstate": ..., "up":..., ...}, ...]
        self.child_rows = []    # [{"Pstate": ..., "Operations": ...}, ...]

        # Reuse statt immer neu
        self.detector = Detector(max_depth=max_depth)
        self.operation = Operation()

    # -------------------------------------------------
    # State / Key Conversion
    # -------------------------------------------------
    def state_to_key(self, state):
        state_array = np.array(state, dtype=int)
        return ",".join(map(str, state_array.flatten()))

    def key_to_state_array(self, state_key):
        values = list(map(int, state_key.split(",")))
        return np.array(values, dtype=int).reshape(self.size, self.size)

    # -------------------------------------------------
    # interne Hilfsfunktionen
    # -------------------------------------------------
    def _empty_q_values(self):
        return {
            "up": 0.0,
            "down": 0.0,
            "left": 0.0,
            "right": 0.0
        }

    def _has_state(self, state_key):
        return state_key in self.q_table_dict

    def _get_q_values(self, state_key):
        return self.q_table_dict.get(state_key)

    def _set_q_value(self, state_key, action, value):
        self.q_table_dict[state_key][action] = value

    def _get_row_like(self, state_key):
        """
        Liefert ein dict ähnlich einer DataFrame-Zeile.
        """
        q = self.q_table_dict[state_key]
        return {
            "state": state_key,
            "up": q["up"],
            "down": q["down"],
            "left": q["left"],
            "right": q["right"]
        }

    # -------------------------------------------------
    # RL
    # -------------------------------------------------
    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

    def choose_action(self, state, available_actions):
        if np.random.rand() < self.epsilon:
            return np.random.choice(available_actions)

        state_key = self.state_to_key(state)
        q_values = self._get_q_values(state_key)

        if q_values is None:
            return np.random.choice(available_actions)

        restricted_q = {a: q_values[a] for a in available_actions}
        max_q_value = max(restricted_q.values())
        best_actions = [a for a, v in restricted_q.items() if v == max_q_value]

        return np.random.choice(best_actions)

    def detect_operations(self, parent_state, child_state, max_depth=5):
        # gleiche Logik wie vorher: detect_BFS(child, parent, depth)
        return self.detector.detect_BFS(child_state, parent_state, max_depth=max_depth)

    def create_q_table_entry(self, state):
        state_key = self.state_to_key(state)

        if self._has_state(state_key):
            return

        self.q_table_dict[state_key] = self._empty_q_values()
        self.state_order.append(state_key)

    def learn(self, state, action, reward, next_state, available_actions):
        state_key = self.state_to_key(state)
        next_state_key = self.state_to_key(next_state)

        if not self._has_state(state_key):
            self.create_q_table_entry(state)

        if not self._has_state(next_state_key):
            self.create_q_table_entry(next_state)

        current_q_value = self.q_table_dict[state_key][action]

        if available_actions:
            next_q_values = self.q_table_dict[next_state_key]
            next_max_q_value = max(next_q_values[a] for a in available_actions)
        else:
            next_max_q_value = 0.0

        new_q_value = (
            (1 - self.learning_rate) * current_q_value
            + self.learning_rate * (reward + next_max_q_value)
        )

        self._set_q_value(state_key, action, new_q_value)

    # -------------------------------------------------
    # Compression
    # -------------------------------------------------
    def compress_q_table(self):
        """
        Gleiche Logik wie dein alter Code.
        Nichts an der Reihenfolge oder Parent/Child-Entscheidung verändert.
        """
        parent_rows = []
        child_rows = []

        states = self.state_order[:]   # gleiche Reihenfolge wie früher
        used_as_child = set()

        for i, candidate_parent_key in enumerate(states):
            if candidate_parent_key in used_as_child:
                continue

            candidate_parent_row = self._get_row_like(candidate_parent_key)
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
                        "Operations": ",".join(operations)
                    })

        self.parent_rows = parent_rows
        self.child_rows = child_rows

    # -------------------------------------------------
    # DataFrame Export
    # -------------------------------------------------
    def q_table_to_dataframe(self):
        rows = []
        for state_key in self.state_order:
            q = self.q_table_dict[state_key]
            rows.append({
                "state": state_key,
                "up": q["up"],
                "down": q["down"],
                "left": q["left"],
                "right": q["right"]
            })
        return pd.DataFrame(rows, columns=["state", "up", "down", "left", "right"])

    def parent_table_to_dataframe(self):
        return pd.DataFrame(
            self.parent_rows,
            columns=["Pstate", "up", "down", "left", "right"]
        )

    def child_table_to_dataframe(self):
        return pd.DataFrame(
            self.child_rows,
            columns=["Pstate", "Operations"]
        )

    # -------------------------------------------------
    # Save
    # -------------------------------------------------
    def save_q_table_single(self, filepath=None, filename=None):
        filepath = filepath or self.filepath
        filename = filename or self.filename
        os.makedirs(filepath, exist_ok=True)

        df = self.q_table_to_dataframe()
        df.to_csv(
            os.path.join(filepath, filename + "_single_" + str(self.size) + ".csv"),
            index=False
        )

    def save_q_table_parent(self, filepath=None, filename=None):
        filepath = filepath or self.filepath
        filename = filename or self.filename
        os.makedirs(filepath, exist_ok=True)

        df = self.parent_table_to_dataframe()
        df.to_csv(
            os.path.join(filepath, filename + "_parent_" + str(self.size) + ".csv"),
            index=False
        )

    def save_q_table_child(self, filepath=None, filename=None):
        filepath = filepath or self.filepath
        filename = filename or self.filename
        os.makedirs(filepath, exist_ok=True)

        df = self.child_table_to_dataframe()
        df.to_csv(
            os.path.join(filepath, filename + "_child_" + str(self.size) + ".csv"),
            index=False
        )

    def compress_and_save_tables(self):
        self.compress_q_table()
        self.save_q_table_parent()
        self.save_q_table_child()

    # -------------------------------------------------
    # Load
    # -------------------------------------------------
    def load_q_table_single(self, filepath=None, filename=None):
        filepath = filepath or self.filepath
        filename = filename or self.filename
        full_path = os.path.join(filepath, filename + "_single_" + str(self.size) + ".csv")

        if not os.path.exists(full_path):
            self.q_table_dict = {}
            self.state_order = []
            return

        df = pd.read_csv(full_path)

        self.q_table_dict = {}
        self.state_order = []

        for _, row in df.iterrows():
            state_key = str(row["state"])
            self.q_table_dict[state_key] = {
                "up": float(row["up"]),
                "down": float(row["down"]),
                "left": float(row["left"]),
                "right": float(row["right"])
            }
            self.state_order.append(state_key)

    def load_q_table_parent(self, filepath=None, filename=None):
        filepath = filepath or self.filepath
        filename = filename or self.filename
        full_path = os.path.join(filepath, filename + "_parent_" + str(self.size) + ".csv")

        if os.path.exists(full_path):
            df = pd.read_csv(full_path)
            self.parent_rows = df.to_dict(orient="records")
        else:
            self.parent_rows = []

    def load_q_table_child(self, filepath=None, filename=None):
        filepath = filepath or self.filepath
        filename = filename or self.filename
        full_path = os.path.join(filepath, filename + "_child_" + str(self.size) + ".csv")

        if os.path.exists(full_path):
            df = pd.read_csv(full_path)
            self.child_rows = df.to_dict(orient="records")
        else:
            self.child_rows = []

    def load_q_table_merge(self, filepath=None, filename=None):
        self.load_q_table_parent(filepath, filename)
        self.load_q_table_child(filepath, filename)

        if len(self.parent_rows) == 0 and len(self.child_rows) == 0:
            self.q_table_dict = {}
            self.state_order = []
            return self.q_table_to_dataframe()

        return self.reconstruct_q_table_from_parent_child()

    # -------------------------------------------------
    # Reconstruction
    # -------------------------------------------------
    def reconstruct_q_table_from_parent_child(self):
        reconstructed_rows = []

        # 1. Alle Parent-States direkt übernehmen
        for row in self.parent_rows:
            reconstructed_rows.append({
                "state": row["Pstate"],
                "up": float(row["up"]),
                "down": float(row["down"]),
                "left": float(row["left"]),
                "right": float(row["right"])
            })

        # 2. Alle Child-States aus Parent + Operationen rekonstruieren
        for row in self.child_rows:
            parent_key = row["Pstate"]
            operations_str = row["Operations"]

            parent_match = None
            for parent_row in self.parent_rows:
                if parent_row["Pstate"] == parent_key:
                    parent_match = parent_row
                    break

            if parent_match is None:
                continue

            parent_state = self.key_to_state_array(parent_key)

            if isinstance(operations_str, str):
                operation_list = [op_code.strip() for op_code in operations_str.split(",") if op_code.strip()]
            else:
                operation_list = ["n"]

            child_state = self.operation.apply_operations(parent_state, operation_list)
            child_key = self.state_to_key(child_state)

            parent_q_values = {
                "up": float(parent_match["up"]),
                "down": float(parent_match["down"]),
                "left": float(parent_match["left"]),
                "right": float(parent_match["right"]),
            }

            child_q_values = self.operation.apply_action_operations(parent_q_values, operation_list)

            reconstructed_rows.append({
                "state": child_key,
                "up": child_q_values["up"],
                "down": child_q_values["down"],
                "left": child_q_values["left"],
                "right": child_q_values["right"]
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

        return reconstructed_df

    def save_q_table_reconstructed(self, filepath=None, filename=None):
        reconstructed_q_table = self.reconstruct_q_table_from_parent_child()
        filepath = filepath or self.filepath
        filename = filename or self.filename
        os.makedirs(filepath, exist_ok=True)
        reconstructed_q_table.to_csv(
            os.path.join(filepath, filename + "_reconstructed_" + str(self.size) + ".csv"),
            index=False
        )

# ----------------------------------------------------------------------
# Training Helper
# ----------------------------------------------------------------------
def run_training_updated(
    env=Environment(size=2),
    filename="q_table",
    filepath="./Data/",
    episodes=1000,
    grid_size=2,
    epsilon=1.0,
    epsilon_decay=0.995,
    epsilon_min=0.01,
    learning_rate=0.1,
    max_depth=5,
    save_interval=10
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
        size=grid_size
    )

    agent.load_q_table_single(filepath=filepath, filename=filename)

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
            print(f"Episode {episode + 1} completed.")
            agent.compress_and_save_tables()
            agent.save_q_table_reconstructed(filepath=filepath, filename=filename)
            agent.save_q_table_single()

    agent.compress_and_save_tables()
    agent.save_q_table_reconstructed(filepath=filepath, filename=filename)
    agent.save_q_table_single()


def main():
    filepath = "./Data/"
    filename = "q_table"

    run_training_updated(
        env=None,
        filename=filename,
        filepath=filepath,
        episodes=1000,
        grid_size=2,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        learning_rate=0.1,
        gamma=1.0,
        max_depth=5,
        save_interval=100,
        compress_interval=500,
        load_existing=True,
    )


if __name__ == "__main__":
    print("Recommendation: Run the 'main.py' file to execute the program.")
    main()