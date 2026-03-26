import csv
import os
import random
from typing import Optional, Tuple

import numpy as np

from Environment.Environment import Environment
from Bachelor.MatrixOperation import Detector, Operation

import csv
import os
import random
from typing import Optional, Tuple

import numpy as np

from Environment.Environment import Environment
from Bachelor.MatrixOperation import Detector, Operation


class QLearningAgent:
    def __init__(
        self,
        env,
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        parent_file="./Data/parents.csv",
        child_file="./Data/children.csv",
        detector_max_depth=10,
    ):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.parent_file = parent_file
        self.child_file = child_file

        self.detector = Detector(max_depth=detector_max_depth)
        self.operation = Operation()

        # parent_id -> {"state": tuple(...), "q": {...}}
        self.parents = {}

        # child_state_tuple -> {"parent_id": int, "operations": str}
        self.children = {}

        # direkter Parent-Lookup
        self.parent_state_to_id = {}

        # nächst freie Parent-ID
        self.next_parent_id = 0

        # NEU: Cache für bereits aufgelöste States
        self.resolution_cache = {}

        # NEU: Bucket-System
        # signature -> [parent_id, parent_id, ...]
        self.parent_buckets = {}

        self.load_tables()

    # --------------------------------------------------
    # State Hilfsfunktionen
    # --------------------------------------------------

    def state_to_tuple(self, state):
        return tuple(int(x) for x in state)

    def tuple_to_matrix(self, state_tuple):
        return np.array(state_tuple, dtype=int).reshape(self.env.size, self.env.size)

    def state_to_csv_string(self, state_tuple):
        return "[" + ",".join(map(str, state_tuple)) + "]"

    def csv_string_to_state(self, state_str):
        state_str = state_str.strip().strip("[]")
        if not state_str:
            return tuple()
        return tuple(int(x.strip()) for x in state_str.split(","))

    # --------------------------------------------------
    # Bucket / Signature
    # --------------------------------------------------

    def get_bucket_signature(self, state_tuple):
        """
        Günstige Signatur, um Kandidaten grob einzugrenzen.
        Relativ sicher auch bei deinen Operationen.

        Enthält:
        - Länge des States
        - Anzahl Nullfelder
        - Anzahl Nicht-Nullfelder
        """
        arr = np.array(state_tuple, dtype=int)
        zero_count = int(np.sum(arr == 0))
        nonzero_count = int(np.sum(arr != 0))
        return (len(state_tuple), zero_count, nonzero_count)

    def get_candidate_score(self, state_tuple, parent_state_tuple):
        """
        Weiche Ähnlichkeitsmetrik zur Sortierung innerhalb eines Buckets.
        Je kleiner, desto ähnlicher.
        """
        a = [x for x in state_tuple if x != 0]
        b = [x for x in parent_state_tuple if x != 0]

        a_sorted = sorted(a)
        b_sorted = sorted(b)

        # Unterschied in Länge + Unterschied in Werten
        length_penalty = abs(len(a_sorted) - len(b_sorted)) * 1000

        min_len = min(len(a_sorted), len(b_sorted))
        value_penalty = sum(abs(a_sorted[i] - b_sorted[i]) for i in range(min_len))

        # Restliche Werte bestrafen
        if len(a_sorted) > min_len:
            value_penalty += sum(abs(x) for x in a_sorted[min_len:])
        if len(b_sorted) > min_len:
            value_penalty += sum(abs(x) for x in b_sorted[min_len:])

        return length_penalty + value_penalty

    def add_parent_to_bucket(self, parent_id, state_tuple):
        sig = self.get_bucket_signature(state_tuple)
        if sig not in self.parent_buckets:
            self.parent_buckets[sig] = []
        self.parent_buckets[sig].append(parent_id)

    # --------------------------------------------------
    # CSV Laden / Speichern
    # --------------------------------------------------

    def load_tables(self):
        self.parents = {}
        self.children = {}
        self.parent_state_to_id = {}
        self.resolution_cache = {}
        self.parent_buckets = {}
        self.next_parent_id = 0

        if os.path.exists(self.parent_file):
            with open(self.parent_file, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    parent_id = int(row["parent_id"])
                    state = self.csv_string_to_state(row["state"])

                    self.parents[parent_id] = {
                        "state": state,
                        "q": {
                            "up": float(row.get("up", 0) or 0),
                            "down": float(row.get("down", 0) or 0),
                            "left": float(row.get("left", 0) or 0),
                            "right": float(row.get("right", 0) or 0),
                        },
                    }

                    self.parent_state_to_id[state] = parent_id
                    self.add_parent_to_bucket(parent_id, state)
                    self.next_parent_id = max(self.next_parent_id, parent_id + 1)

        if os.path.exists(self.child_file):
            with open(self.child_file, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    child_state = self.csv_string_to_state(row["child_state"])
                    parent_id = int(row["parent_id"])
                    operations = row["operations"]

                    self.children[child_state] = {
                        "parent_id": parent_id,
                        "operations": operations,
                    }

                    # Child direkt cachen
                    self.resolution_cache[child_state] = parent_id

    def save_tables(self):
        os.makedirs(os.path.dirname(self.parent_file) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(self.child_file) or ".", exist_ok=True)

        with open(self.parent_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["parent_id", "state", "up", "down", "left", "right"])

            for parent_id, data in sorted(self.parents.items(), key=lambda x: x[0]):
                state_str = self.state_to_csv_string(data["state"])
                q = data["q"]

                writer.writerow([
                    parent_id,
                    state_str,
                    q.get("up", 0.0),
                    q.get("down", 0.0),
                    q.get("left", 0.0),
                    q.get("right", 0.0),
                ])

        with open(self.child_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["child_state", "parent_id", "operations"])

            for child_state, data in self.children.items():
                writer.writerow([
                    self.state_to_csv_string(child_state),
                    data["parent_id"],
                    data["operations"],
                ])

    # --------------------------------------------------
    # Parent / Child Auflösung
    # --------------------------------------------------

    def find_matching_parent(self, state_tuple) -> Tuple[Optional[int], Optional[str]]:
        """
        Schneller als vorher:
        1. Parent direkt?
        2. Child direkt?
        3. Cache?
        4. Nur Parents aus passendem Bucket prüfen
        """
        # 1. direkter Parent
        if state_tuple in self.parent_state_to_id:
            parent_id = self.parent_state_to_id[state_tuple]
            self.resolution_cache[state_tuple] = parent_id
            return parent_id, "n"

        # 2. bereits bekannter Child
        if state_tuple in self.children:
            parent_id = self.children[state_tuple]["parent_id"]
            self.resolution_cache[state_tuple] = parent_id
            return parent_id, self.children[state_tuple]["operations"]

        # 3. Cache
        if state_tuple in self.resolution_cache:
            return self.resolution_cache[state_tuple], "cached"

        # 4. nur Kandidaten aus passendem Bucket
        sig = self.get_bucket_signature(state_tuple)
        candidate_ids = self.parent_buckets.get(sig, [])

        if not candidate_ids:
            return None, None

        # Kandidaten nach Ähnlichkeit sortieren
        candidate_ids = sorted(
            candidate_ids,
            key=lambda pid: self.get_candidate_score(state_tuple, self.parents[pid]["state"])
        )

        state_matrix = self.tuple_to_matrix(state_tuple)

        for parent_id in candidate_ids:
            parent_matrix = self.tuple_to_matrix(self.parents[parent_id]["state"])
            operation_string = self.detector.detect_as_string(state_matrix, parent_matrix)

            if operation_string is not None:
                self.resolution_cache[state_tuple] = parent_id
                return parent_id, operation_string

        return None, None

    def resolve_state(self, state):
        state_tuple = self.state_to_tuple(state)

        parent_id, operations = self.find_matching_parent(state_tuple)

        if parent_id is not None:
            if operations not in ("n", "cached") and state_tuple not in self.children:
                self.children[state_tuple] = {
                    "parent_id": parent_id,
                    "operations": operations,
                }
            return parent_id

        # Neuer Parent
        new_parent_id = self.next_parent_id
        self.next_parent_id += 1

        self.parents[new_parent_id] = {
            "state": state_tuple,
            "q": {"up": 0.0, "down": 0.0, "left": 0.0, "right": 0.0},
        }
        self.parent_state_to_id[state_tuple] = new_parent_id
        self.add_parent_to_bucket(new_parent_id, state_tuple)

        self.resolution_cache[state_tuple] = new_parent_id

        return new_parent_id

    # --------------------------------------------------
    # Q-Table Zugriff
    # --------------------------------------------------

    def get_q_value(self, state, action):
        parent_id = self.resolve_state(state)
        return self.parents[parent_id]["q"].get(action, 0.0)

    def set_q_value(self, state, action, value):
        parent_id = self.resolve_state(state)
        self.parents[parent_id]["q"][action] = value

    # --------------------------------------------------
    # Q-Learning
    # --------------------------------------------------

    def choose_action(self, state, actions):
        if not actions:
            return None

        if random.random() < self.epsilon:
            return random.choice(actions)

        q_values = [self.get_q_value(state, a) for a in actions]
        max_q = max(q_values)
        best_actions = [a for a in actions if self.get_q_value(state, a) == max_q]
        return random.choice(best_actions)

    def learn(self, state, action, reward, next_state, done, next_actions):
        current_q = self.get_q_value(state, action)

        if done or not next_actions:
            max_next_q = 0.0
        else:
            max_next_q = max(self.get_q_value(next_state, a) for a in next_actions)

        new_q = current_q + self.alpha * (
            reward + self.gamma * max_next_q - current_q
        )

        self.set_q_value(state, action, new_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def run_train(
    env,
    episodes=10000,
    path="./Data/",
    parent_filename="parents.csv",
    child_filename="children.csv",
):
    agent = QLearningAgent(
        env,
        parent_file=os.path.join(path, parent_filename),
        child_file=os.path.join(path, child_filename),
    )

    if agent.parents:
        print(f"Geladene Parents: {len(agent.parents)}")
        print(f"Geladene Children: {len(agent.children)}")
    else:
        print("Starte neues Training")

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

            agent.learn(state, action, reward, next_state, done, next_actions)
            state = next_state

        agent.decay_epsilon()

        if (episode + 1) % 100 == 0:
            print(
                f"Episode {episode + 1}, epsilon: {agent.epsilon:.3f}, "
                f"parents: {len(agent.parents)}, children: {len(agent.children)}"
            )

    agent.save_tables()
    return agent
    
                
def main():
    env = Environment(size=4)
    agent = QLearningAgent(env)

    # Laden falls vorhanden
    if os.path.exists("./Data/Code/q_table.pkl"):
        agent.load_q_table("./Data/Code/q_table.pkl")
        print("Q-Table geladen")
    else:
        print("Starte neues Training")

    episodes = 10000

    for episode in range(episodes):
        state = env.reset()
        done = False

        while not done:
            actions = env.find_available_actions()
            if not actions:
                break

            print(f"Episode {episode+1}, State: {state}, Actions: {actions}")

            action = agent.choose_action(state, actions)

            next_state, reward, done, info = env.step(action)

            next_actions = env.find_available_actions()

            agent.learn(state, action, reward, next_state, done, next_actions)

            state = next_state

        agent.decay_epsilon()

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}, epsilon: {agent.epsilon:.3f}")

    # speichern
    agent.save_q_table("./Data/Code/q_table.pkl")
    agent.save_q_table_csv("./Data/Code/q_table.csv")

if __name__ == "__main__":
    print("Recommendation: Run the 'main.py' file to execute the program.")
    main()