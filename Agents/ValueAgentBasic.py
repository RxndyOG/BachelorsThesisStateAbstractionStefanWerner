import os
import csv
import numpy as np
import sqlite3

from Bachelor.PreCalculator import PreCalculator
from Environment.Environment import Environment


class Agent:
    ACTIONS = ["up", "down", "left", "right"]
    ACTION_INDEX = {
        "up": 0,
        "down": 1,
        "left": 2,
        "right": 3
    }

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

        # dict:
        # {
        #   (0,2,4,8): [q_up, q_down, q_left, q_right]
        # }
        self.q_table = {}

        self.p_dict = {}
        self.c_dict = {}
        self.dig_deeper_dict = {}

        self.filepath = filepath
        self.filename = filename

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate

        self.precal = PreCalculator()

        self.size = size
        self.max_depth = max_depth

    # -------------------------------------------------
    # STATE / KEY HELPERS
    # -------------------------------------------------
    def state_to_key(self, state):
        state_array = np.array(state, dtype=int)
        return tuple(state_array.flatten())

    def key_to_state_array(self, state_key):
        return np.array(state_key, dtype=int).reshape(self.size, self.size)

    def state_key_to_str(self, state_key):
        return ",".join(map(str, state_key))

    def state_str_to_key(self, state_str):
        return tuple(map(int, state_str.split(",")))

    def create_sqlite_connection(self, db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("PRAGMA journal_mode=DELETE;")
        cursor.execute("PRAGMA synchronous=OFF;")
        cursor.execute("PRAGMA temp_store=MEMORY;")

        return conn, cursor

    # -------------------------------------------------
    # EPSILON
    # -------------------------------------------------
    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

    # -------------------------------------------------
    # Q-TABLE ENTRY
    # -------------------------------------------------
    def create_q_table_entry(self, state):
        state_key = self.state_to_key(state)

        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0, 0.0, 0.0, 0.0]

    # -------------------------------------------------
    # ACTION CHOICE
    # -------------------------------------------------
    def choose_action(self, state, available_actions):
        if not available_actions:
            return None

        # Exploration
        if np.random.rand() < self.epsilon:
            return np.random.choice(available_actions)

        state_key = self.state_to_key(state)

        if state_key not in self.q_table:
            return np.random.choice(available_actions)

        q_values = self.q_table[state_key]

        # nur verfügbare Actions betrachten
        best_value = None
        best_actions = []

        for action in available_actions:
            idx = self.ACTION_INDEX[action]
            value = q_values[idx]

            if best_value is None or value > best_value:
                best_value = value
                best_actions = [action]
            elif value == best_value:
                best_actions.append(action)

        return np.random.choice(best_actions)

    # -------------------------------------------------
    # LEARNING
    # -------------------------------------------------
    def learn(self, state, action, reward, next_state, available_actions):
        if action is None:
            return

        state_key = self.state_to_key(state)
        next_state_key = self.state_to_key(next_state)

        if state_key not in self.q_table:
            self.create_q_table_entry(state)

        if next_state_key not in self.q_table:
            self.create_q_table_entry(next_state)

        action_idx = self.ACTION_INDEX[action]
        current_q_value = self.q_table[state_key][action_idx]

        # max Q vom nächsten State über verfügbare Actions
        if available_actions:
            next_q_values = self.q_table[next_state_key]
            next_max_q_value = max(
                next_q_values[self.ACTION_INDEX[a]]
                for a in available_actions
            )
        else:
            next_max_q_value = 0.0

        # Achtung:
        # du hattest vorher kein gamma drin, deshalb lasse ich es genauso
        new_q_value = (
            (1 - self.learning_rate) * current_q_value
            + self.learning_rate * (reward + next_max_q_value)
        )

        self.q_table[state_key][action_idx] = new_q_value

    # -------------------------------------------------
    # CSV SAVE / LOAD
    # -------------------------------------------------
    def save_q_table_single(self, filepath=None, filename=None):
        if filepath is None:
            filepath = self.filepath
        if filename is None:
            filename = self.filename

        os.makedirs(filepath, exist_ok=True)
        full_path = os.path.join(filepath, f"{filename}_single_{self.size}.csv")

        with open(full_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file, delimiter=";")
            writer.writerow(["state", "up", "down", "left", "right"])
            i = 0
            for state_key, action_values in self.q_table.items():
                i += 1
                if i % 100 == 0:
                    print(f"100 lines written to csv from {len(self.q_table)}")
                state_str = ",".join(map(str, state_key))
                writer.writerow([state_str] + list(action_values))

        print(f"Single Q-Table saved to: {full_path}")

    def load_q_table_single(self, filepath=None, filename=None):
        if filepath is None:
            filepath = self.filepath
        if filename is None:
            filename = self.filename

        full_path = os.path.join(filepath, f"{filename}_single_{self.size}.csv")

        if not os.path.exists(full_path):
            print(f"No Q-Table found at: {full_path}")
            return

        self.q_table = {}

        with open(full_path, "r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file, delimiter=";")

            for row in reader:
                state_key = tuple(map(int, row["state"].split(",")))
                action_values = [
                    float(row["up"]),
                    float(row["down"]),
                    float(row["left"]),
                    float(row["right"])
                ]
                self.q_table[state_key] = action_values

        print(f"Single Q-Table loaded from: {full_path}")

    # -------------------------------------------------
    # OPTIONAL PLACEHOLDER
    # -------------------------------------------------
    # Diese Funktionen hast du im Trainingscode noch drin.
    # Weil ihr Inhalt hier nicht gezeigt wurde, lasse ich sie erstmal
    # als Platzhalter drin, damit der Code nicht crasht.
    def get_best_direction(self, action_values):
        """
        action_values ist z. B.:
        [up, down, left, right]

        Rückgabe:
        0 = up
        1 = down
        2 = left
        3 = right
        """
        best_value = max(action_values)
        return action_values.index(best_value)

    def action_list_to_dict(self, action_values):
        """
        [up, down, left, right] -> {"up":..., "down":..., ...}
        """
        return {
            "up": action_values[0],
            "down": action_values[1],
            "left": action_values[2],
            "right": action_values[3],
        }
    
    def divide_q_table_chunked(
        self,
        filepath=None,
        filename=None,
        progress_interval=1000,
        insert_batch_size=5000,
        keep_index_db=False
    ):
        """
        RAM-schonende Variante von divide_q_table().

        Ablauf:
        1. Liest die bereits gespeicherte Single-Q-Table CSV zeilenweise.
        2. Nutzt SQLite als on-disk reachable_lookup.
        3. Schreibt Parent- und Child-CSV direkt beim Durchlauf.

        Dadurch bleibt die Logik erhalten, aber der RAM-Verbrauch sinkt stark.
        """

        if filepath is None:
            filepath = self.filepath
        if filename is None:
            filename = self.filename

        os.makedirs(filepath, exist_ok=True)

        single_path = os.path.join(filepath, f"{filename}_single_{self.size}.csv")
        parent_path = os.path.join(filepath, f"{filename}_parent_{self.size}.csv")
        child_path = os.path.join(filepath, f"{filename}_child_{self.size}.csv")
        db_path = os.path.join(filepath, f"{filename}_reachable_index_{self.size}.sqlite")

        if not os.path.exists(single_path):
            raise FileNotFoundError(
                f"Single Q-Table not found at: {single_path}\n"
                f"Save the single Q-table first before dividing."
            )

        if os.path.exists(db_path):
            os.remove(db_path)

        conn, cursor = self.create_sqlite_connection(db_path)

        cursor.execute("""
            CREATE TABLE reachable (
                state TEXT PRIMARY KEY,
                parent_index INTEGER NOT NULL,
                operations TEXT NOT NULL
            )
        """)
        cursor.execute("CREATE INDEX idx_reachable_parent_index ON reachable(parent_index)")

        parent_index_counter = 0
        child_index_counter = 0
        pending_reachable = []

        precalc = PreCalculator()

        with open(single_path, "r", newline="", encoding="utf-8") as single_file, \
             open(parent_path, "w", newline="", encoding="utf-8") as parent_file, \
             open(child_path, "w", newline="", encoding="utf-8") as child_file:

            reader = csv.DictReader(single_file, delimiter=";")
            parent_writer = csv.writer(parent_file, delimiter=";")
            child_writer = csv.writer(child_file, delimiter=";")

            parent_writer.writerow([
                "index",
                "state",
                "up",
                "down",
                "left",
                "right",
                "direction"
            ])

            child_writer.writerow([
                "index",
                "parent_index",
                "operations",
                "direction"
            ])

            for row_number, row in enumerate(reader, start=1):
                state_str = row["state"].strip()
                state_key = self.state_str_to_key(state_str)

                action_values = [
                    float(row["up"]),
                    float(row["down"]),
                    float(row["left"]),
                    float(row["right"])
                ]

                cursor.execute(
                    "SELECT parent_index, operations FROM reachable WHERE state = ?",
                    (state_str,)
                )
                hit = cursor.fetchone()

                # -----------------------------------------
                # FALL 1: State ist schon erreichbar -> Child
                # -----------------------------------------
                if hit is not None:
                    parent_index, operations = hit
                    child_best_direction = self.get_best_direction(action_values)

                    child_writer.writerow([
                        child_index_counter,
                        parent_index,
                        operations,
                        child_best_direction
                    ])
                    child_index_counter += 1

                # -----------------------------------------
                # FALL 2: State ist nicht erreichbar -> Parent
                # -----------------------------------------
                else:
                    parent_best_direction = self.get_best_direction(action_values)

                    parent_writer.writerow([
                        parent_index_counter,
                        state_str,
                        action_values[0],
                        action_values[1],
                        action_values[2],
                        action_values[3],
                        parent_best_direction
                    ])

                    parent_state_array = self.key_to_state_array(state_key)
                    parent_actions_dict = self.action_list_to_dict(action_values)

                    dig_input = [parent_state_array, parent_actions_dict]
                    dig_results = precalc.dig_deeper(dig_input, max_depth=self.max_depth)

                    for item in dig_results:
                        operations = item["path"]

                        if operations == "":
                            continue

                        reachable_state_key = self.state_to_key(item["state"])
                        reachable_state_str = self.state_key_to_str(reachable_state_key)

                        pending_reachable.append((
                            reachable_state_str,
                            parent_index_counter,
                            operations
                        ))

                    parent_index_counter += 1

                # Batch-Insert für SQLite
                if len(pending_reachable) >= insert_batch_size:
                    cursor.executemany("""
                        INSERT OR IGNORE INTO reachable (state, parent_index, operations)
                        VALUES (?, ?, ?)
                    """, pending_reachable)
                    conn.commit()
                    pending_reachable.clear()

                if row_number % progress_interval == 0:
                    print(
                        f"[divide_q_table_chunked] processed={row_number}, "
                        f"parents={parent_index_counter}, children={child_index_counter}"
                    )

            # Rest flushen
            if pending_reachable:
                cursor.executemany("""
                    INSERT OR IGNORE INTO reachable (state, parent_index, operations)
                    VALUES (?, ?, ?)
                """, pending_reachable)
                conn.commit()
                pending_reachable.clear()

        cursor.close()
        conn.close()

        if not keep_index_db and os.path.exists(db_path):
            os.remove(db_path)

        print(f"Parent Q-Table saved to: {parent_path}")
        print(f"Child Q-Table saved to: {child_path}")

    def divide_q_table(self):
        """
        Zerlegt self.q_table in:

        p_dict:
            parent_index -> {
                "state": tuple,
                "up": float,
                "down": float,
                "left": float,
                "right": float,
                "direction": int
            }

        c_dict:
            child_index -> {
                "parent_index": int,
                "operations": str,
                "direction": int
            }

        dig_deeper_dict:
            parent_index -> [
                [parent_index, reachable_state_tuple, operations],
                ...
            ]

        direction:
            0 = up
            1 = down
            2 = left
            3 = right
        """

        self.p_dict = {}
        self.c_dict = {}
        self.dig_deeper_dict = {}

        precalc = PreCalculator()

        # reachable_state -> (parent_index, operations)
        reachable_lookup = {}

        parent_index_counter = 0
        child_index_counter = 0

        q_items = list(self.q_table.items())

        for state_key, action_values in q_items:

            # -----------------------------------------
            # FALL 1:
            # State ist schon durch irgendeinen Parent erreichbar
            # -> Child
            # -----------------------------------------
            if state_key in reachable_lookup:
                parent_index, operations = reachable_lookup[state_key]

                child_best_direction = self.get_best_direction(action_values)

                self.c_dict[child_index_counter] = {
                    "parent_index": parent_index,
                    "operations": operations,
                    "direction": child_best_direction
                }
                child_index_counter += 1
                continue

            # -----------------------------------------
            # FALL 2:
            # State ist NICHT erreichbar
            # -> neuer Parent
            # -----------------------------------------
            parent_best_direction = self.get_best_direction(action_values)

            self.p_dict[parent_index_counter] = {
                "state": state_key,
                "up": action_values[0],
                "down": action_values[1],
                "left": action_values[2],
                "right": action_values[3],
                "direction": parent_best_direction
            }

            parent_state_array = self.key_to_state_array(state_key)
            parent_actions_dict = self.action_list_to_dict(action_values)

            dig_input = [parent_state_array, parent_actions_dict]
            dig_results = precalc.dig_deeper(dig_input, max_depth=self.max_depth)

            self.dig_deeper_dict[parent_index_counter] = []

            for item in dig_results:
                reachable_state_key = self.state_to_key(item["state"])
                operations = item["path"]

                if operations == "":
                    continue

                entry = [
                    parent_index_counter,
                    reachable_state_key,
                    operations
                ]

                self.dig_deeper_dict[parent_index_counter].append(entry)

                if reachable_state_key not in reachable_lookup:
                    reachable_lookup[reachable_state_key] = (
                        parent_index_counter,
                        operations
                    )

            parent_index_counter += 1

    def save_q_table_parent_child(self, filepath=None, filename=None):
        if filepath is None:
            filepath = self.filepath
        if filename is None:
            filename = self.filename

        os.makedirs(filepath, exist_ok=True)

        # -----------------------------------
        # 1. PARENT TABLE
        # -----------------------------------
        parent_path = os.path.join(filepath, f"{filename}_parent_{self.size}.csv")

        with open(parent_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")

            writer.writerow([
                "index",
                "state",
                "up",
                "down",
                "left",
                "right",
                "direction"
            ])

            for index, data in self.p_dict.items():
                state_str = ",".join(map(str, data["state"]))

                writer.writerow([
                    index,
                    state_str,
                    data["up"],
                    data["down"],
                    data["left"],
                    data["right"],
                    data["direction"]
                ])

        print(f"Parent Q-Table saved to: {parent_path}")

        # -----------------------------------
        # 2. CHILD TABLE
        # -----------------------------------
        child_path = os.path.join(filepath, f"{filename}_child_{self.size}.csv")

        with open(child_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")

            writer.writerow([
                "index",
                "parent_index",
                "operations",
                "direction"
            ])

            for index, data in self.c_dict.items():
                writer.writerow([
                    index,
                    data["parent_index"],
                    data["operations"],
                    data["direction"]
                ])

        print(f"Child Q-Table saved to: {child_path}")

    def action_dict_to_list(self, actions_dict):
        """
        {"up":..., "down":..., "left":..., "right":...}
        ->
        [up, down, left, right]
        """
        return [
            actions_dict["up"],
            actions_dict["down"],
            actions_dict["left"],
            actions_dict["right"]
        ]

    def parse_operations(self, operations):
        """
        Unterstützt:
        "rddr"   -> [("r",1), ("d",1), ("d",1), ("r",1)]
        "r2d3"   -> [("r",2), ("d",3)]
        """
        result = []
        i = 0

        while i < len(operations):
            op = operations[i]
            i += 1

            num_str = ""
            while i < len(operations) and operations[i].isdigit():
                num_str += operations[i]
                i += 1

            count = int(num_str) if num_str else 1
            result.append((op, count))

        return result

    def apply_operations_to_state_actions(self, state_array, action_values, operations):
        """
        Wendet operations auf State + Actions an.

        state_array: np.array
        action_values: [up, down, left, right]
        operations: z. B. "rddr" oder "r2d3"
        """
        precalc = PreCalculator()

        current_state = np.array(state_array, dtype=int).copy()
        current_actions = self.action_list_to_dict(list(action_values))

        parsed_ops = self.parse_operations(operations)

        for op, count in parsed_ops:
            if op not in precalc.operations:
                continue

            state_func, action_func = precalc.operations[op]

            for _ in range(count):
                current_state = state_func(current_state)
                current_actions = action_func(current_actions)

        return current_state, self.action_dict_to_list(current_actions)

    def force_best_direction_minimal_loss(self, action_values, wanted_direction, epsilon=1e-6):
        """
        Sorgt dafür, dass wanted_direction die beste Action ist,
        mit möglichst kleiner Änderung.

        Strategie:
        - wenn wanted_direction schon eindeutig beste Action ist -> nichts ändern
        - sonst nur diesen einen Wert minimal über das aktuelle Maximum anheben
        """
        values = list(action_values)

        current_best = max(values)
        current_best_idx = values.index(current_best)

        # Falls gewünschte Richtung schon die beste ist:
        # bei Gleichstand mit anderen machen wir sie leicht eindeutig
        if current_best_idx == wanted_direction:
            tied_indices = [i for i, v in enumerate(values) if v == current_best]
            if len(tied_indices) > 1:
                values[wanted_direction] = current_best + epsilon
            return values

        # gewünschte Richtung minimal über das bisherige Maximum heben
        values[wanted_direction] = current_best + epsilon
        return values

    def save_q_table_reconstructed_chunked(self, filepath=None, filename=None, keep_parent_db=False):
        """
        Rekonstruiert die Q-Table direkt aus Parent- und Child-CSV,
        ohne p_dict/c_dict komplett im RAM zu halten.
        """
        if filepath is None:
            filepath = self.filepath
        if filename is None:
            filename = self.filename

        os.makedirs(filepath, exist_ok=True)

        parent_path = os.path.join(filepath, f"{filename}_parent_{self.size}.csv")
        child_path = os.path.join(filepath, f"{filename}_child_{self.size}.csv")
        reconstructed_path = os.path.join(filepath, f"{filename}_reconstructed_{self.size}.csv")
        parent_db_path = os.path.join(filepath, f"{filename}_parent_lookup_{self.size}.sqlite")

        if not os.path.exists(parent_path):
            raise FileNotFoundError(f"Parent CSV not found at: {parent_path}")
        if not os.path.exists(child_path):
            raise FileNotFoundError(f"Child CSV not found at: {child_path}")

        if os.path.exists(parent_db_path):
            os.remove(parent_db_path)

        conn, cursor = self.create_sqlite_connection(parent_db_path)

        cursor.execute("""
            CREATE TABLE parents (
                parent_index INTEGER PRIMARY KEY,
                state TEXT NOT NULL,
                up REAL NOT NULL,
                down REAL NOT NULL,
                left REAL NOT NULL,
                right REAL NOT NULL,
                direction INTEGER NOT NULL
            )
        """)

        # Parents in SQLite laden
        with open(parent_path, "r", newline="", encoding="utf-8") as parent_file:
            reader = csv.DictReader(parent_file, delimiter=";")
            batch = []

            for row in reader:
                batch.append((
                    int(row["index"]),
                    row["state"].strip(),
                    float(row["up"]),
                    float(row["down"]),
                    float(row["left"]),
                    float(row["right"]),
                    int(row["direction"])
                ))

                if len(batch) >= 5000:
                    cursor.executemany("""
                        INSERT INTO parents (
                            parent_index, state, up, down, left, right, direction
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, batch)
                    conn.commit()
                    batch.clear()

            if batch:
                cursor.executemany("""
                    INSERT INTO parents (
                        parent_index, state, up, down, left, right, direction
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, batch)
                conn.commit()
                batch.clear()

        # Rekonstruierte CSV schreiben
        with open(reconstructed_path, "w", newline="", encoding="utf-8") as recon_file:
            writer = csv.writer(recon_file, delimiter=";")
            writer.writerow(["state", "up", "down", "left", "right"])

            # 1. Parents direkt schreiben
            cursor.execute("""
                SELECT parent_index, state, up, down, left, right, direction
                FROM parents
                ORDER BY parent_index
            """)

            for row in cursor.fetchall():
                _, state_str, up, down, left, right, _ = row
                writer.writerow([state_str, up, down, left, right])

            # 2. Children rekonstruieren
            with open(child_path, "r", newline="", encoding="utf-8") as child_file:
                child_reader = csv.DictReader(child_file, delimiter=";")

                for i, row in enumerate(child_reader, start=1):
                    parent_index = int(row["parent_index"])
                    operations = row["operations"]
                    wanted_direction = int(row["direction"])

                    cursor.execute("""
                        SELECT state, up, down, left, right
                        FROM parents
                        WHERE parent_index = ?
                    """, (parent_index,))
                    parent_row = cursor.fetchone()

                    if parent_row is None:
                        continue

                    parent_state_str, up, down, left, right = parent_row
                    parent_state_key = self.state_str_to_key(parent_state_str)

                    parent_state = self.key_to_state_array(parent_state_key)
                    parent_actions = [up, down, left, right]

                    child_state_array, child_actions = self.apply_operations_to_state_actions(
                        parent_state,
                        parent_actions,
                        operations
                    )

                    reconstructed_best_direction = self.get_best_direction(child_actions)

                    if reconstructed_best_direction != wanted_direction:
                        child_actions = self.force_best_direction_minimal_loss(
                            child_actions,
                            wanted_direction
                        )

                    child_state_key = self.state_to_key(child_state_array)
                    child_state_str = self.state_key_to_str(child_state_key)

                    writer.writerow([child_state_str] + list(child_actions))

                    if i % 5000 == 0:
                        print(f"[reconstruct] processed children: {i}")

        cursor.close()
        conn.close()

        if not keep_parent_db and os.path.exists(parent_db_path):
            os.remove(parent_db_path)

        print(f"Reconstructed Q-Table saved to: {reconstructed_path}")

    def build_reconstructed_q_table(self):
        """
        Baut die rekonstruierte Q-Table als dict:

        {
            state_tuple: [up, down, left, right]
        }

        Parent-Einträge werden direkt übernommen.
        Child-Einträge werden aus Parent + operations rekonstruiert.

        Falls die rekonstruierte beste Direction nicht mit der im Child
        gespeicherten Direction übereinstimmt, werden die Actions minimal
        angepasst, damit die gespeicherte Child-Direction wieder beste Action ist.
        """
        reconstructed = {}

        # -----------------------------------
        # 1. Parents direkt übernehmen
        # -----------------------------------
        for parent_index, pdata in self.p_dict.items():
            state_key = tuple(pdata["state"])
            reconstructed[state_key] = [
                pdata["up"],
                pdata["down"],
                pdata["left"],
                pdata["right"]
            ]

        # -----------------------------------
        # 2. Childs rekonstruieren
        # -----------------------------------
        for child_index, cdata in self.c_dict.items():
            parent_index = cdata["parent_index"]
            operations = cdata["operations"]
            wanted_direction = cdata["direction"]

            if parent_index not in self.p_dict:
                continue

            pdata = self.p_dict[parent_index]

            parent_state = self.key_to_state_array(pdata["state"])
            parent_actions = [
                pdata["up"],
                pdata["down"],
                pdata["left"],
                pdata["right"]
            ]

            child_state_array, child_actions = self.apply_operations_to_state_actions(
                parent_state,
                parent_actions,
                operations
            )

            # Prüfen, ob gewünschte Child-Direction noch passt
            reconstructed_best_direction = self.get_best_direction(child_actions)

            if reconstructed_best_direction != wanted_direction:
                child_actions = self.force_best_direction_minimal_loss(
                    child_actions,
                    wanted_direction
                )

            child_state_key = self.state_to_key(child_state_array)
            reconstructed[child_state_key] = child_actions

        self.reconstructed_q_table = reconstructed
        return reconstructed

    def save_q_table_reconstructed(self, filepath=None, filename=None):
        if filepath is None:
            filepath = self.filepath
        if filename is None:
            filename = self.filename

        os.makedirs(filepath, exist_ok=True)

        reconstructed = self.build_reconstructed_q_table()
        reconstructed_path = os.path.join(filepath, f"{filename}_reconstructed_{self.size}.csv")

        with open(reconstructed_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(["state", "up", "down", "left", "right"])

            for state_key, action_values in reconstructed.items():
                state_str = ",".join(map(str, state_key))
                writer.writerow([state_str] + list(action_values))

        print(f"Reconstructed Q-Table saved to: {reconstructed_path}")


def run_training_basic(
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
            agent.save_q_table_single()

    agent.save_q_table_single()
    agent.divide_q_table_chunked()
    agent.save_q_table_reconstructed_chunked()


def main():
    filepath = "./Data/"
    filename = "q_table"

    grid_size = 2
    env = Environment(size=grid_size)

    agent = Agent(
        environment=env,
        filepath=filepath,
        filename=filename,
        max_depth=5,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        learning_rate=0.1,
        size=grid_size
    )

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
            agent.divide_q_table_chunked()
            agent.save_q_table_reconstructed_chunked()
            agent.save_q_table_single()

    agent.divide_q_table_chunked()
    agent.save_q_table_reconstructed_chunked()
    agent.save_q_table_single()


if __name__ == "__main__":
    print("Recommendation: Run the 'main.py' file to execute the program.")
    main()