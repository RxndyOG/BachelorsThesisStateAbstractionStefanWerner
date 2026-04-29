import csv
import os
import sqlite3
import numpy as np

from Bachelor.PreCalculator import PreCalculator
from Environment.Environment import Environment
import time
import threading
from datetime import datetime

try:
    import psutil
except ImportError:
    psutil = None

class MemoryMonitor:
    """
    Misst den RAM-Verbrauch des aktuellen Python-Prozesses während eines Code-Abschnitts.
    Benötigt psutil. Wenn psutil nicht installiert ist, bleiben die Werte None.
    """

    def __init__(self, interval=0.1):
        self.interval = interval
        self.running = False
        self.thread = None
        self.peak_mb = None
        self.start_mb = None
        self.end_mb = None

    def _get_memory_mb(self):
        if psutil is None:
            return None

        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)

    def _watch(self):
        while self.running:
            current_mb = self._get_memory_mb()

            if current_mb is not None:
                if self.peak_mb is None or current_mb > self.peak_mb:
                    self.peak_mb = current_mb

            time.sleep(self.interval)

    def start(self):
        self.start_mb = self._get_memory_mb()
        self.peak_mb = self.start_mb
        self.running = True

        if psutil is not None:
            self.thread = threading.Thread(target=self._watch, daemon=True)
            self.thread.start()

    def stop(self):
        self.running = False

        if self.thread is not None:
            self.thread.join()

        self.end_mb = self._get_memory_mb()


def measure_stage(stage_name, function_to_run):
    """
    Misst Laufzeit und RAM eines bestimmten Programmabschnitts.
    """
    monitor = MemoryMonitor()
    monitor.start()

    start_time = time.perf_counter()
    function_to_run()
    end_time = time.perf_counter()

    monitor.stop()

    duration_seconds = end_time - start_time

    return {
        "stage": stage_name,
        "duration_seconds": duration_seconds,
        "ram_start_mb": monitor.start_mb,
        "ram_end_mb": monitor.end_mb,
        "ram_peak_mb": monitor.peak_mb,
    }


def append_timing_results(filepath, filename, grid_size, episodes, max_depth, timing_rows):
    """
    Speichert die gemessenen Zeiten und RAM-Werte in eine CSV.
    Die Datei wird erweitert und nicht überschrieben.
    """
    os.makedirs(filepath, exist_ok=True)

    timing_path = os.path.join(
        filepath,
        f"{filename}_timing_results_{grid_size}.csv"
    )

    file_exists = os.path.exists(timing_path)

    with open(timing_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter=";")

        if not file_exists:
            writer.writerow([
                "timestamp",
                "filename",
                "grid_size",
                "episodes",
                "max_depth",
                "stage",
                "duration_seconds",
                "ram_start_mb",
                "ram_end_mb",
                "ram_peak_mb",
            ])

        timestamp = datetime.now().isoformat(timespec="seconds")

        for row in timing_rows:
            writer.writerow([
                timestamp,
                filename,
                grid_size,
                episodes,
                max_depth,
                row["stage"],
                row["duration_seconds"],
                row["ram_start_mb"],
                row["ram_end_mb"],
                row["ram_peak_mb"],
            ])

    print(f"Timing results appended to: {timing_path}")

class Agent:
    """
    Q-Learning Agent für 2048 mit zusätzlicher Unterstützung für:

    - Speichern/Laden der normalen Q-Table
    - Aufteilen der Q-Table in Parent- und Child-Einträge
    - Rekonstruktion einer Q-Table aus Parent-/Child-Daten

    Die Lernlogik bleibt dabei unverändert.
    """

    ACTIONS = ["up", "down", "left", "right"]
    ACTION_INDEX = {
        "up": 0,
        "down": 1,
        "left": 2,
        "right": 3,
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
        size=2,
    ):
        self.environment = environment
        self.filepath = filepath
        self.filename = filename

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate

        self.size = size
        self.max_depth = max_depth

        # Normale Q-Table:
        # { (0, 2, 4, 8): [q_up, q_down, q_left, q_right] }
        self.q_table = {}

        # Datenstrukturen für Parent-/Child-Aufteilung
        self.p_dict = {}
        self.c_dict = {}
        self.dig_deeper_dict = {}

        # Rekonstruierte Tabelle
        self.reconstructed_q_table = {}

    # -------------------------------------------------
    # STATE / KEY / STRING HELPER
    # -------------------------------------------------
    def state_to_key(self, state):
        """
        Wandelt einen State in einen flachen Tuple-Key um.
        Beispiel:
        [[0,2],
         [4,8]]
        -> (0,2,4,8)
        """
        state_array = np.array(state, dtype=int)
        return tuple(state_array.flatten())

    def key_to_state_array(self, state_key):
        """
        Wandelt einen Tuple-Key zurück in ein NumPy-Array
        mit der passenden Grid-Größe.
        """
        return np.array(state_key, dtype=int).reshape(self.size, self.size)

    def state_key_to_str(self, state_key):
        """
        Wandelt einen State-Key in einen CSV-tauglichen String um.
        Beispiel:
        (0,2,4,8) -> "0,2,4,8"
        """
        return ",".join(map(str, state_key))

    def state_str_to_key(self, state_str):
        """
        Wandelt einen CSV-State-String zurück in einen Tuple-Key um.
        Beispiel:
        "0,2,4,8" -> (0,2,4,8)
        """
        return tuple(map(int, state_str.split(",")))

    # -------------------------------------------------
    # SQLITE HELPER
    # -------------------------------------------------
    def create_sqlite_connection(self, db_path):
        """
        Erstellt eine SQLite-Verbindung mit pragmatischen Performance-Settings,
        damit große Zwischendatenmengen möglichst RAM-schonend verarbeitet
        werden können.
        """
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
        """
        Reduziert epsilon schrittweise bis zum Minimalwert.
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

    # -------------------------------------------------
    # Q-TABLE BASIS
    # -------------------------------------------------
    def create_q_table_entry(self, state):
        """
        Legt einen neuen State in der Q-Table an, falls er noch nicht existiert.
        Initialwerte aller vier Aktionen sind 0.0.
        """
        state_key = self.state_to_key(state)

        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0, 0.0, 0.0, 0.0]

    def choose_action(self, state, available_actions):
        """
        Wählt eine Aktion mittels epsilon-greedy:

        - Mit Wahrscheinlichkeit epsilon: zufällige Exploration
        - Sonst: beste bekannte Aktion aus der Q-Table
        - Bei Gleichstand: zufällige Wahl unter den besten Aktionen

        Es werden nur aktuell verfügbare Aktionen betrachtet.
        """
        if not available_actions:
            return None

        # Exploration
        if np.random.rand() < self.epsilon:
            return np.random.choice(available_actions)

        state_key = self.state_to_key(state)

        if state_key not in self.q_table:
            return np.random.choice(available_actions)

        q_values = self.q_table[state_key]

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

    def learn(self, state, action, reward, next_state, available_actions):
        """
        Führt das Q-Value-Update für einen Schritt aus.

        Hinweis:
        Hier wird bewusst dieselbe Logik wie im Original beibehalten.
        Es gibt also weiterhin keinen expliziten Gamma-Faktor.
        """
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

        if available_actions:
            next_q_values = self.q_table[next_state_key]
            next_max_q_value = max(
                next_q_values[self.ACTION_INDEX[a]]
                for a in available_actions
            )
        else:
            next_max_q_value = 0.0

        new_q_value = (
            (1 - self.learning_rate) * current_q_value
            + self.learning_rate * (reward + next_max_q_value)
        )

        self.q_table[state_key][action_idx] = new_q_value

    # -------------------------------------------------
    # CSV: SINGLE Q-TABLE
    # -------------------------------------------------
    def save_q_table_single(self, filepath=None, filename=None):
        """
        Speichert die aktuelle vollständige Q-Table als CSV.
        """
        if filepath is None:
            filepath = self.filepath
        if filename is None:
            filename = self.filename

        os.makedirs(filepath, exist_ok=True)
        full_path = os.path.join(filepath, f"{filename}_single_{self.size}.csv")

        with open(full_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file, delimiter=";")
            writer.writerow(["state", "up", "down", "left", "right"])

            for i, (state_key, action_values) in enumerate(self.q_table.items(), start=1):
                if i % 10000 == 0:
                    print(f"10000 lines written to csv from {len(self.q_table)}")

                state_str = self.state_key_to_str(state_key)
                writer.writerow([state_str] + list(action_values))

        print(f"Single Q-Table saved to: {full_path}")

    def load_q_table_single(self, filepath=None, filename=None):
        """
        Lädt eine vollständige Q-Table aus CSV in den Arbeitsspeicher.
        """
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
                state_key = self.state_str_to_key(row["state"])
                action_values = [
                    float(row["up"]),
                    float(row["down"]),
                    float(row["left"]),
                    float(row["right"]),
                ]
                self.q_table[state_key] = action_values

        print(f"Single Q-Table loaded from: {full_path}")

    # -------------------------------------------------
    # ACTION / DIRECTION HELPER
    # -------------------------------------------------
    def get_best_direction(self, action_values):
        """
        Gibt den Index der besten Aktion zurück.

        Reihenfolge:
        0 = up
        1 = down
        2 = left
        3 = right
        """
        best_value = max(action_values)
        return action_values.index(best_value)

    def action_list_to_dict(self, action_values):
        """
        Wandelt eine Aktionsliste in ein Dictionary um.

        [up, down, left, right]
        ->
        {"up": ..., "down": ..., "left": ..., "right": ...}
        """
        return {
            "up": action_values[0],
            "down": action_values[1],
            "left": action_values[2],
            "right": action_values[3],
        }

    def action_dict_to_list(self, actions_dict):
        """
        Wandelt ein Aktions-Dictionary in eine Liste um.

        {"up": ..., "down": ..., "left": ..., "right": ...}
        ->
        [up, down, left, right]
        """
        return [
            actions_dict["up"],
            actions_dict["down"],
            actions_dict["left"],
            actions_dict["right"],
        ]

    # -------------------------------------------------
    # Q-TABLE DIVISION (RAM-SCHONEND)
    # -------------------------------------------------
    def divide_q_table_chunked(
        self,
        filepath=None,
        filename=None,
        progress_interval=1000,
        insert_batch_size=5000,
        keep_index_db=False,
    ):
        """
        Teilt eine bereits gespeicherte Single-Q-Table CSV in Parent- und Child-Dateien auf.

        Vorgehen:
        1. Single-Q-Table zeilenweise lesen
        2. Erreichbare Child-States in SQLite zwischenspeichern
        3. Parent- und Child-CSV direkt beim Durchlauf schreiben

        Vorteil:
        Die Logik bleibt gleich, aber der RAM-Verbrauch sinkt deutlich.
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

        cursor.execute(
            """
            CREATE TABLE reachable (
                state TEXT PRIMARY KEY,
                parent_index INTEGER NOT NULL,
                operations TEXT NOT NULL
            )
            """
        )
        cursor.execute("CREATE INDEX idx_reachable_parent_index ON reachable(parent_index)")

        parent_index_counter = 0
        child_index_counter = 0
        pending_reachable = []

        precalc = PreCalculator()

        with (
            open(single_path, "r", newline="", encoding="utf-8") as single_file,
            open(parent_path, "w", newline="", encoding="utf-8") as parent_file,
            open(child_path, "w", newline="", encoding="utf-8") as child_file,
        ):
            reader = csv.DictReader(single_file, delimiter=";")
            parent_writer = csv.writer(parent_file, delimiter=";")
            child_writer = csv.writer(child_file, delimiter=";")

            parent_writer.writerow(["index", "state", "up", "down", "left", "right"])
            child_writer.writerow(["parent_index", "operations", "direction"])

            for row_number, row in enumerate(reader, start=1):
                state_str = row["state"].strip()
                state_key = self.state_str_to_key(state_str)

                action_values = [
                    float(row["up"]),
                    float(row["down"]),
                    float(row["left"]),
                    float(row["right"]),
                ]

                cursor.execute(
                    "SELECT parent_index, operations FROM reachable WHERE state = ?",
                    (state_str,),
                )
                hit = cursor.fetchone()

                # Fall 1: State ist bereits über einen Parent erreichbar -> Child
                if hit is not None:
                    parent_index, operations = hit
                    child_best_direction = self.get_best_direction(action_values)

                    child_writer.writerow(
                        [parent_index, operations, child_best_direction]
                    )
                    child_index_counter += 1

                # Fall 2: State ist noch nicht erreichbar -> neuer Parent
                else:
                    parent_writer.writerow(
                        [
                            parent_index_counter,
                            state_str,
                            action_values[0],
                            action_values[1],
                            action_values[2],
                            action_values[3],
                        ]
                    )

                    parent_state_array = self.key_to_state_array(state_key)
                    parent_actions_dict = self.action_list_to_dict(action_values)

                    dig_input = [parent_state_array, parent_actions_dict]
                    dig_results = precalc.dig_deeper(
                        dig_input,
                        max_depth=self.max_depth,
                    )

                    for item in dig_results:
                        operations = item["path"]

                        if operations == "":
                            continue

                        reachable_state_key = self.state_to_key(item["state"])
                        reachable_state_str = self.state_key_to_str(reachable_state_key)

                        pending_reachable.append(
                            (
                                reachable_state_str,
                                parent_index_counter,
                                operations,
                            )
                        )

                    parent_index_counter += 1

                if len(pending_reachable) >= insert_batch_size:
                    cursor.executemany(
                        """
                        INSERT OR IGNORE INTO reachable (state, parent_index, operations)
                        VALUES (?, ?, ?)
                        """,
                        pending_reachable,
                    )
                    conn.commit()
                    pending_reachable.clear()

                if row_number % progress_interval == 0:
                    print(
                        f"[divide_q_table_chunked] processed={row_number}, "
                        f"parents={parent_index_counter}, children={child_index_counter}"
                    )

            if pending_reachable:
                cursor.executemany(
                    """
                    INSERT OR IGNORE INTO reachable (state, parent_index, operations)
                    VALUES (?, ?, ?)
                    """,
                    pending_reachable,
                )
                conn.commit()
                pending_reachable.clear()

        cursor.close()
        conn.close()

        if not keep_index_db and os.path.exists(db_path):
            os.remove(db_path)

        print(f"Parent Q-Table saved to: {parent_path}")
        print(f"Child Q-Table saved to: {child_path}")


    def save_q_table_parent_child(self, filepath=None, filename=None):
        """
        Speichert die bereits berechneten Parent- und Child-Dictionaries
        als zwei CSV-Dateien.
        """
        if filepath is None:
            filepath = self.filepath
        if filename is None:
            filename = self.filename

        os.makedirs(filepath, exist_ok=True)

        parent_path = os.path.join(filepath, f"{filename}_parent_{self.size}.csv")
        with open(parent_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(["index", "state", "up", "down", "left", "right"])

            for index, data in self.p_dict.items():
                state_str = self.state_key_to_str(data["state"])
                writer.writerow(
                    [
                        index,
                        state_str,
                        data["up"],
                        data["down"],
                        data["left"],
                        data["right"],
                    ]
                )

        print(f"Parent Q-Table saved to: {parent_path}")

        child_path = os.path.join(filepath, f"{filename}_child_{self.size}.csv")
        with open(child_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(["parent_index", "operations", "direction"])

            for _, data in self.c_dict.items():
                writer.writerow(
                    [
                        data["parent_index"],
                        data["operations"],
                        data["direction"],
                    ]
                )

        print(f"Child Q-Table saved to: {child_path}")

    # -------------------------------------------------
    # OPERATIONEN / REKONSTRUKTION
    # -------------------------------------------------
    def parse_operations(self, operations):
        """
        Zerlegt einen Operationsstring in (operation, anzahl)-Paare.

        Beispiele:
        "rddr" -> [("r",1), ("d",1), ("d",1), ("r",1)]
        "r2d3" -> [("r",2), ("d",3)]
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
        Wendet einen Operationsstring sowohl auf den State
        als auch auf die zugehörigen Action-Werte an.

        Dadurch bleiben State-Transformation und Action-Transformation
        synchron.
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

    def force_best_direction_minimal_loss(
        self,
        action_values,
        wanted_direction,
        epsilon=1e-6,
    ):
        """
        Erzwingt, dass wanted_direction die beste Aktion ist,
        mit möglichst kleiner Änderung der Werte.

        Strategie:
        - Ist wanted_direction bereits eindeutig die beste: nichts ändern
        - Ist sie im Gleichstand: minimal anheben
        - Ist sie nicht die beste: minimal über das aktuelle Maximum anheben
        """
        values = list(action_values)

        current_best = max(values)
        current_best_idx = values.index(current_best)

        if current_best_idx == wanted_direction:
            tied_indices = [i for i, v in enumerate(values) if v == current_best]
            if len(tied_indices) > 1:
                values[wanted_direction] = current_best + epsilon
            return values

        values[wanted_direction] = current_best + epsilon
        return values

    def save_q_table_reconstructed_chunked(
        self,
        filepath=None,
        filename=None,
        keep_parent_db=False,
    ):
        """
        Rekonstruiert die Q-Table direkt aus Parent- und Child-CSV,
        ohne p_dict und c_dict vollständig im RAM zu halten.
        """
        if filepath is None:
            filepath = self.filepath
        if filename is None:
            filename = self.filename

        os.makedirs(filepath, exist_ok=True)

        parent_path = os.path.join(filepath, f"{filename}_parent_{self.size}.csv")
        child_path = os.path.join(filepath, f"{filename}_child_{self.size}.csv")
        reconstructed_path = os.path.join(
            filepath,
            f"{filename}_reconstructed_{self.size}.csv",
        )
        parent_db_path = os.path.join(
            filepath,
            f"{filename}_parent_lookup_{self.size}.sqlite",
        )

        if not os.path.exists(parent_path):
            raise FileNotFoundError(f"Parent CSV not found at: {parent_path}")
        if not os.path.exists(child_path):
            raise FileNotFoundError(f"Child CSV not found at: {child_path}")

        if os.path.exists(parent_db_path):
            os.remove(parent_db_path)

        conn, cursor = self.create_sqlite_connection(parent_db_path)

        cursor.execute(
            """
            CREATE TABLE parents (
                parent_index INTEGER PRIMARY KEY,
                state TEXT NOT NULL,
                up REAL NOT NULL,
                down REAL NOT NULL,
                left REAL NOT NULL,
                right REAL NOT NULL
            )
            """
        )

        # Parents in SQLite laden
        with open(parent_path, "r", newline="", encoding="utf-8") as parent_file:
            reader = csv.DictReader(parent_file, delimiter=";")
            batch = []

            for row in reader:
                batch.append(
                    (
                        int(row["index"]),
                        row["state"].strip(),
                        float(row["up"]),
                        float(row["down"]),
                        float(row["left"]),
                        float(row["right"]),
                    )
                )

                if len(batch) >= 5000:
                    cursor.executemany(
                        """
                        INSERT INTO parents (
                            parent_index, state, up, down, left, right
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        batch,
                    )
                    conn.commit()
                    batch.clear()

            if batch:
                cursor.executemany(
                    """
                    INSERT INTO parents (
                        parent_index, state, up, down, left, right
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    batch,
                )
                conn.commit()
                batch.clear()

        with open(reconstructed_path, "w", newline="", encoding="utf-8") as recon_file:
            writer = csv.writer(recon_file, delimiter=";")
            writer.writerow(["state", "up", "down", "left", "right"])

            # Parents direkt schreiben
            cursor.execute(
                """
                SELECT parent_index, state, up, down, left, right
                FROM parents
                ORDER BY parent_index
                """
            )

            for row in cursor.fetchall():
                _, state_str, up, down, left, right = row
                writer.writerow([state_str, up, down, left, right])

            # Children rekonstruieren
            with open(child_path, "r", newline="", encoding="utf-8") as child_file:
                child_reader = csv.DictReader(child_file, delimiter=";")

                for i, row in enumerate(child_reader, start=1):
                    parent_index = int(row["parent_index"])
                    operations = row["operations"]
                    wanted_direction = int(row["direction"])

                    cursor.execute(
                        """
                        SELECT state, up, down, left, right
                        FROM parents
                        WHERE parent_index = ?
                        """,
                        (parent_index,),
                    )
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
                        operations,
                    )

                    reconstructed_best_direction = self.get_best_direction(child_actions)

                    if reconstructed_best_direction != wanted_direction:
                        child_actions = self.force_best_direction_minimal_loss(
                            child_actions,
                            wanted_direction,
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
        Baut eine rekonstruierte Q-Table im RAM auf.

        Vorgehen:
        1. Parent-Einträge direkt übernehmen
        2. Child-Einträge aus Parent + Operations rekonstruieren
        3. Falls nötig, beste Richtung minimal anpassen
        """
        reconstructed = {}

        for _, pdata in self.p_dict.items():
            state_key = tuple(pdata["state"])
            reconstructed[state_key] = [
                pdata["up"],
                pdata["down"],
                pdata["left"],
                pdata["right"],
            ]

        for _, cdata in self.c_dict.items():
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
                pdata["right"],
            ]

            child_state_array, child_actions = self.apply_operations_to_state_actions(
                parent_state,
                parent_actions,
                operations,
            )

            reconstructed_best_direction = self.get_best_direction(child_actions)

            if reconstructed_best_direction != wanted_direction:
                child_actions = self.force_best_direction_minimal_loss(
                    child_actions,
                    wanted_direction,
                )

            child_state_key = self.state_to_key(child_state_array)
            reconstructed[child_state_key] = child_actions

        self.reconstructed_q_table = reconstructed
        return reconstructed
    
    def load_q_table_parent_child(self, filepath=None, filename=None):
        """
        Lädt Parent- und Child-Q-Table aus CSV in p_dict und c_dict.
        """
        if filepath is None:
            filepath = self.filepath
        if filename is None:
            filename = self.filename

        parent_path = os.path.join(filepath, f"{filename}_parent_{self.size}.csv")
        child_path = os.path.join(filepath, f"{filename}_child_{self.size}.csv")

        if not os.path.exists(parent_path):
            raise FileNotFoundError(f"Parent Q-Table not found at: {parent_path}")
        if not os.path.exists(child_path):
            raise FileNotFoundError(f"Child Q-Table not found at: {child_path}")

        self.p_dict = {}
        self.c_dict = {}

        with open(parent_path, "r", newline="", encoding="utf-8") as parent_file:
            reader = csv.DictReader(parent_file, delimiter=";")

            for row in reader:
                parent_index = int(row["index"])
                state_key = self.state_str_to_key(row["state"].strip())

                self.p_dict[parent_index] = {
                    "state": state_key,
                    "up": float(row["up"]),
                    "down": float(row["down"]),
                    "left": float(row["left"]),
                    "right": float(row["right"]),
                }

        with open(child_path, "r", newline="", encoding="utf-8") as child_file:
            reader = csv.DictReader(child_file, delimiter=";")

            for child_index, row in enumerate(reader):
                self.c_dict[child_index] = {
                    "parent_index": int(row["parent_index"]),
                    "operations": row["operations"].strip(),
                    "direction": int(row["direction"]),
                }

        print(f"Parent Q-Table loaded from: {parent_path}")
        print(f"Child Q-Table loaded from: {child_path}")


    def reconstruct_from_parent_child_to_q_table(self, filepath=None, filename=None):
        """
        Lädt Parent- und Child-CSV, rekonstruiert daraus die vollständige Q-Table
        und setzt self.q_table auf die rekonstruierte Tabelle.
        """
        self.load_q_table_parent_child(filepath=filepath, filename=filename)
        self.build_reconstructed_q_table()

        if not self.reconstructed_q_table:
            raise ValueError("Reconstructed Q-Table could not be built.")

        self.q_table = dict(self.reconstructed_q_table)
        self.reconstructed_q_table = {}

        print("Reconstructed Q-Table loaded into agent.q_table")
    
    def load_q_table_reconstructed(self, filepath=None, filename=None):
        """
        Rekonstruiert die vollständige Q-Table aus Parent- und Child-Dateien
        und lädt sie direkt in self.q_table.
        """
        self.reconstruct_from_parent_child_to_q_table(filepath=filepath, filename=filename)  

def count_non_zero(state_str):
    """
    Zählt, wie viele Felder eines State-Strings ungleich 0 sind.
    Beispiel:
    "0,2,0,4" -> 2
    """
    values = list(map(int, state_str.split(",")))
    return sum(1 for v in values if v != 0)


def state_sum(state_str):
    """
    Berechnet die Summe aller Werte eines State-Strings.
    Beispiel:
    "0,2,0,4" -> 6
    """
    values = list(map(int, state_str.split(",")))
    return sum(values)


def run_training(
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
    save_interval=10,
):
    """
    Führt Training aus und misst getrennt:

    1. Training ohne Speichern
    2. Speichern der Single-Q-Table
    3. Splitten in Parent-/Child-Q-Table
    4. Rekonstruktion der Q-Table

    Die Ergebnisse werden am Ende in eine CSV appended.
    """

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
    )

    # Optional: bestehende rekonstruierte Q-Table laden.
    # Diese Zeit wird NICHT gemessen, weil du nur Training/Speichern/Split/Reconstruct messen willst.
    try:
        agent.load_q_table_reconstructed(filepath=filepath, filename=filename)
    except FileNotFoundError:
        print("No parent/child Q-table found. Starting with empty Q-table.")
    except ValueError:
        print("Could not reconstruct Q-table. Starting with empty Q-table.")

    sorted_items = sorted(
        agent.q_table.items(),
        key=lambda item: (
            count_non_zero(",".join(map(str, item[0]))),
            state_sum(",".join(map(str, item[0]))),
        ),
    )

    agent.q_table = dict(sorted_items)

    timing_rows = []

    def training_only():
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

    timing_rows.append(
        measure_stage(
            "training_without_saving",
            training_only,
        )
    )

    timing_rows.append(
        measure_stage(
            "save_single_q_table",
            lambda: agent.save_q_table_single(filepath=filepath, filename=filename),
        )
    )

    timing_rows.append(
        measure_stage(
            "split_parent_child",
            lambda: agent.divide_q_table_chunked(filepath=filepath, filename=filename),
        )
    )

    timing_rows.append(
        measure_stage(
            "reconstruct_q_table",
            lambda: agent.save_q_table_reconstructed_chunked(filepath=filepath, filename=filename),
        )
    )

    append_timing_results(
        filepath=filepath,
        filename=filename,
        grid_size=grid_size,
        episodes=episodes,
        max_depth=max_depth,
        timing_rows=timing_rows,
    )

    print("\n----- TIMING RESULTS -----")
    for row in timing_rows:
        print(
            f"{row['stage']}: "
            f"{row['duration_seconds']:.4f}s, "
            f"RAM start={row['ram_start_mb']}, "
            f"RAM end={row['ram_end_mb']}, "
            f"RAM peak={row['ram_peak_mb']}"
        )

def main():
    """
    Standard-Startpunkt für ein Training mit anschließendem
    Divide- und Reconstruct-Schritt.
    """
    filepath = "./Data/"
    filename = "q_table"
    grid_size = 2
    episodes = 1000

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
        size=grid_size,
    )

    agent.load_q_table_reconstructed(filepath=filepath, filename=filename)

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
            agent.save_q_table_single()

    agent.divide_q_table_chunked()
    agent.save_q_table_reconstructed_chunked()
    agent.save_q_table_single()


if __name__ == "__main__":
    print("Recommendation: Run the 'main.py' file to execute the program.")
    main()