import os
import random
import csv

# -----------------------------
# CONFIG
# -----------------------------
OUTPUT_DIR = "./test/"
NUM_ROWS_FULL = 100000
STATE_SIZE = 16

ACTIONS = ["up", "down", "left", "right"]
OPS = ["r", "d"]  # rotation, divide (kannst erweitern)


# -----------------------------
# HELPERS
# -----------------------------
def random_state(size=16):
    # erzeugt z.B. (0,2,4,8)
    return tuple(random.choice([0, 2, 4, 8, 16]) for _ in range(size))


def random_actions():
    # random floats
    return [round(random.uniform(-2, 2), 3) for _ in range(4)]


def random_operations(max_len=6):
    length = random.randint(1, max_len)
    return "".join(random.choice(OPS) for _ in range(length))


# -----------------------------
# MAIN GENERATION
# -----------------------------
def generate_files():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    full_data = []

    # -----------------------------------
    # 1. FULL FILE (10000)
    # -----------------------------------
    for _ in range(NUM_ROWS_FULL):
        state = random_state()
        actions = random_actions()
        full_data.append((state, actions))

    full_path = os.path.join(OUTPUT_DIR, "q_table_full.csv")

    with open(full_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["state"] + ACTIONS)

        for state, actions in full_data:
            state_str = ",".join(map(str, state))
            writer.writerow([state_str] + actions)

    print("Created:", full_path)

    # -----------------------------------
    # 2. HALF FILE (5000)
    # -----------------------------------
    half_data = full_data[:50000]

    half_path = os.path.join(OUTPUT_DIR, "q_table_half.csv")

    with open(half_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["state"] + ACTIONS)

        for state, actions in half_data:
            state_str = ",".join(map(str, state))
            writer.writerow([state_str] + actions)

    print("Created:", half_path)

    # -----------------------------------
    # 3. WITH OPERATIONS
    # -----------------------------------
    ops_data = []

    ops_path = os.path.join(OUTPUT_DIR, "q_table_with_ops.csv")

    with open(ops_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["state"] + ACTIONS + ["operations"])

        for state, actions in half_data:
            op = random_operations()
            ops_data.append((state, actions, op))

            state_str = ",".join(map(str, state))
            writer.writerow([state_str] + actions + [op])

    print("Created:", ops_path)

    # -----------------------------------
    # 4. INDEXED FILE
    # -----------------------------------
    indexed_path = os.path.join(OUTPUT_DIR, "q_table_indexed.csv")

    with open(indexed_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["index"] + ACTIONS + ["operations"])

        for i, (state, actions, op) in enumerate(ops_data):
            writer.writerow([i] + actions + [op])

    print("Created:", indexed_path)

    # -----------------------------------
    # 5. STATE MAPPING (BONUS)
    # -----------------------------------
    mapping_path = os.path.join(OUTPUT_DIR, "state_mapping.csv")

    with open(mapping_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["index", "state", "operations"])

        for i, (state, actions, op) in enumerate(ops_data):
            state_str = ",".join(map(str, state))
            writer.writerow([i, state_str, op])

    print("Created:", mapping_path)

    # -----------------------------------
    # 6. INDEX + OPERATIONS + DIRECTION
    # -----------------------------------
    direction_path = os.path.join(OUTPUT_DIR, "index_operations_direction.csv")

    with open(direction_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["index", "operations", "direction"])

        for i, (state, actions, op) in enumerate(ops_data):
            # random direction (0–3)
            direction = random.randint(0, 3)

            writer.writerow([i, op, direction])

    print("Created:", direction_path)

   # -----------------------------------
    # 7. HALF FILE + BEST DIRECTION
    # -----------------------------------
    direction_half_path = os.path.join(OUTPUT_DIR, "q_table_half_with_direction.csv")

    with open(direction_half_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["state", "up", "down", "left", "right", "direction"])

        for state, actions in half_data:
            # beste Action finden (argmax)
            best_direction = actions.index(max(actions))

            state_str = ",".join(map(str, state))
            writer.writerow([state_str] + actions + [best_direction])

    print("Created:", direction_half_path)

# -----------------------------------
    # 8. INDEX + OPERATIONS (COMPRESSED) + DIRECTION
    # -----------------------------------

    def canonicalize_operations(op_string):
        r_count = 0
        d_count = 0

        i = 0
        while i < len(op_string):
            op = op_string[i]
            i += 1

            num_str = ""
            while i < len(op_string) and op_string[i].isdigit():
                num_str += op_string[i]
                i += 1

            count = int(num_str) if num_str else 1

            if op == "r":
                r_count += count
            elif op == "d":
                d_count += count

        # Rotation modulo 4
        r_count = r_count % 4

        result = []

        if d_count > 0:
            result.append(f"d{d_count}" if d_count > 1 else "d")

        if r_count > 0:
            result.append(f"r{r_count}" if r_count > 1 else "r")

        return "".join(result)

    def compress_operations(op_string):
        if not op_string:
            return op_string

        compressed = []
        current_char = op_string[0]
        count = 1

        for c in op_string[1:]:
            if c == current_char:
                count += 1
            else:
                if count > 1:
                    compressed.append(f"{current_char}{count}")
                else:
                    compressed.append(current_char)
                current_char = c
                count = 1

        # letzten Block hinzufügen
        if count > 1:
            compressed.append(f"{current_char}{count}")
        else:
            compressed.append(current_char)

        return "".join(compressed)


    compressed_path = os.path.join(OUTPUT_DIR, "index_operations_direction_compressed.csv")

    with open(compressed_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["index", "operations", "direction"])

        for i, (state, actions, op) in enumerate(ops_data):
            direction = random.randint(0, 3)

            #compressed_op = compress_operations(op)
            canonicalize_op = canonicalize_operations(op)    

            writer.writerow([i, canonicalize_op, direction])

    print("Created:", compressed_path)


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    generate_files()