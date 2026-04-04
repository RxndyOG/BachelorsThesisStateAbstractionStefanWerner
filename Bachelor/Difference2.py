import pandas as pd

ACTIONS = ["up", "down", "left", "right"]

reconstructed_path = "./Data/q_table_basic_reconstructed_2.csv"
original_path = "./Data/q_table_basic_single_2.csv"

# CSVs korrekt mit ; laden
reconstructed_data = pd.read_csv(reconstructed_path, sep=";")
original_data = pd.read_csv(original_path, sep=";")

# Optional: Leerzeichen in Spaltennamen entfernen
reconstructed_data.columns = reconstructed_data.columns.str.strip()
original_data.columns = original_data.columns.str.strip()

# State als Index
reconstructed_data = reconstructed_data.set_index("state")
original_data = original_data.set_index("state")

# Nur die echten Action-Spalten verwenden, falls später mal extra Spalten dazukommen
original_action_columns = [col for col in ACTIONS if col in original_data.columns]
reconstructed_action_columns = [col for col in ACTIONS if col in reconstructed_data.columns]
common_action_columns = [col for col in ACTIONS if col in original_action_columns and col in reconstructed_action_columns]

# States vergleichen
common_states = original_data.index.intersection(reconstructed_data.index)
missing_in_reconstructed = original_data.index.difference(reconstructed_data.index)
missing_in_original = reconstructed_data.index.difference(original_data.index)

print("----- STATE CHECK -----")
print(f"States in original: {len(original_data)}")
print(f"States in reconstructed: {len(reconstructed_data)}")
print(f"Common states: {len(common_states)}")
print(f"Missing in reconstructed: {len(missing_in_reconstructed)}")
print(f"Missing in original: {len(missing_in_original)}")

if len(missing_in_reconstructed) > 0:
    print("\nBeispiele missing in reconstructed:")
    print(list(missing_in_reconstructed[:10]))

if len(missing_in_original) > 0:
    print("\nBeispiele missing in original:")
    print(list(missing_in_original[:10]))

# Nur gemeinsame States vergleichen
original_common = original_data.loc[common_states, common_action_columns].copy()
reconstructed_common = reconstructed_data.loc[common_states, common_action_columns].copy()

# Alles numerisch machen
for col in common_action_columns:
    original_common[col] = pd.to_numeric(original_common[col], errors="coerce")
    reconstructed_common[col] = pd.to_numeric(reconstructed_common[col], errors="coerce")

# Zeilen entfernen, wo gar keine gültigen Werte vorhanden sind
valid_rows_mask = (
    original_common[common_action_columns].notna().all(axis=1) &
    reconstructed_common[common_action_columns].notna().all(axis=1)
)

original_common = original_common[valid_rows_mask]
reconstructed_common = reconstructed_common[valid_rows_mask]

# Policy vergleichen
if len(original_common) > 0:
    orig_best = original_common.idxmax(axis=1)
    recon_best = reconstructed_common.idxmax(axis=1)

    same_policy = (orig_best == recon_best).sum()
    total = len(orig_best)

    print(f"\nSame best action: {same_policy}/{total} ({same_policy/total*100:.2f}%)")
else:
    orig_best = pd.Series(dtype=object)
    recon_best = pd.Series(dtype=object)
    print("\nSame best action: Keine vergleichbaren States gefunden.")

print("\n----- POLICY MISMATCH STATES -----")

policy_diff_mask = orig_best != recon_best
different_states = orig_best[policy_diff_mask]

print(f"Anzahl unterschiedlicher Policies: {len(different_states)}")

for state in different_states.index[:10]:
    print("\nState:", state)
    print("Original best:", orig_best[state])
    print("Reconstructed best:", recon_best[state])

    print("Original values:")
    print(original_common.loc[state].to_dict())

    print("Reconstructed values:")
    print(reconstructed_common.loc[state].to_dict())

print("\n----- ACTION CHECK -----")
print("Verglichene Action-Spalten:", common_action_columns)

action_errors = {}
total_error_sum = 0.0
total_value_count = 0

for action in common_action_columns:
    orig_values = original_common[action]
    recon_values = reconstructed_common[action]

    valid_mask = orig_values.notna() & recon_values.notna()
    orig_values = orig_values[valid_mask]
    recon_values = recon_values[valid_mask]

    abs_diff = (orig_values - recon_values).abs()

    if len(abs_diff) > 0:
        avg_error = abs_diff.mean()
        total_error = abs_diff.sum()
        count = len(abs_diff)
    else:
        avg_error = None
        total_error = 0.0
        count = 0

    action_errors[action] = {
        "average_error": avg_error,
        "total_error": total_error,
        "count": count
    }

    total_error_sum += total_error
    total_value_count += count

overall_average_error = total_error_sum / total_value_count if total_value_count > 0 else None

print("\n----- RESULTS PER ACTION -----")
for action, stats in action_errors.items():
    print(
        f"Action {action}: "
        f"Average Error = {stats['average_error']}, "
        f"Total Error = {stats['total_error']}, "
        f"Compared Values = {stats['count']}"
    )

print("\n----- OVERALL RESULT -----")
print(f"Overall Average Error: {overall_average_error}")
print(f"Total Error Sum: {total_error_sum}")
print(f"Total Compared Action Values: {total_value_count}")