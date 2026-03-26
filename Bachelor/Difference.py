import pandas as pd

reconstructed_data = pd.read_csv("./Data/q_table_basic_reconstructed.csv", sep=",")
original_data = pd.read_csv("./Data/q_table_basic_single.csv", sep=",")

reconstructed_data = reconstructed_data.set_index("state")
original_data = original_data.set_index("state")


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

original_common = original_data.loc[common_states].copy()
reconstructed_common = reconstructed_data.loc[common_states].copy()

orig_best = original_common.idxmax(axis=1)
recon_best = reconstructed_common.idxmax(axis=1)

same_policy = (orig_best == recon_best).sum()
total = len(orig_best)

print(f"Same best action: {same_policy}/{total} ({same_policy/total*100:.2f}%)")

common_action_columns = [col for col in original_common.columns if col in reconstructed_common.columns]

print("\n----- POLICY MISMATCH STATES -----")

# Boolean Maske für unterschiedliche Policies
policy_diff_mask = orig_best != recon_best

# Alle betroffenen States
different_states = orig_best[policy_diff_mask]

print(f"Anzahl unterschiedlicher Policies: {len(different_states)}")

# Durchgehen und Details ausgeben
for state in different_states.index[:10]:  # nur erste 20 anzeigen (sonst spam)
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

    orig_values = pd.to_numeric(original_common[action], errors="coerce")
    recon_values = pd.to_numeric(reconstructed_common[action], errors="coerce")

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