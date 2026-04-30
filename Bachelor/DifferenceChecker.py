import os
import pandas as pd

ACTIONS = ["up", "down", "left", "right"]


def prepare_dataframe(path):
    df = pd.read_csv(path, sep=";")
    df.columns = df.columns.str.strip()

    if "state" not in df.columns:
        raise ValueError(f"'state' column not found in {path}")

    existing_action_cols = [col for col in ACTIONS if col in df.columns]
    df = df[["state"] + existing_action_cols].copy()

    df["state"] = df["state"].astype(str).str.strip()

    for col in existing_action_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    duplicate_mask = df["state"].duplicated(keep=False)
    duplicate_count = duplicate_mask.sum()

    if duplicate_count > 0:
        print(f"\n[INFO] {path} enthält {duplicate_count} Zeilen mit doppelten States.")
        print("Beispiele doppelte States:")
        print(df.loc[duplicate_mask, "state"].head(10).tolist())

    df = df.drop_duplicates(subset="state", keep="first")
    df = df.set_index("state")

    return df


def start_difference(filename, filepath="./Data/", grid_size=4, max_depth=5):
    os.makedirs(filepath, exist_ok=True)

    reconstructed_path = f"{filepath}{filename}_reconstructed_{grid_size}_{max_depth}.csv"
    original_path = f"{filepath}{filename}_single_{grid_size}_{max_depth}.csv"

    original_data = prepare_dataframe(original_path)
    reconstructed_data = prepare_dataframe(reconstructed_path)

    original_action_columns = [col for col in ACTIONS if col in original_data.columns]
    reconstructed_action_columns = [col for col in ACTIONS if col in reconstructed_data.columns]
    common_action_columns = [
        col for col in ACTIONS
        if col in original_action_columns and col in reconstructed_action_columns
    ]

    if not common_action_columns:
        raise ValueError("Keine gemeinsamen Action-Spalten gefunden.")

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

    original_common = original_data.loc[common_states, common_action_columns].copy()
    reconstructed_common = reconstructed_data.loc[common_states, common_action_columns].copy()

    original_common = original_common.sort_index()
    reconstructed_common = reconstructed_common.sort_index()

    valid_rows_mask = (
        original_common.notna().all(axis=1) &
        reconstructed_common.notna().all(axis=1)
    )

    original_common = original_common.loc[valid_rows_mask].copy()
    reconstructed_common = reconstructed_common.loc[valid_rows_mask].copy()

    common_valid_states = original_common.index.intersection(reconstructed_common.index)
    original_common = original_common.loc[common_valid_states].sort_index()
    reconstructed_common = reconstructed_common.loc[common_valid_states].sort_index()

    # -----------------------------
    # POLICY AGREEMENT
    # -----------------------------
    if len(original_common) > 0:
        orig_best = original_common.idxmax(axis=1)
        recon_best = reconstructed_common.idxmax(axis=1)

        same_policy = (orig_best.to_numpy() == recon_best.to_numpy()).sum()
        total_policy_compared = len(orig_best)
        policy_agreement_percent = same_policy / total_policy_compared * 100

        print(
            f"\nSame best action: "
            f"{same_policy}/{total_policy_compared} "
            f"({policy_agreement_percent:.2f}%)"
        )
    else:
        orig_best = pd.Series(dtype=object)
        recon_best = pd.Series(dtype=object)
        same_policy = 0
        total_policy_compared = 0
        policy_agreement_percent = None
        print("\nSame best action: Keine vergleichbaren States gefunden.")

    # -----------------------------
    # POLICY MISMATCHES
    # -----------------------------
    print("\n----- POLICY MISMATCH STATES -----")

    mismatch_rows = []

    if len(orig_best) > 0 and len(recon_best) > 0:
        policy_diff_mask = orig_best.to_numpy() != recon_best.to_numpy()
        different_states = orig_best.index[policy_diff_mask]

        print(f"Anzahl unterschiedlicher Policies: {len(different_states)}")

        for state in different_states:
            row = {
                "state": state,
                "original_best": orig_best.loc[state],
                "reconstructed_best": recon_best.loc[state],
            }

            for action in common_action_columns:
                row[f"original_{action}"] = original_common.loc[state, action]
                row[f"reconstructed_{action}"] = reconstructed_common.loc[state, action]
                row[f"abs_diff_{action}"] = abs(
                    original_common.loc[state, action]
                    - reconstructed_common.loc[state, action]
                )

            mismatch_rows.append(row)

        for state in different_states[:10]:
            print("\nState:", state)
            print("Original best:", orig_best.loc[state])
            print("Reconstructed best:", recon_best.loc[state])

            print("Original values:")
            print(original_common.loc[state].to_dict())

            print("Reconstructed values:")
            print(reconstructed_common.loc[state].to_dict())
    else:
        print("Keine vergleichbaren Policy-Unterschiede gefunden.")

    # -----------------------------
    # ACTION ERRORS
    # -----------------------------
    print("\n----- ACTION CHECK -----")
    print("Verglichene Action-Spalten:", common_action_columns)

    action_errors = {}
    total_error_sum = 0.0
    total_value_count = 0

    for action in common_action_columns:
        orig_values = original_common[action]
        recon_values = reconstructed_common[action]

        abs_diff = (orig_values - recon_values).abs()

        if len(abs_diff) > 0:
            avg_error = abs_diff.mean()
            total_error = abs_diff.sum()
            max_error = abs_diff.max()
            count = len(abs_diff)
        else:
            avg_error = None
            total_error = 0.0
            max_error = None
            count = 0

        action_errors[action] = {
            "average_error": avg_error,
            "total_error": total_error,
            "max_error": max_error,
            "count": count,
        }

        total_error_sum += total_error
        total_value_count += count

    overall_average_error = (
        total_error_sum / total_value_count
        if total_value_count > 0
        else None
    )

    print("\n----- RESULTS PER ACTION -----")
    for action, stats in action_errors.items():
        print(
            f"Action {action}: "
            f"Average Error = {stats['average_error']}, "
            f"Total Error = {stats['total_error']}, "
            f"Max Error = {stats['max_error']}, "
            f"Compared Values = {stats['count']}"
        )

    print("\n----- OVERALL RESULT -----")
    print(f"Overall Average Error: {overall_average_error}")
    print(f"Total Error Sum: {total_error_sum}")
    print(f"Total Compared Action Values: {total_value_count}")

    # -----------------------------
    # CSV EXPORT APPEND
    # -----------------------------
    result_prefix = f"{filepath}{filename}_difference_results_{grid_size}_{max_depth}"

    summary_path = f"{result_prefix}_summary.csv"
    action_error_path = f"{result_prefix}_actions.csv"
    mismatch_path = f"{result_prefix}_policy_mismatches.csv"

    summary_df = pd.DataFrame([
        {
            "filename": filename,
            "grid_size": grid_size,
            "max_depth": max_depth,
            "states_original": len(original_data),
            "states_reconstructed": len(reconstructed_data),
            "common_states": len(common_states),
            "valid_compared_states": len(common_valid_states),
            "missing_in_reconstructed": len(missing_in_reconstructed),
            "missing_in_original": len(missing_in_original),
            "same_best_action": same_policy,
            "total_policy_compared": total_policy_compared,
            "policy_agreement_percent": policy_agreement_percent,
            "overall_average_q_error": overall_average_error,
            "total_error_sum": total_error_sum,
            "total_compared_action_values": total_value_count,
        }
    ])

    summary_df.to_csv(
        summary_path,
        sep=";",
        index=False,
        encoding="utf-8",
        mode="a",
        header=not os.path.exists(summary_path),
    )

    action_error_df = pd.DataFrame([
        {
            "filename": filename,
            "grid_size": grid_size,
            "max_depth": max_depth,
            "action": action,
            "average_error": stats["average_error"],
            "total_error": stats["total_error"],
            "max_error": stats.get("max_error"),
            "compared_values": stats["count"],
        }
        for action, stats in action_errors.items()
    ])

    action_error_df.to_csv(
        action_error_path,
        sep=";",
        index=False,
        encoding="utf-8",
        mode="a",
        header=not os.path.exists(action_error_path),
    )

    mismatch_df = pd.DataFrame(mismatch_rows)

    mismatch_df.to_csv(
        mismatch_path,
        sep=";",
        index=False,
        encoding="utf-8",
        mode="a",
        header=not os.path.exists(mismatch_path),
    )

    print("\n----- CSV EXPORT APPEND -----")
    print(f"Summary appended to: {summary_path}")
    print(f"Action errors appended to: {action_error_path}")
    print(f"Policy mismatches appended to: {mismatch_path}")


def check_differences(filename="q_table_basic", filepath="./Data/", grid_size=4, max_depth=5):
    start_difference(
        filename=filename,
        filepath=filepath,
        grid_size=grid_size,
        max_depth=max_depth,
    )


def main():
    check_differences(
        filename="q_table_basic",
        filepath="./Data/",
        grid_size=4,
        max_depth=5,
    )


if __name__ == "__main__":
    main()