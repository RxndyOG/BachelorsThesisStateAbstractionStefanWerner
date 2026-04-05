import numpy as np
import csv
import os
from datetime import datetime

from Environment.Environment import Environment
from Agents.Agent import Agent


class PlayTester:
    ACTION_COLUMNS = ["up", "down", "left", "right"]

    def __init__(self, filepath="./Data/", filename="q_table", grid_size=2, max_depth=5):
        self.filepath = filepath
        self.filename = filename
        self.grid_size = grid_size
        self.max_depth = max_depth

        self.single_q_table = None
        self.reconstructed_q_table = None

    def create_agent(self):
        env = Environment(size=self.grid_size)

        agent = Agent(
            environment=env,
            filepath=self.filepath,
            filename=self.filename,
            max_depth=self.max_depth,
            epsilon=0.0,
            epsilon_decay=1.0,
            epsilon_min=0.0,
            learning_rate=0.1,
            size=self.grid_size
        )

        return env, agent

    def create_env(self):
        return Environment(size=self.grid_size)

    def state_to_key(self, state):
        state_array = np.array(state, dtype=int)
        return tuple(state_array.flatten())

    def state_to_string(self, state):
        if state is None:
            return ""
        return ",".join(map(str, np.array(state, dtype=int).flatten()))

    def load_single_q_table_once(self):
        if self.single_q_table is not None:
            return self.single_q_table

        env, agent = self.create_agent()
        agent.load_q_table_single(filepath=self.filepath, filename=self.filename)
        self.single_q_table = agent.q_table

        print("Single Q-Table loaded once.")
        return self.single_q_table

    def load_q_table_reconstructed_once(self):
        if self.reconstructed_q_table is not None:
            return self.reconstructed_q_table

        full_path = os.path.join(
            self.filepath,
            f"{self.filename}_reconstructed_{self.grid_size}.csv"
        )

        if not os.path.exists(full_path):
            print(f"No reconstructed Q-Table found at: {full_path}")
            self.reconstructed_q_table = {}
            return self.reconstructed_q_table

        q_table = {}

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
                q_table[state_key] = action_values

        self.reconstructed_q_table = q_table
        print(f"Reconstructed Q-Table loaded once from: {full_path}")
        return self.reconstructed_q_table

    def choose_action_from_table(self, q_table, state_key, available_actions):
        if not q_table:
            return np.random.choice(available_actions)

        if state_key not in q_table:
            return np.random.choice(available_actions)

        action_values = q_table[state_key]

        q_values = {}
        for action in available_actions:
            idx = self.ACTION_COLUMNS.index(action)
            q_values[action] = float(action_values[idx])

        max_q = max(q_values.values())
        best_actions = [action for action, value in q_values.items() if value == max_q]

        return np.random.choice(best_actions)

    def get_highest_tile(self, state):
        state_array = np.array(state)
        return int(np.max(state_array))

    def play_with_single(self, render=False):
        env = self.create_env()
        q_table = self.load_single_q_table_once()

        return self.play_episode(
            env=env,
            q_table=q_table,
            render=render,
            title="Single Q-Table"
        )

    def play_with_reconstructed(self, render=False):
        env = self.create_env()
        q_table = self.load_q_table_reconstructed_once()

        return self.play_episode(
            env=env,
            q_table=q_table,
            render=render,
            title="Reconstructed Q-Table"
        )

    def play_episode(self, env, q_table, render=False, title="Play"):
        state = env.reset()
        done = False
        total_reward = 0.0
        move_count = 0

        max_reached_tile = self.get_highest_tile(state)
        max_reached_state = np.array(state).copy()

        if render:
            print(f"\n--- {title}: Start ---")
            if hasattr(env, "render"):
                env.render()
            else:
                print(state)

        while not done:
            available_actions = env.find_available_actions()

            if not available_actions:
                break

            state_key = self.state_to_key(state)
            action = self.choose_action_from_table(q_table, state_key, available_actions)

            next_state, reward, done, info = env.step(action)

            total_reward += reward
            move_count += 1
            state = next_state

            current_highest_tile = self.get_highest_tile(state)
            if current_highest_tile > max_reached_tile:
                max_reached_tile = current_highest_tile
                max_reached_state = np.array(state).copy()

            if render:
                print(
                    f"Move {move_count:03d} | "
                    f"Action: {action:<5} | "
                    f"Reward: {reward:<6} | "
                    f"Total: {total_reward}"
                )

                if hasattr(env, "render"):
                    env.render()
                else:
                    print(state)

        highest_tile = self.get_highest_tile(state)

        if render:
            print(f"\n--- {title}: Finished ---")
            print(f"Moves: {move_count}")
            print(f"Total Reward: {total_reward}")
            print(f"Highest Tile: {highest_tile}")
            print(f"Max Reached Tile: {max_reached_tile}")
            print("Max Reached State:")
            print(max_reached_state)
            print("Final State:")
            print(state)

        return {
            "title": title,
            "moves": move_count,
            "total_reward": total_reward,
            "highest_tile": highest_tile,
            "final_state": state,
            "max_reached_tile": max_reached_tile,
            "max_reached_state": max_reached_state
        }

    def run_multiple_single(self, runs=100, render_every=0):
        results = []

        print("\n" + "=" * 60)
        print(f"RUNNING SINGLE Q-TABLE FOR {runs} GAMES")
        print("=" * 60)

        self.load_single_q_table_once()

        for i in range(runs):
            render = render_every > 0 and (i + 1) % render_every == 0
            result = self.play_with_single(render=render)
            results.append(result)

            if (i + 1) % 10 == 0 or i == runs - 1:
                print(f"Single: finished {i + 1}/{runs}")

        return self.calculate_statistics(results, label="Single")

    def run_multiple_reconstructed(self, runs=100, render_every=0):
        results = []

        print("\n" + "=" * 60)
        print(f"RUNNING RECONSTRUCTED Q-TABLE FOR {runs} GAMES")
        print("=" * 60)

        self.load_q_table_reconstructed_once()

        for i in range(runs):
            render = render_every > 0 and (i + 1) % render_every == 0
            result = self.play_with_reconstructed(render=render)
            results.append(result)

            if (i + 1) % 10 == 0 or i == runs - 1:
                print(f"Reconstructed: finished {i + 1}/{runs}")

        return self.calculate_statistics(results, label="Reconstructed")

    def calculate_statistics(self, results, label="Stats"):
        rewards = [r["total_reward"] for r in results]
        moves = [r["moves"] for r in results]
        highest_tiles = [r["highest_tile"] for r in results]
        max_reached_tiles = [r["max_reached_tile"] for r in results]

        best_run = None
        if results:
            best_run = max(results, key=lambda r: r["max_reached_tile"])

        stats = {
            "label": label,
            "games": len(results),
            "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
            "min_reward": float(np.min(rewards)) if rewards else 0.0,
            "max_reward": float(np.max(rewards)) if rewards else 0.0,
            "avg_moves": float(np.mean(moves)) if moves else 0.0,
            "min_moves": int(np.min(moves)) if moves else 0,
            "max_moves": int(np.max(moves)) if moves else 0,
            "avg_highest_tile": float(np.mean(highest_tiles)) if highest_tiles else 0.0,
            "max_highest_tile": int(np.max(highest_tiles)) if highest_tiles else 0,
            "avg_max_reached_tile": float(np.mean(max_reached_tiles)) if max_reached_tiles else 0.0,
            "max_reached_tile": int(np.max(max_reached_tiles)) if max_reached_tiles else 0,
            "max_reached_state": np.array(best_run["max_reached_state"]).copy() if best_run else None,
            "reached_8": sum(1 for t in highest_tiles if t >= 8),
            "reached_16": sum(1 for t in highest_tiles if t >= 16),
            "reached_32": sum(1 for t in highest_tiles if t >= 32),
            "reached_64": sum(1 for t in highest_tiles if t >= 64),
            "reached_128": sum(1 for t in highest_tiles if t >= 128),
            "reached_256": sum(1 for t in highest_tiles if t >= 256),
            "reached_512": sum(1 for t in highest_tiles if t >= 512),
        }

        return stats

    def print_statistics(self, stats):
        print("\n" + "-" * 60)
        print(f"STATISTICS: {stats['label']}")
        print("-" * 60)
        print(f"Games played:              {stats['games']}")
        print(f"Average reward:            {stats['avg_reward']:.4f}")
        print(f"Min reward:                {stats['min_reward']:.4f}")
        print(f"Max reward:                {stats['max_reward']:.4f}")
        print(f"Average moves:             {stats['avg_moves']:.4f}")
        print(f"Min moves:                 {stats['min_moves']}")
        print(f"Max moves:                 {stats['max_moves']}")
        print(f"Average highest tile:      {stats['avg_highest_tile']:.4f}")
        print(f"Max highest tile:          {stats['max_highest_tile']}")
        print(f"Average max reached tile:  {stats['avg_max_reached_tile']:.4f}")
        print(f"Max reached tile:          {stats['max_reached_tile']}")
        print(f"Reached tile 8:            {stats['reached_8']} times")
        print(f"Reached tile 16:           {stats['reached_16']} times")
        print(f"Reached tile 32:           {stats['reached_32']} times")
        print(f"Reached tile 64:           {stats['reached_64']} times")
        print(f"Reached tile 128:          {stats['reached_128']} times")
        print(f"Reached tile 256:          {stats['reached_256']} times")
        print(f"Reached tile 512:          {stats['reached_512']} times")

        print("Max reached state:")
        if stats["max_reached_state"] is not None:
            print(stats["max_reached_state"])
        else:
            print("None")

    def determine_better_model(self, single_stats, reconstructed_stats):
        # Hauptkriterium: avg_reward
        # Fallback: avg_highest_tile
        # Fallback: avg_moves
        if reconstructed_stats["avg_reward"] > single_stats["avg_reward"]:
            return "Reconstructed"
        if reconstructed_stats["avg_reward"] < single_stats["avg_reward"]:
            return "Single"

        if reconstructed_stats["avg_highest_tile"] > single_stats["avg_highest_tile"]:
            return "Reconstructed"
        if reconstructed_stats["avg_highest_tile"] < single_stats["avg_highest_tile"]:
            return "Single"

        if reconstructed_stats["avg_moves"] > single_stats["avg_moves"]:
            return "Reconstructed"
        if reconstructed_stats["avg_moves"] < single_stats["avg_moves"]:
            return "Single"

        return "Equal"

    def append_comparison_to_csv(self, single_stats, reconstructed_stats, csv_name="playtest_results.csv"):
        os.makedirs(self.filepath, exist_ok=True)
        full_path = os.path.join(self.filepath, csv_name)

        better_model = self.determine_better_model(single_stats, reconstructed_stats)

        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filename": self.filename,
            "grid_size": self.grid_size,
            "max_depth": self.max_depth,
            "episodes_played": single_stats["games"],

            "single_avg_reward": single_stats["avg_reward"],
            "single_min_reward": single_stats["min_reward"],
            "single_max_reward": single_stats["max_reward"],
            "single_avg_moves": single_stats["avg_moves"],
            "single_avg_highest_tile": single_stats["avg_highest_tile"],
            "single_max_highest_tile": single_stats["max_highest_tile"],
            "single_avg_max_reached_tile": single_stats["avg_max_reached_tile"],
            "single_max_reached_tile": single_stats["max_reached_tile"],
            "single_max_reached_state": self.state_to_string(single_stats["max_reached_state"]),

            "reconstructed_avg_reward": reconstructed_stats["avg_reward"],
            "reconstructed_min_reward": reconstructed_stats["min_reward"],
            "reconstructed_max_reward": reconstructed_stats["max_reward"],
            "reconstructed_avg_moves": reconstructed_stats["avg_moves"],
            "reconstructed_avg_highest_tile": reconstructed_stats["avg_highest_tile"],
            "reconstructed_max_highest_tile": reconstructed_stats["max_highest_tile"],
            "reconstructed_avg_max_reached_tile": reconstructed_stats["avg_max_reached_tile"],
            "reconstructed_max_reached_tile": reconstructed_stats["max_reached_tile"],
            "reconstructed_max_reached_state": self.state_to_string(reconstructed_stats["max_reached_state"]),

            "reward_diff_reconstructed_minus_single": reconstructed_stats["avg_reward"] - single_stats["avg_reward"],
            "moves_diff_reconstructed_minus_single": reconstructed_stats["avg_moves"] - single_stats["avg_moves"],
            "highest_tile_diff_reconstructed_minus_single": reconstructed_stats["avg_highest_tile"] - single_stats["avg_highest_tile"],

            "better_model": better_model
        }

        fieldnames = list(row.keys())
        file_exists = os.path.exists(full_path)

        with open(full_path, "a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter=";")

            if not file_exists:
                writer.writeheader()

            writer.writerow(row)

        print(f"\nComparison appended to CSV: {full_path}")

    def compare_both_multiple(self, runs=100, render_every=0, save_csv=True, csv_name="playtest_results.csv"):
        single_stats = self.run_multiple_single(runs=runs, render_every=render_every)
        reconstructed_stats = self.run_multiple_reconstructed(runs=runs, render_every=render_every)

        self.print_statistics(single_stats)
        self.print_statistics(reconstructed_stats)

        print("\n" + "#" * 60)
        print("DIRECT COMPARISON")
        print("#" * 60)

        reward_diff = reconstructed_stats["avg_reward"] - single_stats["avg_reward"]
        moves_diff = reconstructed_stats["avg_moves"] - single_stats["avg_moves"]
        tile_diff = reconstructed_stats["avg_highest_tile"] - single_stats["avg_highest_tile"]
        better_model = self.determine_better_model(single_stats, reconstructed_stats)

        print(f"Episodes played:                 {runs}")
        print(f"Average reward difference:       {reward_diff:.4f}")
        print(f"Average moves difference:        {moves_diff:.4f}")
        print(f"Average highest tile difference: {tile_diff:.4f}")
        print(f"Better model:                    {better_model}")

        if save_csv:
            self.append_comparison_to_csv(
                single_stats=single_stats,
                reconstructed_stats=reconstructed_stats,
                csv_name=csv_name
            )

        return single_stats, reconstructed_stats


def play(filepath="./Data/", filename="q_table", grid_size=2, max_depth=5, runs=1000, render_every=0, save_csv=True):
    tester = PlayTester(
        filepath=filepath,
        filename=filename,
        grid_size=grid_size,
        max_depth=max_depth
    )

    tester.compare_both_multiple(runs=runs, render_every=render_every, save_csv=save_csv)


def main():
    tester = PlayTester(
        filepath="./Data/",
        filename="q_table_basic",
        grid_size=2,
        max_depth=5
    )

    tester.compare_both_multiple(runs=1000, render_every=0, save_csv=True)


if __name__ == "__main__":
    main()