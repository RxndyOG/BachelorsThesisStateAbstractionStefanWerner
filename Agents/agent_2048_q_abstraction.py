import math
import random
from collections import defaultdict

ACTIONS = ["up", "down", "left", "right"]


class Game2048Env:
    def __init__(self, size=4, seed=42):
        self.size = size
        self.rng = random.Random(seed)
        self.board = None
        self.score = 0
        self.reset()

    def clone_board(self):
        return [row[:] for row in self.board]

    def reset(self):
        self.board = [[0] * self.size for _ in range(self.size)]
        self.score = 0
        self._spawn_tile()
        self._spawn_tile()
        return self.clone_board()

    def _empty_cells(self):
        return [(r, c) for r in range(self.size) for c in range(self.size) if self.board[r][c] == 0]

    def _spawn_tile(self):
        empties = self._empty_cells()
        if not empties:
            return False
        r, c = self.rng.choice(empties)
        self.board[r][c] = 4 if self.rng.random() < 0.1 else 2
        return True

    @staticmethod
    def _compress_line(line):
        vals = [x for x in line if x != 0]
        reward = 0
        out = []
        i = 0
        while i < len(vals):
            if i + 1 < len(vals) and vals[i] == vals[i + 1]:
                merged = vals[i] * 2
                out.append(merged)
                reward += merged
                i += 2
            else:
                out.append(vals[i])
                i += 1
        out += [0] * (len(line) - len(out))
        changed = out != line
        return out, reward, changed

    def _move_left(self):
        new_board = []
        total_reward = 0
        changed_any = False
        for row in self.board:
            new_row, reward, changed = self._compress_line(row)
            new_board.append(new_row)
            total_reward += reward
            changed_any = changed_any or changed
        self.board = new_board
        return changed_any, total_reward

    @staticmethod
    def _reverse_rows(board):
        return [list(reversed(row)) for row in board]

    @staticmethod
    def _transpose(board):
        return [list(x) for x in zip(*board)]

    def _apply_action_to_board(self, board, action):
        original = [row[:] for row in board]

        if action == 0:  # up
            self.board = self._transpose(original)
            changed, reward = self._move_left()
            new_board = self._transpose(self.board)
        elif action == 1:  # down
            self.board = self._transpose(original)
            self.board = self._reverse_rows(self.board)
            changed, reward = self._move_left()
            new_board = self._reverse_rows(self.board)
            new_board = self._transpose(new_board)
        elif action == 2:  # left
            self.board = [row[:] for row in original]
            changed, reward = self._move_left()
            new_board = self.board
        elif action == 3:  # right
            self.board = self._reverse_rows(original)
            changed, reward = self._move_left()
            new_board = self._reverse_rows(self.board)
        else:
            raise ValueError("Unknown action")

        self.board = [row[:] for row in original]
        return new_board, reward, changed

    def legal_actions(self):
        snapshot = self.clone_board()
        legal = []
        for a in range(4):
            new_board, _, changed = self._apply_action_to_board(snapshot, a)
            if changed and new_board != snapshot:
                legal.append(a)
        self.board = snapshot
        return legal

    def is_done(self):
        if self._empty_cells():
            return False
        snapshot = self.clone_board()
        for a in range(4):
            _, _, changed = self._apply_action_to_board(snapshot, a)
            if changed:
                self.board = snapshot
                return False
        self.board = snapshot
        return True

    def step(self, action):
        snapshot = self.clone_board()
        new_board, reward, changed = self._apply_action_to_board(snapshot, action)
        self.board = [row[:] for row in new_board]
        if changed:
            self.score += reward
            self._spawn_tile()
        else:
            reward = -2
        done = self.is_done()
        return self.clone_board(), reward, done, {"changed": changed}


def board_to_raw_key(board):
    return "|".join(",".join(map(str, row)) for row in board)


def board_to_exponents(board):
    exps = []
    for row in board:
        exp_row = []
        for v in row:
            exp_row.append(0 if v == 0 else int(math.log2(v)))
        exps.append(exp_row)
    return exps


def bucket_max_exp(max_exp):
    if max_exp <= 4:
        return "early"
    if max_exp <= 7:
        return "mid"
    return "late"


def bucket_empty(num_empty):
    if num_empty >= 9:
        return "many_empty"
    if num_empty >= 5:
        return "medium_empty"
    return "few_empty"


def board_to_abstract_key(board):
    exps = board_to_exponents(board)
    nonzero = [v for row in exps for v in row if v > 0]
    if not nonzero:
        return "shape=((0,0,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0));level=empty;space=many_empty"

    min_nonzero = min(nonzero)
    max_exp = max(nonzero)
    num_empty = sum(1 for row in board for v in row if v == 0)

    normalized = []
    for row in exps:
        out_row = []
        for v in row:
            if v == 0:
                out_row.append(0)
            else:
                out_row.append(v - min_nonzero + 1)
        normalized.append(tuple(out_row))

    shape = tuple(normalized)
    return f"shape={shape};level={bucket_max_exp(max_exp)};space={bucket_empty(num_empty)}"


class QLearningAgent:
    def __init__(self, state_encoder, alpha=0.12, gamma=0.97, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.996):
        self.encode = state_encoder
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    def select_action(self, board, legal_actions, rng):
        state = self.encode(board)
        if rng.random() < self.epsilon:
            return rng.choice(legal_actions)
        qvals = self.q[state]
        return max(legal_actions, key=lambda a: qvals[a])

    def update(self, state_board, action, reward, next_board, done):
        s = self.encode(state_board)
        ns = self.encode(next_board)
        next_max = 0.0 if done else max(self.q[ns])
        td_target = reward + self.gamma * next_max
        self.q[s][action] += self.alpha * (td_target - self.q[s][action])

    def decay(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train(agent, episodes=2000, max_steps=300, seed=123):
    env = Game2048Env(seed=seed)
    rng = random.Random(seed)
    for _ in range(episodes):
        board = env.reset()
        for _ in range(max_steps):
            legal = env.legal_actions()
            if not legal:
                break
            action = agent.select_action(board, legal, rng)
            next_board, reward, done, _ = env.step(action)
            agent.update(board, action, reward, next_board, done)
            board = next_board
            if done:
                break
        agent.decay()
    return agent


if __name__ == "__main__":
    raw_agent = train(QLearningAgent(board_to_raw_key))
    abstract_agent = train(QLearningAgent(board_to_abstract_key))
    print("Raw states:", len(raw_agent.q))
    print("Abstract states:", len(abstract_agent.q))
