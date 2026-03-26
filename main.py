import argparse

from Agents.Agent import run_training
from Agents.AgentUpdated import run_training_updated
from Agents.AgentReworked import run_training_reworked
from Agents.AgentReworked2 import run_training_reworked2
from Agents.AgentNormalizer import run_training_normalizer
from Agents.ValueAgentBasic import run_training_basic
from Agents.PlayerBasic import play
from Bachelor.MatrixOperation import Detector, Operation
from Environment.Environment import Environment

def printOperations(args):
    print(Operation().print_operations())
    pass

def run_play(args):
    play()

def run_train(args):
    env = Environment(size=int(args.grid))
    env.reset()
    if args.agent == "value":
        if args.updated == "False":
            run_training_basic(env=env, episodes=args.episodes, filename=args.file, grid_size=int(args.grid), filepath=args.path, max_depth=args.depth, save_interval=args.save)
        else:
            run_training_normalizer(env=env, episodes=args.episodes, filename=args.file, grid_size=int(args.grid), filepath=args.path, max_depth=args.depth, save_interval=args.save)
    else:
        run_training(env=env,episodes=args.episodes, filename=args.file, grid_size=int(args.grid), filepath=args.path, max_depth=args.depth, save_interval=args.save)

def main():
    parser = argparse.ArgumentParser(description="Matrix Operation Example")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_operations = subparsers.add_parser("operations")
    parser_operations.set_defaults(func=printOperations)

    parser_play = subparsers.add_parser("play")
    parser_play.set_defaults(func=run_play)

    parser_run = subparsers.add_parser("train", help="Training function for the Agent")
    parser_run.add_argument("--agent", "-a",
        choices=["value", "policy"],
        required=True,
        help="Select the agent type"
    )
    parser_run.add_argument("--grid", "-g",
        choices=["4", "3", "2"],
        required=True,
        help="Select grid size"
    )
    parser_run.add_argument("--depth", "-d",
                            type=int, 
                            required=False, 
                            help="Select MaxDepth"
    )
    parser_run.add_argument("--episodes", "-e",
                            type=int, 
                            required=False, 
                            help="Number of episodes to run the training for"
    )
    parser_run.add_argument("-f", "--file", "--File",
                            type=str,
                            required=False,
                            help="Location where the Q-Table should be saved to filename"
    )
    parser_run.add_argument("--path", "-p", "--Path",
                            type=str,
                            required=False,
                            help="Location where the Q-Table should be saved to Path")
    parser_run.add_argument("--save", "-s",
                            type=int,
                            required=False,
                            help="Save Interval"
    )
    parser_run.add_argument("--updated", "-u",
                            choices=["False", "True"],
                            required=False,
                            help="Use Updated Matrix Operation"
    )

    parser_run.set_defaults(func=run_train)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()