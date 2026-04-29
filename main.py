import argparse

from Agents.Agent import run_training
from Agents.Agent_Player import play
from Bachelor.DifferenceChecker import check_differences
from Environment.Environment import Environment, user_play

def user_play_wrapper(args):
    user_play(size=int(args.grid_size))

def difference(args):
    check_differences(filename=args.filename, filepath=args.filepath, grid_size=int(args.grid_size))

def run_play(args):
    play(filename=args.filename, filepath=args.filepath, grid_size=int(args.grid_size),max_depth=int(args.max_depth), runs=1000, render_every=0, save_csv=True)

def run_train(args):
    env = Environment(size=int(args.grid))
    env.reset()
    if args.agent == "value":
        if args.updated == "False":
            run_training(env=env, episodes=args.episodes, filename=args.file, grid_size=int(args.grid), filepath=args.path, max_depth=args.depth, save_interval=args.save)
        else:
            pass
    else:
        pass

def main():
    parser = argparse.ArgumentParser(description="Matrix Operation Example")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_check = subparsers.add_parser("check")
    parser_check.set_defaults(func=difference)
    parser_check.add_argument("--filename", "-f",
                            type=str,
                            required=False,
                            help="Location where the Q-Table should be loaded from filename")
    parser_check.add_argument("--filepath", "-p",
                            type=str,
                            required=False,
                            help="Location where the Q-Table should be loaded from Path")
    parser_check.add_argument("--grid_size", "-g",
                            choices=["4", "3", "2"],
                            required=False,
                            help="Select grid size")
    parser_check.add_argument("--max_depth", "-d",
                            type=int, 
                            required=False, 
                            help="Select MaxDepth")
    
    parser_user_play = subparsers.add_parser("user_play")
    parser_user_play.set_defaults(func=user_play_wrapper)
    parser_user_play.add_argument("--grid_size", "-g",
                            choices=["4", "3", "2"],
                            required=False,
                            help="Select grid size")

    parser_play = subparsers.add_parser("play")
    parser_play.set_defaults(func=run_play)
    parser_play.add_argument("--filepath", "-p",
                            type=str,
                            required=False,
                            help="Location where the Q-Table should be loaded from Path")
    parser_play.add_argument("--filename", "-f",
                            type=str,
                            required=False,
                            help="Location where the Q-Table should be loaded from filename")
    parser_play.add_argument("--grid_size", "-g",
                            choices=["4", "3", "2"],
                            required=False,
                            help="Select grid size")
    parser_play.add_argument("--max_depth", "-d",
                            type=int, 
                            required=False, 
                            help="Select MaxDepth")
    parser_play.add_argument("--render_every", "-r",
                            type=int, 
                            required=False, 
                            help="Render every n episodes")
    parser_play.add_argument("--save_csv", "-s",
                            choices=["False", "True"],
                            required=False,
                            help="Save results to CSV")
    parser_play.add_argument("--runs", "-e",
                            type=int, 
                            required=False, 
                            help="Number of runs to compare")
    
    parser_run = subparsers.add_parser("train", help="Training function for the Agent")
    parser_run.add_argument("--agent", "-a",
        choices=["value", "policy"],
        required=True,
        help="Select the agent type")
    parser_run.add_argument("--grid", "-g",
        choices=["4", "3", "2"],
        required=True,
        help="Select grid size")
    parser_run.add_argument("--depth", "-d",
                            type=int, 
                            required=False, 
                            help="Select MaxDepth")
    parser_run.add_argument("--episodes", "-e",
                            type=int, 
                            required=False, 
                            help="Number of episodes to run the training for")
    parser_run.add_argument("-f", "--file", "--File",
                            type=str,
                            required=False,
                            help="Location where the Q-Table should be saved to filename")
    parser_run.add_argument("--path", "-p", "--Path",
                            type=str,
                            required=False,
                            help="Location where the Q-Table should be saved to Path")
    parser_run.add_argument("--save", "-s",
                            type=int,
                            required=False,
                            help="Save Interval")
    parser_run.add_argument("--updated", "-u",
                            choices=["False", "True"],
                            required=False,
                            help="Use Updated Matrix Operation")

    parser_run.set_defaults(func=run_train)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()