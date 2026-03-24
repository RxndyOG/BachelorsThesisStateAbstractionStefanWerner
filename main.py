import argparse

from Bachelor.MatrixOperation import Detector, Operation

def printOperations(args):
    print(Operation().print_operations())
    pass

def run_game(args):
    print("Spiel starten mit folgenden Einstellungen:")
    print("Agent:", args.agent)
    print("Grid:", args.grid)
    print("End Value:", args.endValue)
    print("Greedy:", args.greedy)
    print("File:", args.file)

def main():
    parser = argparse.ArgumentParser(description="Matrix Operation Example")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # operations
    parser_operations = subparsers.add_parser("operations")
    parser_operations.set_defaults(func=printOperations)

    # play / train / run
    parser_run = subparsers.add_parser("train", help="Training function for the Agent")
    parser_run.add_argument(
        "--agent", "-a",
        choices=["value", "policy"],
        required=True,
        help="Select the agent type"
    )
    parser_run.add_argument(
        "--grid", "-g",
        choices=["4", "3", "2"],
        required=True,
        help="Select grid size"
    )
    parser_run.add_argument("--endValue", "-e",
                            type=int, 
                            required=False, 
                            help="Select possible end value where programm should complete"
    )
    parser_run.add_argument("--greedy", "-r",
                            type=bool, 
                            required=False, 
                            help="determens the greedy levels of the agent"
    )
    parser_run.add_argument("-f", "--file", "--File",
                            type=str,
                            required=False,
                            help="Location where the Q-Table should be saved to"
    )

    parser_run.set_defaults(func=run_game)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()