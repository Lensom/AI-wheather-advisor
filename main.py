"""AI Weather Advisor entry point.

    python main.py                      classic flow: city -> structured advice
    python main.py --ask                interactive questions (Gemini calls get_weather)
    python main.py --ask "question"     one question, then exit
"""

import argparse

from cli import ask_main, main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Weather Advisor")
    parser.add_argument(
        "--ask",
        nargs="?",
        const="",
        metavar="QUESTION",
        help="free-form question mode with tool calling; without a value: interactive",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(main() if args.ask is None else ask_main(args.ask))
