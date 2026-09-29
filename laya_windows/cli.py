"""Run Laya typed decisions on Windows (PyTorch, CPU or CUDA)."""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="HF repo id or local checkpoint directory")
    parser.add_argument("--state", help="State text (or use --state-file / stdin '-')")
    parser.add_argument("--state-file", help="File whose contents are the state")
    parser.add_argument("--questions", required=True,
                        help="JSON file of question definitions, or an inline JSON string")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Default: cuda if available")
    parser.add_argument("--offline", action="store_true", help="Use cached weights only")
    args = parser.parse_args()

    if args.state_file:
        state = Path(args.state_file).read_text(encoding="utf-8")
    elif args.state == "-" or args.state is None:
        state = sys.stdin.read()
    else:
        state = args.state
    q = args.questions
    questions = json.loads(Path(q).read_text(encoding="utf-8") if Path(q).is_file() else q)

    from .agent import DEFAULT_MODEL, load

    agent = load(args.model or DEFAULT_MODEL, device=args.device, local_files_only=args.offline)
    result = agent.predict(state, questions)
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
