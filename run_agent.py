#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from app.agent.langgraph_agent import analyze_fundus_case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LangGraph fundus agent with Ollama tools.")
    parser.add_argument("--image", required=True, help="Path to fundus image.")
    parser.add_argument("--query", required=True, help="Clinical question.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze_fundus_case(query=args.query, image_path=args.image)
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
