import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from pydantic import ValidationError

from .agent import build_agent
from .demo import DemoProvider, demo_parameters
from .providers import SearchAPIProvider
from .report import render_itinerary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plan round-trip flights and hotels with LangGraph."
    )
    parser.add_argument("request", nargs="?", help="Natural-language trip request")
    parser.add_argument(
        "--params", type=Path, help="JSON parameters; the LLM still controls planning"
    )
    parser.add_argument(
        "--demo", action="store_true", help="Use synthetic search data with the real planning LLM"
    )
    parser.add_argument(
        "--format",
        choices=["report", "json"],
        default="report",
        help="Output format (default: a Markdown itinerary report)",
    )
    parser.add_argument(
        "--max-turns", type=int, default=18, help="Maximum model turns (default: 18)"
    )
    parser.add_argument(
        "--max-searches",
        type=int,
        default=4,
        help="Maximum distinct flight/hotel searches (default: 4)",
    )
    args = parser.parse_args()
    if args.request and args.params:
        parser.error("Use either a request or --params, not both.")
    if not args.request and not args.params and not args.demo:
        parser.error("Provide a request, --params, or --demo.")
    load_dotenv()
    provider = None
    try:
        if args.params:
            state = {"parameters": json.loads(args.params.read_text())}
            if not isinstance(state["parameters"], dict):
                raise ValueError("Parameter JSON must be an object.")
        elif args.request:
            state = {"request": args.request}
        else:
            state = {"parameters": demo_parameters()}
        provider = DemoProvider() if args.demo else SearchAPIProvider()
        result = build_agent(
            provider, max_steps=args.max_turns, max_searches=args.max_searches
        ).invoke(state)["result"]
        result.setdefault("source", provider.source)
        print(
            json.dumps(result, indent=2, ensure_ascii=False)
            if args.format == "json"
            else result["report"],
            end="\n" if args.format == "json" else "",
        )
        return 0 if result["status"] == "success" else 2
    except (OSError, ValueError, ValidationError) as exc:
        result = {"status": "error", "message": str(exc)}
        print(json.dumps(result, indent=2) if args.format == "json" else render_itinerary(result))
        return 2
    finally:
        if isinstance(provider, SearchAPIProvider):
            provider.close()


if __name__ == "__main__":
    sys.exit(main())
