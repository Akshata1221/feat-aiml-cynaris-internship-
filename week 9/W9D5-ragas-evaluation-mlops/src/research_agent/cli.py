import argparse
from pathlib import Path
from .pipeline import run_baseline, save_report
from .tracking import log_run

def main():
    parser = argparse.ArgumentParser(description="Generate a multi-agent research report.")
    parser.add_argument("--topic", required=True, help="Research topic")
    parser.add_argument("--mode", choices=["offline", "web", "crew", "graph"], default="offline")
    parser.add_argument("--output", default="outputs/research_report.md")
    args = parser.parse_args()

    if args.mode == "crew":
        from .crew import run_crew
        content = run_crew(args.topic)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(content, encoding="utf-8")
    elif args.mode == "graph":
        from .graph import run_graph
        content = run_graph(args.topic)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(content, encoding="utf-8")
    else:
        report = run_baseline(args.topic, use_web=(args.mode == "web"))
        save_report(report, args.output)
        content = report.to_markdown()

    log_run(args.topic, args.mode, args.output)
    print(content)
    print(f"\nReport saved to: {args.output}")

if __name__ == "__main__":
    main()
