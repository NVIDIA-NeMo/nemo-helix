#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

BENCHMARKS_DIR = Path(__file__).parent.parent / "benchmarks"
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def get_git_metadata() -> dict:
    def run_git(args: list[str]) -> str:
        result = subprocess.run(
            ["git"] + args,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    return {
        "commit": run_git(["rev-parse", "HEAD"]),
        "commit_short": run_git(["rev-parse", "--short", "HEAD"]),
        "branch": run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
    }


def get_system_metadata() -> dict:
    metadata = {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
    }

    if platform.system() == "Darwin":
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            metadata["cpu"] = result.stdout.strip()
    elif platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        metadata["cpu"] = line.split(":")[1].strip()
                        break
        except FileNotFoundError:
            pass

    return metadata


def run_benchmark(output_name: str | None = None) -> None:
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)

    if output_name:
        if not output_name.endswith(".json"):
            output_name += ".json"
        output_file = BENCHMARKS_DIR / output_name
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file = BENCHMARKS_DIR / f"{timestamp}.json"

    print("Running CLI benchmarks...")
    print(f"Results will be saved to: {output_file}")

    nhx_bin = PROJECT_ROOT / ".venv" / "bin" / "nhx"

    result = subprocess.run(
        [
            "hyperfine",
            "--warmup",
            "3",
            "--min-runs",
            "10",
            "--export-json",
            str(output_file),
            "--command-name",
            "nhx --help",
            f"{nhx_bin} --help",
        ],
        check=False,
    )

    if result.returncode != 0:
        print("Benchmark failed!")
        sys.exit(1)

    with open(output_file) as f:
        data = json.load(f)

    data["metadata"] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **get_git_metadata(),
        **get_system_metadata(),
    }

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    print()
    print(f"Benchmark complete. Results saved to: {output_file}")
    print()
    print("To compare with previous runs:")
    print("  uv run python packages/nemo_helix_ext/scripts/benchmark.py list")


def load_benchmark(filepath: Path) -> dict:
    with open(filepath) as f:
        return json.load(f)


def format_time(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    return f"{seconds:.3f}s"


def format_memory(bytes_val: float) -> str:
    if bytes_val == 0:
        return ""
    mb = bytes_val / (1024 * 1024)
    return f"{mb:.1f}MB"


def format_change(current: float, previous: float) -> str:
    if previous == 0:
        return ""
    change_pct = ((current - previous) / previous) * 100
    if change_pct < -5:
        return f"\033[32m{change_pct:+.1f}%\033[0m"
    elif change_pct > 5:
        return f"\033[31m{change_pct:+.1f}%\033[0m"
    return f"{change_pct:+.1f}%"


def format_date(timestamp: str) -> str:
    if not timestamp:
        return ""
    try:
        dt = datetime.fromisoformat(timestamp)
        return dt.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return ""


def truncate(s: str, length: int) -> str:
    if len(s) <= length:
        return s
    return s[: length - 1] + "…"


def print_benchmark_row(
    name: str,
    data: dict,
    baseline_mean: float | None = None,
) -> float | None:
    metadata = data.get("metadata", {})
    commit = metadata.get("commit_short", "")[:10]
    branch = truncate(metadata.get("branch", ""), 15)
    date = format_date(metadata.get("timestamp", ""))

    for result in data.get("results", []):
        mean = result.get("mean", 0)
        stddev = result.get("stddev", 0)
        min_time = result.get("min", 0)
        max_time = result.get("max", 0)
        memory = result.get("memory_usage_byte", [])
        max_memory = max(memory) if memory else 0

        change = format_change(mean, baseline_mean) if baseline_mean else ""

        print(
            f"{name:<12} {date:<18} {commit:<12} {branch:<17} "
            f"{format_time(mean):<9} {format_time(stddev):<9} "
            f"{format_time(min_time):<9} {format_time(max_time):<9} "
            f"{format_memory(max_memory):<8} {change:<10}"
        )
        return mean
    return None


def list_benchmarks(limit: int | None = None) -> None:
    if not BENCHMARKS_DIR.exists():
        print(f"No benchmarks directory found at {BENCHMARKS_DIR}")
        return

    all_files = list(BENCHMARKS_DIR.glob("*.json"))
    if not all_files:
        print("No benchmark results found.")
        return

    baseline_file = BENCHMARKS_DIR / "baseline.json"
    other_files = sorted([f for f in all_files if f.name != "baseline.json"])

    if limit:
        other_files = other_files[-limit:]

    print(
        f"\n{'Name':<12} {'Date':<18} {'Commit':<12} {'Branch':<17} {'Mean':<9} {'StdDev':<9} {'Min':<9} {'Max':<9} {'Memory':<8} {'vs base':<10}"
    )
    print("-" * 128)

    baseline_mean = None
    if baseline_file.exists():
        try:
            data = load_benchmark(baseline_file)
            baseline_mean = print_benchmark_row("baseline", data)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error reading baseline: {e}")

    for filepath in other_files:
        try:
            data = load_benchmark(filepath)
            name = filepath.stem
            if name.startswith("20") and "_" in name:
                name = name.split("_", 1)[1]
            print_benchmark_row(name, data, baseline_mean)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error reading {filepath}: {e}")

    print()


def compare_two(file1: str, file2: str) -> None:
    path1 = BENCHMARKS_DIR / file1 if not Path(file1).is_absolute() else Path(file1)
    path2 = BENCHMARKS_DIR / file2 if not Path(file2).is_absolute() else Path(file2)

    if not path1.suffix:
        path1 = path1.with_suffix(".json")
    if not path2.suffix:
        path2 = path2.with_suffix(".json")

    if not path1.exists():
        print(f"File not found: {path1}")
        return
    if not path2.exists():
        print(f"File not found: {path2}")
        return

    data1 = load_benchmark(path1)
    data2 = load_benchmark(path2)

    print("\nComparing:")
    print(f"  Baseline: {path1.stem}")
    print(f"  Current:  {path2.stem}")
    print()

    results1 = {r["command"]: r for r in data1.get("results", [])}
    results2 = {r["command"]: r for r in data2.get("results", [])}

    all_commands = set(results1.keys()) | set(results2.keys())

    print(
        f"{'Command':<15} {'Base Time':<12} {'Curr Time':<12} {'Change':<12} {'Base Mem':<10} {'Curr Mem':<10} {'Status'}"
    )
    print("-" * 95)

    for cmd in sorted(all_commands):
        r1 = results1.get(cmd, {})
        r2 = results2.get(cmd, {})

        mean1 = r1.get("mean", 0)
        mean2 = r2.get("mean", 0)
        mem1 = r1.get("memory_usage_byte", [])
        mem2 = r2.get("memory_usage_byte", [])
        max_mem1 = max(mem1) if mem1 else 0
        max_mem2 = max(mem2) if mem2 else 0

        if mean1 and mean2:
            change_pct = ((mean2 - mean1) / mean1) * 100
            if change_pct < -5:
                status = "\033[32mFASTER\033[0m"
            elif change_pct > 5:
                status = "\033[31mSLOWER\033[0m"
            else:
                status = "~same"
            change_str = f"{change_pct:+.1f}%"
        else:
            change_str = "N/A"
            status = ""

        print(
            f"{cmd[:14]:<15} {format_time(mean1):<12} {format_time(mean2):<12} "
            f"{change_str:<12} {format_memory(max_mem1):<10} {format_memory(max_mem2):<10} {status}"
        )

    print()


def show_latest() -> None:
    if not BENCHMARKS_DIR.exists():
        print(f"No benchmarks directory found at {BENCHMARKS_DIR}")
        return

    files = sorted(BENCHMARKS_DIR.glob("*.json"), reverse=True)
    if not files:
        print("No benchmark results found.")
        return

    latest = files[0]
    data = load_benchmark(latest)

    print(f"\nLatest benchmark: {latest.stem}")
    print("-" * 50)

    for result in data.get("results", []):
        command = result.get("command", "unknown")
        mean = result.get("mean", 0)
        stddev = result.get("stddev", 0)
        min_time = result.get("min", 0)
        max_time = result.get("max", 0)
        runs = len(result.get("times", []))

        print(f"Command: {command}")
        print(f"  Mean:   {format_time(mean)} ± {format_time(stddev)}")
        print(f"  Range:  {format_time(min_time)} - {format_time(max_time)}")
        print(f"  Runs:   {runs}")

    print()


def compare_baseline() -> None:
    if not BENCHMARKS_DIR.exists():
        print(f"No benchmarks directory found at {BENCHMARKS_DIR}")
        return

    baseline_file = BENCHMARKS_DIR / "baseline.json"
    if not baseline_file.exists():
        print("No baseline.json found. Run with -o baseline first.")
        return

    other_files = sorted([f for f in BENCHMARKS_DIR.glob("*.json") if f.name != "baseline.json"])
    if not other_files:
        print("No other benchmark results to compare against baseline.")
        return

    latest = other_files[-1]
    compare_two(str(baseline_file), str(latest))


def main() -> None:
    parser = argparse.ArgumentParser(description="CLI benchmark runner and comparison tool")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    run_parser = subparsers.add_parser("run", help="Run the benchmark")
    run_parser.add_argument("-o", "--output", help="Output filename (default: timestamp-based)")

    list_parser = subparsers.add_parser("list", help="List all benchmark results")
    list_parser.add_argument("-n", "--limit", type=int, help="Limit number of results")

    compare_parser = subparsers.add_parser("compare", help="Compare two benchmark files")
    compare_parser.add_argument("baseline", help="Baseline benchmark file (or timestamp)")
    compare_parser.add_argument("current", help="Current benchmark file (or timestamp)")

    subparsers.add_parser("latest", help="Show latest benchmark result")
    subparsers.add_parser("compare-baseline", help="Compare baseline with the most recent run")

    args = parser.parse_args()

    if args.command == "run":
        run_benchmark(args.output)
    elif args.command == "list":
        list_benchmarks(args.limit)
    elif args.command == "compare":
        compare_two(args.baseline, args.current)
    elif args.command == "compare-baseline":
        compare_baseline()
    elif args.command == "latest":
        show_latest()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
