# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "tabulate",
# ]
# ///

"""
Script that will parse the output of `docker buildx bake` and summarize the
steps that take the most time.  This is very helpful for performance tuning
and determining which steps aren't caching but should.
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict

from tabulate import tabulate


@dataclass
class Report:
    total_steps: int
    cached_steps: int
    done_steps: int
    timed_steps: Dict[str, float]
    total_done_time: float
    steps: list[dict]


def parse_buildx_logs(lines: list[str]):
    """
    Parses buildx bake logs line by line to identify each step, its status
    (DONE or CACHED), and the time taken for 'DONE' steps.

    Args:
        log_data (str): A string containing the entire buildx log.

    Returns:
        list: A list of dictionaries, where each dictionary represents a step
              and contains its details.
    """
    # Regex to capture the step line: #<num> [<description>] <command>
    step_start_pattern = re.compile(r"^#(\d+)\s+\[([^\]]+)\]\s+(.+?)$")
    # Regex to capture the 'DONE' status and time taken
    done_pattern = re.compile(r"^#(\d+)\s+DONE\s+([\d.]+)s$")
    # Regex to capture the 'CACHED' status
    cached_pattern = re.compile(r"^#(\d+)\s+CACHED$")

    parsed_steps = defaultdict(dict)
    current_step = None

    for line in lines:
        line = line.strip()

        # Check if the line is the start of a new step
        step_match = step_start_pattern.match(line)
        if step_match:
            # If a new step starts, finalize the previous one if it exists
            if current_step and current_step.get("status") == "UNKNOWN":
                current_step["status"] = "DONE"  # Default to DONE if no status found
                current_step["time_taken_seconds"] = 0.0  # Assign a default time
            key = step_match.group(2).split()[0]
            message = step_match.group(3)
            step_number = int(step_match.group(1))

            # Start a new step entry
            parsed_steps[step_number].update(
                {
                    "step_number": step_number,
                    "key": key,
                    "description": f"{key} {message}",
                    "status": "UNKNOWN",
                    "time_taken_seconds": None,
                }
            )
            current_step = parsed_steps[step_number]
            continue

        # Check for DONE status line
        done_match = done_pattern.match(line)
        if done_match:
            step_number = int(done_match.group(1))
            if step_number in parsed_steps:
                parsed_steps[step_number].update(
                    {
                        "status": "DONE",
                        "time_taken_seconds": float(done_match.group(2)),
                    }
                )
            continue

        # Check for CACHED status line
        cached_match = cached_pattern.match(line)
        if cached_match:
            step_number = int(cached_match.group(1))
            if step_number in parsed_steps:
                parsed_steps[step_number].update(
                    {
                        "status": "CACHED",
                    }
                )
            continue

    return parsed_steps


def generate_report(parsed_results) -> Report | None:
    """
    Generates a summary report and prints the top N longest-running steps.

    Args:
        parsed_results (dict): A dictionary of dictionaries from parse_buildx_logs.
    """
    if not parsed_results:
        print("No parsable steps found in the logs.")
        return

    total_steps = len(parsed_results)
    cached_steps = len([step for step in parsed_results.values() if step["status"] == "CACHED"])
    done_steps = total_steps - cached_steps

    timed_steps = defaultdict(float)
    for step in parsed_results.values():
        timed_steps[step["key"]] += step["time_taken_seconds"] or 0.0

    total_done_time = sum(
        step["time_taken_seconds"]
        for step in parsed_results.values()
        if step["status"] == "DONE" and step["time_taken_seconds"] is not None
    )

    report = Report(
        total_steps=total_steps,
        cached_steps=cached_steps,
        done_steps=done_steps,
        timed_steps=timed_steps,
        total_done_time=total_done_time,
        steps=parsed_results,
    )
    return report


def print_report(report, top_n: int = 20):
    # Filter for DONE steps and sort by time taken
    done_steps_list = [step for step in report.steps.values() if step["status"] == "DONE"]
    done_steps_list.sort(key=lambda x: x["time_taken_seconds"], reverse=True)

    top_N_longest = done_steps_list[:top_n]

    # Print the report
    print("\n" + "=" * 40)
    print("        Buildx Bake Log Report")
    print("=" * 40)
    print(f"Total Steps: {report.total_steps}")
    print(f"Cached Steps: {report.cached_steps}")
    print(f"Done Steps: {report.done_steps}")
    print(f"Total Time for 'DONE' steps: {report.total_done_time:.2f}s")
    print("=" * 40)

    print(f"\nTop {top_n} Longest-Running 'DONE' Steps:")
    if not top_N_longest:
        print("No 'DONE' steps to report.")
    else:
        # Breakdown by step
        headers = ["N", "time", "step", "description"]

        data = []
        for i, step in enumerate(top_N_longest, 1):
            data.append(
                [
                    i,
                    step["time_taken_seconds"],
                    step["step_number"],
                    step["description"],
                ]
            )
        print(tabulate(data, headers=headers))
        print("=" * 40)

        # Breakdown by target
        headers = [
            "timetarget",
        ]
        data = [[t, k] for k, t in sorted(report.timed_steps.items(), key=lambda x: -x[1])]
        print(tabulate(data, headers=headers))

    return report


def stream_lines(filename: str):
    if filename == "-":
        # Read from standard input
        print("Reading log data from standard input. Press Ctrl+D when done.")
        for line in sys.stdin:
            yield line
    else:
        with open(sys.argv[1], "r") as fp:
            for line in fp:
                yield line


if __name__ == "__main__":
    import sys

    if len(sys.argv) <= 1:
        raise Exception("Filename required")

    lines = list(stream_lines(sys.argv[1]))
    parsed = parse_buildx_logs(lines)
    report = generate_report(parsed)
    print_report(report)
