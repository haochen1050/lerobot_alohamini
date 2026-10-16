#!/usr/bin/env python3
"""
Tabulate base-alignment trials from their JSONL logs (examples/debug/align_to_table.py), one row per trial,
with a summary of success rate, final errors, pulses and time. For robustness testing from many start poses.

  python examples/debug/align_report.py                                # all logs
  python examples/debug/align_report.py --since 20261016 --csv outputs/table_alignment/report.csv
  python examples/debug/align_report.py outputs/table_alignment/logs/align_20261016_0443*.jsonl

The start pose is the one the controller recorded ("measured", or "estimated" when the tag was only found
after a search: then the heading assumes the wheels turned exactly as commanded, so wheel slip during the
search shows up as heading error). For older logs without it, the first measurement is used.
Give each trial a --note (e.g. the start spot) when running align_to_table.py to tell trials apart here.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

DEFAULT_LOGS = Path("outputs/table_alignment/logs")

COLUMNS = [
    ("log", "log", 15),
    ("note", "note", 14),
    ("result", "result", 10),
    ("pulses", "pulses", 6),
    ("time_s", "time s", 6),
    ("start_source", "start", 9),
    ("start_distance_cm", "dist0", 6),
    ("start_lateral_cm", "lat0", 6),
    ("start_heading_deg", "head0", 6),
    ("search_deg", "search", 6),
    ("radial_cm", "radial", 6),
    ("orbit_deg", "orbit", 6),
    ("recoveries", "recov", 5),
    ("final_distance_cm", "dist", 6),
    ("final_lateral_cm", "lat", 6),
    ("final_heading_deg", "head", 6),
    ("reason", "reason", 0),
]


def read_events(path: Path) -> list[dict]:
    events = []
    for line in path.read_text().splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a line cut short by a crash or power loss
    return events


def pose_fields(prefix: str, error: dict | None) -> dict:
    """Pose errors as signed text, so +/- reads the same in the table and the CSV."""
    if not error:
        return {f"{prefix}_distance_cm": None, f"{prefix}_lateral_cm": None, f"{prefix}_heading_deg": None}
    return {
        f"{prefix}_distance_cm": f"{error['distance_error_m'] * 100:+.1f}",
        f"{prefix}_lateral_cm": f"{error['lateral_error_m'] * 100:+.1f}",
        f"{prefix}_heading_deg": f"{error['heading_error_deg']:+.1f}",
    }


def first(events: list[dict], kind: str) -> dict | None:
    return next((e for e in events if e.get("event") == kind), None)


def last(events: list[dict], kind: str) -> dict | None:
    return next((e for e in reversed(events) if e.get("event") == kind), None)


def summarize(path: Path) -> dict | None:
    events = read_events(path)
    trial = first(events, "trial")
    if trial is None:
        return None
    row = {"log": path.stem.removeprefix("align_")[:15], "file": str(path), "note": trial.get("note") or ""}
    summary = last(events, "summary")

    if summary is not None:
        row.update(
            result=summary["state"],
            reason=summary.get("reason", ""),
            pulses=summary["steps"],
            time_s=round(summary["duration_s"], 1),
            start_source=summary["start_source"],
            search_deg=round(summary["search_rotation_deg"]),
            radial_cm=round(summary.get("radial_m", 0.0) * 100),
            orbit_deg=round(summary["orbit_deg"]),
            recoveries=summary["recoveries"],
        )
        start, final = summary.get("start_error"), summary.get("final_error")
    else:
        # Older logs: piece the row together from the individual events.
        end = last(events, "ready") or last(events, "fault") or last(events, "aborted")
        kind = end["event"] if end else "incomplete"
        row.update(
            result={"ready": "BASE_READY", "fault": "FAULT", "aborted": "ABORTED"}.get(kind, "INCOMPLETE"),
            reason=(end or {}).get("reason", ""),
            pulses=(end or {}).get("steps"),
            time_s=round(events[-1]["t"] - first(events, "start")["t"], 1)
            if first(events, "start")
            else None,
            search_deg=round(
                sum(e["omega_degps"] * e["duration_s"] for e in events if e["event"] == "search_pulse")
            ),
            radial_cm=None,
            orbit_deg=None,
            recoveries=sum(1 for e in events if e["event"] == "recover"),
        )
        initial = first(events, "initial")
        measure = first(events, "measure")
        if initial and initial.get("valid"):
            row["start_source"], start = "initial", initial["error"]
        elif initial and initial.get("raw_error"):
            row["start_source"], start = "initial", initial["raw_error"]
        elif measure and not row["search_deg"]:
            row["start_source"], start = "1st meas", measure["error"]
        else:
            row["start_source"], start = "unknown", None
        final = (last(events, "ready") or {}).get("error") or (last(events, "measure") or {}).get("error")

    row.update(pose_fields("start", start))
    row.update(pose_fields("final", final))
    return row


def collect(paths: list[str], since: str | None) -> list[Path]:
    files: list[Path] = []
    for p in map(Path, paths or [str(DEFAULT_LOGS)]):
        files += sorted(p.glob("align_*.jsonl")) if p.is_dir() else [p]
    if since:
        files = [f for f in files if f.stem.removeprefix("align_") >= since]
    return files


TEXT_COLUMNS = {"log", "note", "result", "start_source", "reason"}


def fmt(value, width: int, right: bool) -> str:
    """Cell text, '-' for missing; width 0 = no padding."""
    text = "-" if value is None else f"{value:.1f}" if isinstance(value, float) else str(value)
    if not width:
        return text
    text = text[:width]
    return text.rjust(width) if right else text.ljust(width)


def print_table(rows: list[dict]) -> None:
    print("  ".join(fmt(title, width, key not in TEXT_COLUMNS) for key, title, width in COLUMNS))
    for row in rows:
        print("  ".join(fmt(row.get(key), width, key not in TEXT_COLUMNS) for key, _title, width in COLUMNS))


def print_stats(rows: list[dict]) -> None:
    ready = [r for r in rows if r["result"] == "BASE_READY"]
    print(f"\n{len(rows)} trials, {len(ready)} BASE_READY ({100 * len(ready) / len(rows):.0f}%)")
    others = [r for r in rows if r["result"] != "BASE_READY"]
    if others:
        reasons: dict[str, int] = {}
        for r in others:
            key = f"{r['result']}: {(r.get('reason') or '').split(':')[0].split('(')[0].strip()}"
            reasons[key] = reasons.get(key, 0) + 1
        for key, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"  {n} x {key}")
    if not ready:
        return
    for key, label, unit in (
        ("final_distance_cm", "distance", "cm"),
        ("final_lateral_cm", "lateral", "cm"),
        ("final_heading_deg", "heading", "deg"),
    ):
        values = [abs(float(r[key])) for r in ready if r[key] is not None]
        if values:
            print(
                f"  final |{label}|: median {statistics.median(values):.1f} {unit}, worst {max(values):.1f} {unit}"
            )
    for key, label in (("pulses", "pulses"), ("time_s", "time s")):
        values = [r[key] for r in ready if r[key] is not None]
        if values:
            print(f"  {label}: median {statistics.median(values):.0f}, max {max(values):.0f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="*", help=f"Log files or directories (default: {DEFAULT_LOGS})")
    p.add_argument("--since", help="Only logs from this timestamp on, e.g. 20261016 or 20261016_1400")
    p.add_argument("--csv", help="Also write the rows to this CSV file")
    args = p.parse_args()

    rows = [row for f in collect(args.paths, args.since) if (row := summarize(f)) is not None]
    if not rows:
        raise SystemExit("No alignment logs found")
    print_table(rows)
    print_stats(rows)
    if args.csv:
        keys = [key for key, _title, _width in COLUMNS] + ["file"]
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
