#!/usr/bin/env python3
"""Resume, Monitor, or Query Google Antigravity Remote Research Sandbox.

This script supports:
1. Monitoring/polling an existing background interaction until completion.
2. Downloading generated research reports from the sandbox to runs/latest/.
3. Resuming the exact same remote sandbox and conversation by passing
   environment_id and previous_interaction_id for follow-up turns.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("resume_research")

DEFAULT_STATE_FILE = Path("runs/state.json")
DEFAULT_OUTPUT_DIR = Path("runs/latest")
TARGET_REPORT_FILES = [
    "literature_review.md",
    "novelty_report.md",
    "dataset_report.md",
    "phase0_report.md",
    "phase1_math_spec.md",
    "synthetic_benchmarks.py",
    "toy_flow_sampler.py",
    "phase1_report.md",
    "flow_retopo_model.py",
    "sampler_ablation.py",
    "baseline_comparison.py",
    "phase2_report.md",
]


def load_state(state_file: Path) -> Dict[str, Any]:
    """Load persisted interaction state from JSON."""
    if not state_file.exists():
        logger.error(
            f"State file not found at: {state_file.resolve()}\n"
            "Please run 'python scripts/launch_phase0.py' first to start a research session."
        )
        sys.exit(1)
    try:
        with open(state_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to parse state file {state_file}: {e}")
        sys.exit(1)


def update_state(state_file: Path, updates: Dict[str, Any]) -> Dict[str, Any]:
    """Update state file with new fields and timestamps."""
    state = load_state(state_file)
    state.update(updates)
    state["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    return state


def verify_genai_sdk() -> Any:
    """Verify that google-genai is installed and initialize client."""
    try:
        import google.genai as genai
    except ImportError:
        logger.error("google-genai is not installed. Run: pip install 'google-genai>=2.3.0'")
        sys.exit(1)

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.warning(
            "Neither GEMINI_API_KEY nor GOOGLE_API_KEY is set in environment.\n"
            "The client will attempt to use default credentials."
        )
    try:
        if api_key:
            return genai.Client(api_key=api_key)
        return genai.Client()
    except Exception as e:
        logger.error(f"Failed to initialize GenAI client: {e}")
        sys.exit(1)


def extract_reports_from_text(output_text: str, output_dir: Path) -> List[str]:
    """Fallback extractor: parse delimited markdown blocks from agent response."""
    extracted = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename in TARGET_REPORT_FILES:
        target_path = output_dir / filename
        pattern = re.compile(
            rf"```(?:markdown:)?{re.escape(filename)}\s*\n(.*?)```",
            re.DOTALL | re.IGNORECASE,
        )
        match = pattern.search(output_text)
        if match:
            content = match.group(1).strip()
            target_path.write_text(content, encoding="utf-8")
            logger.info(f"Extracted and saved {filename} from interaction response text")
            extracted.append(filename)

    return extracted


def download_reports(
    client: Any,
    environment_id: Optional[str],
    interaction: Any,
    output_dir: Path,
) -> List[str]:
    """Download reports using client.environments.files and fallback text extraction."""
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded_files: List[str] = []

    # Save output text if present
    output_text = getattr(interaction, "output_text", None) or ""
    if output_text:
        (output_dir / "interaction_output.txt").write_text(output_text, encoding="utf-8")
        logger.info(f"Saved raw interaction output text to {output_dir / 'interaction_output.txt'}")

    # Attempt native environment file download
    if environment_id and hasattr(client, "environments") and hasattr(client.environments, "files"):
        for filename in TARGET_REPORT_FILES:
            target_path = output_dir / filename
            try:
                content = client.environments.files.download(
                    environment=environment_id,
                    path=filename,
                )
                if content:
                    if isinstance(content, str):
                        target_path.write_text(content, encoding="utf-8")
                    else:
                        target_path.write_bytes(content)
                    logger.info(f"Successfully downloaded {filename} via Environments API")
                    downloaded_files.append(filename)
            except Exception as e:
                logger.debug(f"Direct download for {filename} returned: {e}")

    # Fallback to response text extraction if files missing
    missing = [f for f in TARGET_REPORT_FILES if f not in downloaded_files]
    if missing and output_text:
        extracted = extract_reports_from_text(output_text, output_dir)
        for ef in extracted:
            if ef not in downloaded_files:
                downloaded_files.append(ef)

    return downloaded_files


def poll_interaction(
    client: Any,
    interaction_id: str,
    environment_id: Optional[str],
    state_file: Path,
    output_dir: Path,
    poll_interval: int = 15,
    timeout: int = 3600,
) -> None:
    """Continuously poll an interaction until terminal status."""
    logger.info(f"Polling interaction '{interaction_id}' (Poll interval: {poll_interval}s)...")
    start_time = time.time()
    last_status = "unknown"

    while True:
        elapsed = int(time.time() - start_time)
        if elapsed > timeout:
            logger.error(f"Polling timed out after {timeout} seconds.")
            return

        interaction = None
        current_status = "in_progress"

        try:
            interaction = client.interactions.get(id=interaction_id)
            current_status = getattr(interaction, "status", "unknown")
        except Exception as e:
            logger.debug(f"Querying interaction returned: {e}. Checking environment files...")

        # Also probe environments.files directly for completion
        if environment_id and hasattr(client, "environments") and hasattr(client.environments, "files"):
            try:
                env_files = client.environments.files.list(environment=environment_id, path="")
                file_names = [getattr(f, "name", "") for f in getattr(env_files, "files", [])]
                phase2_targets = ["flow_retopo_model.py", "sampler_ablation.py", "baseline_comparison.py", "phase2_report.md"]
                if all(req in file_names for req in phase2_targets):
                    logger.info("All Phase 2 target artifacts detected in remote sandbox!")
                    current_status = "completed"
            except Exception as e:
                logger.debug(f"Environments files check: {e}")

        if current_status != last_status:
            logger.info(f"Interaction status: {current_status}")
            last_status = current_status
            update_state(state_file, {"status": current_status})

        if current_status == "completed":
            logger.info("Task completed! Downloading research reports...")
            downloaded = download_reports(
                client=client,
                environment_id=environment_id,
                interaction=interaction,
                output_dir=output_dir,
            )
            logger.info(f"Downloaded reports: {downloaded}")
            update_state(
                state_file,
                {
                    "status": "completed",
                    "downloaded_reports": downloaded,
                },
            )
            print_completion_summary(output_dir)
            break
        elif current_status in ["failed", "cancelled"]:
            logger.error(f"Interaction ended with status: {current_status}")
            error_details = getattr(interaction, "error", None) or "No details"
            logger.error(f"Error details: {error_details}")
            update_state(state_file, {"status": current_status, "error": str(error_details)})
            sys.exit(1)
        else:
            logger.info(f"Agent running in sandbox... [Elapsed: {elapsed}s | Status: {current_status}]")
            time.sleep(poll_interval)


def print_completion_summary(output_dir: Path) -> None:
    """Print a readable summary of downloaded reports."""
    print("\n" + "=" * 70)
    print("PHASE 0 RESEARCH REPORTS DOWNLOADED")
    print("=" * 70)
    for filename in TARGET_REPORT_FILES:
        filepath = output_dir / filename
        if filepath.exists():
            size_kb = filepath.stat().st_size / 1024.0
            print(f"  [x] {filename:<25} ({size_kb:.1f} KB) -> {filepath.resolve()}")
        else:
            print(f"  [ ] {filename:<25} (Not found)")
    print("=" * 70)

    phase0_report = output_dir / "phase0_report.md"
    if phase0_report.exists():
        content = phase0_report.read_text(encoding="utf-8")
        decision_match = re.search(r"\[(GO|CONDITIONAL GO|NO-GO)\]", content, re.IGNORECASE)
        if decision_match:
            print(f"\n>> Phase 0 Verdict: [{decision_match.group(1).upper()}]")
        print("Review phase0_report.md before proceeding with human approval.")


def send_followup_turn(
    client: Any,
    environment_id: str,
    previous_interaction_id: str,
    prompt: str,
    agent: str,
    state_file: Path,
    background: bool = True,
) -> None:
    """Send a follow-up interaction in the same remote sandbox and conversation."""
    logger.info(f"Resuming conversation in sandbox '{environment_id}'...")
    logger.info(f"Previous interaction ID: '{previous_interaction_id}'")

    try:
        interaction = client.interactions.create(
            agent=agent,
            environment=environment_id,
            previous_interaction_id=previous_interaction_id,
            input=prompt,
            background=background,
        )
    except Exception as e:
        logger.error(f"Failed to resume interaction: {e}")
        sys.exit(1)

    new_interaction_id = getattr(interaction, "id", None) or getattr(interaction, "name", "unknown")
    new_status = getattr(interaction, "status", "in_progress")

    logger.info(f"Follow-up interaction initiated: {new_interaction_id} (Status: {new_status})")
    update_state(
        state_file,
        {
            "interaction_id": new_interaction_id,
            "status": new_status,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monitor, download reports, or resume conversations with the Antigravity remote agent"
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=DEFAULT_STATE_FILE,
        help=f"Path to runs/state.json (default: {DEFAULT_STATE_FILE})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save reports (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Check and print current interaction status without blocking",
    )
    parser.add_argument(
        "--poll",
        action="store_true",
        help="Poll until completion and download reports",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download reports from the existing interaction/environment immediately",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Send a follow-up prompt to resume the conversation in the same sandbox",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="Send contents of a file as a follow-up prompt",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=15,
        help="Seconds between polls (default: 15)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Max polling timeout in seconds (default: 3600)",
    )

    args = parser.parse_args()

    state = load_state(args.state_file)
    interaction_id = state.get("interaction_id")
    environment_id = state.get("environment_id")
    agent = state.get("agent", "antigravity-preview-09-2026")

    if not interaction_id:
        logger.error("State file does not contain a valid 'interaction_id'.")
        sys.exit(1)

    client = verify_genai_sdk()

    # Case 1: Send follow-up prompt to resume conversation in the same sandbox
    if args.prompt or args.prompt_file:
        followup_text = args.prompt
        if args.prompt_file:
            if not args.prompt_file.exists():
                logger.error(f"Prompt file not found: {args.prompt_file}")
                sys.exit(1)
            followup_text = args.prompt_file.read_text(encoding="utf-8")

        if not environment_id:
            logger.warning("No environment_id found in state; resuming will rely on previous_interaction_id alone.")

        send_followup_turn(
            client=client,
            environment_id=environment_id or "remote",
            previous_interaction_id=interaction_id,
            prompt=followup_text,
            agent=agent,
            state_file=args.state_file,
            background=True,
        )
        return

    # Case 2: Status check only
    if args.status:
        try:
            interaction = client.interactions.get(id=interaction_id)
            current_status = getattr(interaction, "status", "unknown")
            print(f"Interaction ID:  {interaction_id}")
            print(f"Environment ID:  {environment_id}")
            print(f"Current Status:  {current_status}")
            update_state(args.state_file, {"status": current_status})
        except Exception as e:
            logger.error(f"Failed to query interaction status: {e}")
            sys.exit(1)
        return

    # Case 3: Download reports directly
    if args.download:
        try:
            interaction = client.interactions.get(id=interaction_id)
            downloaded = download_reports(
                client=client,
                environment_id=environment_id,
                interaction=interaction,
                output_dir=args.output_dir,
            )
            logger.info(f"Downloaded files: {downloaded}")
            print_completion_summary(args.output_dir)
        except Exception as e:
            logger.error(f"Failed to download reports: {e}")
            sys.exit(1)
        return

    # Case 4: Default or explicit --poll: Poll until completion
    poll_interaction(
        client=client,
        interaction_id=interaction_id,
        environment_id=environment_id,
        state_file=args.state_file,
        output_dir=args.output_dir,
        poll_interval=args.poll_interval,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
