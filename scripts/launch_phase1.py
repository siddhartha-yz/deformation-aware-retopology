#!/usr/bin/env python3
"""Launch Phase 1 Mathematical Formulation & Representation Sandbox via Google Antigravity.

This script advances the research project to Phase 1 by resuming the remote
Linux sandbox from Phase 0 (or creating a fresh one) and launching the
representation sandbox task in background mode.

State is persisted to runs/state.json.
Reports and prototype code are downloaded to runs/latest/ upon completion.
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

# Setup structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("launch_phase1")

DEFAULT_AGENT = "antigravity-preview-09-2026"
DEFAULT_STATE_FILE = Path("runs/state.json")
DEFAULT_OUTPUT_DIR = Path("runs/latest")
TARGET_REPORT_FILES = [
    "phase1_math_spec.md",
    "synthetic_benchmarks.py",
    "toy_flow_sampler.py",
    "phase1_report.md",
]


def load_file_content(path: Path) -> str:
    """Load text content from a file with descriptive error handling."""
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path.resolve()}")
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to read {path}: {e}") from e


def verify_genai_sdk() -> Any:
    """Verify that google-genai is installed and >= 2.3.0."""
    try:
        import google.genai as genai
    except ImportError:
        logger.error("google-genai is not installed. Run: pip install 'google-genai>=2.3.0'")
        sys.exit(1)

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.warning(
            "Neither GEMINI_API_KEY nor GOOGLE_API_KEY environment variable is set.\n"
            "The client will attempt to use default credentials or Vertex AI configuration."
        )
    try:
        if api_key:
            return genai.Client(api_key=api_key)
        return genai.Client()
    except Exception as e:
        logger.error(f"Failed to initialize GenAI Client: {e}")
        sys.exit(1)


def load_state(state_file: Path) -> Dict[str, Any]:
    """Load persisted state from JSON."""
    if not state_file.exists():
        logger.error(f"State file not found at: {state_file.resolve()}")
        sys.exit(1)
    with open(state_file, "r", encoding="utf-8") as f:
        return json.load(f)


def persist_state(
    state_file: Path,
    interaction_id: str,
    environment_id: Optional[str],
    status: str,
    agent: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist interaction and environment state to JSON."""
    state_file.parent.mkdir(parents=True, exist_ok=True)
    existing_state: Dict[str, Any] = {}
    if state_file.exists():
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                existing_state = json.load(f)
        except Exception:
            existing_state = {}

    state_data: Dict[str, Any] = {
        **existing_state,
        "interaction_id": interaction_id,
        "environment_id": environment_id,
        "status": status,
        "agent": agent,
        "phase": "phase1",
        "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    if extra:
        state_data.update(extra)

    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(state_data, f, indent=2)
    logger.info(f"State successfully updated in {state_file.resolve()}")


def extract_reports_from_text(output_text: str, output_dir: Path) -> List[str]:
    """Fallback extractor: parse delimited markdown/python blocks from agent response."""
    extracted = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename in TARGET_REPORT_FILES:
        target_path = output_dir / filename
        pattern = re.compile(
            rf"```(?:markdown:|python:)?{re.escape(filename)}\s*\n(.*?)```",
            re.DOTALL | re.IGNORECASE,
        )
        match = pattern.search(output_text)
        if match:
            content = match.group(1).strip()
            target_path.write_text(content, encoding="utf-8")
            logger.info(f"Extracted and saved {filename} from interaction response")
            extracted.append(filename)

    return extracted


def download_reports(
    client: Any,
    environment_id: Optional[str],
    interaction: Any,
    output_dir: Path,
) -> List[str]:
    """Download research artifacts from the remote sandbox to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded_files: List[str] = []

    output_text = getattr(interaction, "output_text", None) or ""
    if output_text:
        (output_dir / "phase1_interaction_output.txt").write_text(output_text, encoding="utf-8")

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

    missing_files = [f for f in TARGET_REPORT_FILES if f not in downloaded_files]
    if missing_files and output_text:
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
    agent_name: str = DEFAULT_AGENT,
) -> None:
    """Poll interaction until terminal status."""
    logger.info(f"Monitoring Phase 1 background interaction '{interaction_id}'...")
    start_time = time.time()
    last_status = "in_progress"

    while True:
        elapsed = int(time.time() - start_time)
        if elapsed > timeout:
            logger.error(f"Polling timed out after {timeout} seconds.")
            return

        try:
            interaction = client.interactions.get(id=interaction_id)
            current_status = getattr(interaction, "status", "unknown")
        except Exception as e:
            logger.warning(f"Error querying interaction status: {e}. Retrying in {poll_interval}s...")
            time.sleep(poll_interval)
            continue

        if current_status != last_status:
            logger.info(f"Phase 1 Status changed: {last_status} -> {current_status}")
            last_status = current_status
            persist_state(
                state_file,
                interaction_id=interaction_id,
                environment_id=environment_id,
                status=current_status,
                agent=agent_name,
            )

        if current_status == "completed":
            logger.info("Remote research agent has completed Phase 1 execution!")
            downloaded = download_reports(
                client=client,
                environment_id=environment_id,
                interaction=interaction,
                output_dir=output_dir,
            )
            logger.info(f"Downloaded {len(downloaded)} artifacts to {output_dir.resolve()}: {downloaded}")
            persist_state(
                state_file,
                interaction_id=interaction_id,
                environment_id=environment_id,
                status="completed",
                agent=agent_name,
                extra={"phase1_artifacts": downloaded},
            )
            break
        elif current_status in ["failed", "cancelled"]:
            logger.error(f"Interaction ended with terminal status: {current_status}")
            error_details = getattr(interaction, "error", None) or "No details"
            logger.error(f"Error details: {error_details}")
            sys.exit(1)
        else:
            logger.info(f"Agent executing Phase 1 in sandbox... [Elapsed: {elapsed}s | Status: {current_status}]")
            time.sleep(poll_interval)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch Phase 1 Mathematical Formulation & Representation Sandbox"
    )
    parser.add_argument(
        "--agent",
        default=DEFAULT_AGENT,
        help=f"Target Antigravity agent name (default: {DEFAULT_AGENT})",
    )
    parser.add_argument(
        "--fresh-env",
        action="store_true",
        help="Force creation of a new remote sandbox environment instead of resuming existing sandbox",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=DEFAULT_STATE_FILE,
        help=f"Path to persist state JSON (default: {DEFAULT_STATE_FILE})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save downloaded reports (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait and poll for completion rather than exiting immediately",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=15,
        help="Polling interval in seconds (default: 15)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout in seconds when --wait is specified (default: 3600)",
    )
    args = parser.parse_args()

    client = verify_genai_sdk()
    state = load_state(args.state_file)

    existing_env_id = state.get("environment_id")
    prev_interaction_id = state.get("interaction_id")

    env_to_use = "remote" if (args.fresh_env or not existing_env_id) else existing_env_id
    logger.info(f"Target Sandbox Environment: {env_to_use}")

    prompt_path = Path("prompts/phase1.md")
    prompt_content = load_file_content(prompt_path)

    full_prompt = f"""# PHASE 1 DIRECTIVE: MATHEMATICAL FORMULATION & REPRESENTATION SANDBOX
Human review of Phase 0 has been officially APPROVED with a [GO] verdict.
Proceed with execution of Phase 1 work packages as specified below.

{prompt_content}
"""

    logger.info(f"Launching Phase 1 with agent '{args.agent}' (background=True)...")

    create_kwargs: Dict[str, Any] = {
        "agent": args.agent,
        "environment": env_to_use,
        "background": True,
        "input": full_prompt,
    }
    if not args.fresh_env and prev_interaction_id:
        create_kwargs["previous_interaction_id"] = prev_interaction_id

    try:
        interaction = client.interactions.create(**create_kwargs)
    except Exception as e:
        logger.error(f"Failed to launch Phase 1 interaction: {e}")
        sys.exit(1)

    new_interaction_id = getattr(interaction, "id", None) or getattr(interaction, "name", "unknown")
    environment_id = getattr(interaction, "environment_id", None) or existing_env_id
    status = getattr(interaction, "status", "in_progress")

    logger.info("=" * 70)
    logger.info(f"Phase 1 Interaction created: {new_interaction_id}")
    logger.info(f"Remote Environment ID: {environment_id}")
    logger.info(f"Status: {status}")
    logger.info("=" * 70)

    persist_state(
        state_file=args.state_file,
        interaction_id=new_interaction_id,
        environment_id=environment_id,
        status=status,
        agent=args.agent,
    )

    if args.wait:
        poll_interaction(
            client=client,
            interaction_id=new_interaction_id,
            environment_id=environment_id,
            state_file=args.state_file,
            output_dir=args.output_dir,
            poll_interval=args.poll_interval,
            timeout=args.timeout,
            agent_name=args.agent,
        )
    else:
        logger.info(
            "\nPhase 1 is now executing autonomously in Google's cloud sandbox.\n"
            "To monitor progress or download artifacts once finished, run:\n"
            "    python scripts/resume_research.py --poll\n"
        )


if __name__ == "__main__":
    main()
