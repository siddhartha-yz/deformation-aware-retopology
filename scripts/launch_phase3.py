#!/usr/bin/env python3
"""Launch Phase 3 Scaling, Kinematic Conditioning & Production Retopology via Google Antigravity.

This script advances the research project to Phase 3 by resuming the remote
Linux sandbox from Phase 2 and launching the production retopology task in background mode.

State is persisted to runs/state.json.
Reports, training harness, and Blender plugin are downloaded to runs/latest/ upon completion.
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
logger = logging.getLogger("launch_phase3")

DEFAULT_AGENT = "antigravity-preview-09-2026"
DEFAULT_STATE_FILE = Path("runs/state.json")
DEFAULT_OUTPUT_DIR = Path("runs/latest")
TARGET_REPORT_FILES = [
    "dataset_pipeline.py",
    "train_flow_retopo.py",
    "blender_retopo_addon.py",
    "phase3_report.md",
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
        "phase": "phase3",
        "current_phase": "phase3_in_progress" if status == "in_progress" else f"phase3_{status}",
        "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    if extra:
        state_data.update(extra)

    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(state_data, f, indent=2)
    logger.info(f"State successfully updated in {state_file.resolve()}")


def download_reports(
    client: Any,
    environment_id: Optional[str],
    interaction: Optional[Any],
    output_dir: Path,
) -> List[str]:
    """Download research artifacts from the remote sandbox to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded_files: List[str] = []

    output_text = getattr(interaction, "output_text", None) or "" if interaction else ""
    if output_text:
        (output_dir / "phase3_interaction_output.txt").write_text(output_text, encoding="utf-8")

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
    """Poll interaction until terminal status or all artifacts are present in environment."""
    logger.info(f"Monitoring Phase 3 background interaction '{interaction_id}'...")
    start_time = time.time()
    last_status = "in_progress"

    while True:
        elapsed = int(time.time() - start_time)
        if elapsed > timeout:
            logger.error(f"Polling timed out after {timeout} seconds.")
            return

        interaction = None
        current_status = "in_progress"

        try:
            interaction = client.interactions.get(id=interaction_id)
            current_status = getattr(interaction, "status", "in_progress")
        except Exception as e:
            logger.debug(f"Querying interaction returned: {e}. Checking environment files...")

        # Also probe environments.files directly for completed artifacts
        if environment_id and hasattr(client, "environments") and hasattr(client.environments, "files"):
            try:
                env_files = client.environments.files.list(environment=environment_id, path="")
                file_names = [getattr(f, "name", "") for f in getattr(env_files, "files", [])]
                all_found = all(req in file_names for req in TARGET_REPORT_FILES)
                if all_found:
                    logger.info("All Phase 3 target artifacts detected in remote sandbox!")
                    current_status = "completed"
            except Exception as e:
                logger.debug(f"Environments files check: {e}")

        if current_status != last_status:
            logger.info(f"Phase 3 Status changed: {last_status} -> {current_status}")
            last_status = current_status
            persist_state(
                state_file,
                interaction_id=interaction_id,
                environment_id=environment_id,
                status=current_status,
                agent=agent_name,
            )

        if current_status == "completed":
            logger.info("Remote research agent has completed Phase 3 execution!")
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
                extra={"phase3_artifacts": downloaded},
            )
            break
        elif current_status in ["failed", "cancelled"]:
            logger.error(f"Interaction ended with terminal status: {current_status}")
            error_details = getattr(interaction, "error", None) or "No details"
            logger.error(f"Error details: {error_details}")
            sys.exit(1)
        else:
            logger.info(f"Agent executing Phase 3 in sandbox... [Elapsed: {elapsed}s | Status: {current_status}]")
            time.sleep(poll_interval)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch Phase 3 Scaling, Kinematic Conditioning & Production Retopology"
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

    env_to_use = "remote" if (args.fresh_env or not existing_env_id) else existing_env_id
    logger.info(f"Target Sandbox Environment: {env_to_use}")

    prompt_path = Path("prompts/phase3.md")
    prompt_content = load_file_content(prompt_path)

    # First, upload prompt to sandbox environment if available
    if env_to_use != "remote":
        try:
            client.environments.files.upload(
                environment=env_to_use,
                path="phase3.md",
                file=str(prompt_path),
                overwrite=True,
            )
            logger.info("Uploaded prompts/phase3.md into remote sandbox environment.")
        except Exception as e:
            logger.warning(f"Note on prompt upload: {e}")

    full_prompt = f"""# PHASE 3 DIRECTIVE: SCALING, KINEMATIC CONDITIONING & PRODUCTION RETOPOLOGY
Human review of Phase 2 has been officially APPROVED with a [GO / APPROVED] verdict.
Proceed with execution of Phase 3 work packages as specified below.

{prompt_content}
"""

    logger.info(f"Launching Phase 3 with agent '{args.agent}' (background=True)...")

    create_kwargs: Dict[str, Any] = {
        "agent": args.agent,
        "environment": env_to_use,
        "background": True,
        "input": full_prompt,
    }

    try:
        interaction = client.interactions.create(**create_kwargs)
    except Exception as e:
        logger.error(f"Failed to launch Phase 3 interaction: {e}")
        sys.exit(1)

    new_interaction_id = getattr(interaction, "id", None) or getattr(interaction, "name", "unknown")
    environment_id = getattr(interaction, "environment_id", None) or existing_env_id
    status = getattr(interaction, "status", "in_progress")

    logger.info("=" * 70)
    logger.info(f"Phase 3 Interaction created: {new_interaction_id}")
    logger.info(f"Remote Environment ID: {environment_id}")
    logger.info(f"Status: {status}")
    logger.info("=" * 70)

    persist_state(
        state_file=args.state_file,
        interaction_id=new_interaction_id,
        environment_id=environment_id,
        status=status,
        agent=args.agent,
        extra={"human_review_status": "approved", "phase": "phase3", "current_phase": "phase3_in_progress"},
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
            "\nPhase 3 is now executing autonomously in Google's cloud sandbox.\n"
            "To monitor progress or download artifacts once finished, run:\n"
            "    python scripts/resume_research.py --poll\n"
        )


if __name__ == "__main__":
    main()
