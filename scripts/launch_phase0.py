#!/usr/bin/env python3
"""Launch Phase 0 Autonomous Research via Google Antigravity Managed Agent.

This script provisions a remote Linux sandbox environment in Google Cloud,
inlines/uploads AGENTS.md, research_spec.md, and prompts/phase0.md, and starts
the Antigravity research agent in background mode.

State is persisted to runs/state.json.
Reports are downloaded to runs/latest/ upon completion.
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
logger = logging.getLogger("launch_phase0")

# Default settings
DEFAULT_AGENT = "antigravity-preview-09-2026"
DEFAULT_ENVIRONMENT = "remote"
DEFAULT_STATE_FILE = Path("runs/state.json")
DEFAULT_OUTPUT_DIR = Path("runs/latest")
TARGET_REPORT_FILES = [
    "literature_review.md",
    "novelty_report.md",
    "dataset_report.md",
    "phase0_report.md",
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
    except ImportError as e:
        logger.error(
            "The 'google-genai' package is not installed.\n"
            "Please install it using: pip install 'google-genai>=2.3.0'"
        )
        sys.exit(1)

    # Check version if available
    version = getattr(genai, "__version__", None)
    if version:
        logger.info(f"Using google-genai SDK version: {version}")
    return genai


def initialize_genai_client(genai_module: Any) -> Any:
    """Initialize the Google GenAI client, verifying API credentials."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.warning(
            "Neither GEMINI_API_KEY nor GOOGLE_API_KEY environment variable is set.\n"
            "The client will attempt to use default credentials or Vertex AI configuration.\n"
            "To set an API key: export GEMINI_API_KEY='your_api_key' or add it to a .env file."
        )
    try:
        if api_key:
            return genai_module.Client(api_key=api_key)
        return genai_module.Client()
    except Exception as e:
        logger.error(f"Failed to initialize GenAI Client: {e}")
        sys.exit(1)


def compose_phase0_prompt(
    agents_md: str, research_spec_md: str, phase0_prompt_md: str
) -> str:
    """Compose the comprehensive prompt containing full context and directives."""
    composed = f"""# AUTONOMOUS RESEARCH PROJECT: CONDITIONAL MESH RETOPOLOGY
# EXECUTION ENVIRONMENT: REMOTE SANDBOX (antigravity-preview-09-2026)

You are provided with the complete foundational context and operational constraints
for this research project. The documents below are also to be treated as local workspace
references within your remote sandbox.

================================================================================
SECTION 1: AGENTS.MD (Operational Roles & Popperian Falsification Rules)
================================================================================
{agents_md}

================================================================================
SECTION 2: RESEARCH_SPEC.MD (Problem Formulation, Hypothesis & Falsification Gates)
================================================================================
{research_spec_md}

================================================================================
SECTION 3: PHASE 0 DIRECTIVE & DELIVERABLES SPECIFICATION
================================================================================
{phase0_prompt_md}

================================================================================
MANDATORY WORKSPACE ACTION INSTRUCTIONS FOR THE AGENT
================================================================================
1. Create and write the following 4 markdown reports in your current working directory:
   - literature_review.md
   - novelty_report.md
   - dataset_report.md
   - phase0_report.md
2. Ensure each report is rigorous, deeply technical, and strictly adheres to the
   guidelines and falsification gates specified above.
3. In addition to saving these files to disk in your sandbox, output the full contents
   of each report inside clearly delimited markdown code blocks in your final response:
   ```markdown:literature_review.md
   ...
   ```
   ```markdown:novelty_report.md
   ...
   ```
   ```markdown:dataset_report.md
   ...
   ```
   ```markdown:phase0_report.md
   ...
   ```
4. Conclude with a clear executive summary and your definitive GO / NO-GO decision.
5. DO NOT ATTEMPT TO TRAIN ANY MODELS OR BEGIN PHASE 1. Stop for human review.
"""
    return composed


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
        "phase": "phase0",
        "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    if "created_at" not in state_data:
        state_data["created_at"] = state_data["updated_at"]

    if extra:
        state_data.update(extra)

    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(state_data, f, indent=2)
    logger.info(f"State successfully persisted to {state_file.resolve()}")


def upload_workspace_files(
    client: Any, environment_id: str, files_to_upload: List[Path]
) -> None:
    """Attempt to upload workspace files into the remote sandbox environment."""
    if not hasattr(client, "environments") or not hasattr(
        client.environments, "files"
    ):
        logger.info(
            "Client does not expose environments.files upload API; skipping direct file upload (prompt is already inlined)."
        )
        return

    for file_path in files_to_upload:
        if not file_path.exists():
            continue
        try:
            with open(file_path, "rb") as f:
                client.environments.files.upload(
                    environment=environment_id,
                    path=file_path.name,
                    file=f,
                    mime_type="text/markdown",
                    overwrite=True,
                )
            logger.info(f"Uploaded {file_path.name} to remote sandbox {environment_id}")
        except Exception as e:
            logger.warning(
                f"Could not upload {file_path.name} to sandbox ({e}). Prompt inlining ensures the agent has full context."
            )


def extract_reports_from_text(
    output_text: str, output_dir: Path
) -> List[str]:
    """Fallback extractor: parse delimited markdown blocks from agent response."""
    extracted = []
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pattern 1: ```markdown:filename.md\n(content)\n```
    # Pattern 2: ```filename.md\n(content)\n```
    # Pattern 3: # filename.md ...
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
            logger.info(f"Extracted and saved {filename} from interaction response")
            extracted.append(filename)

    return extracted


def download_reports_from_sandbox(
    client: Any,
    environment_id: Optional[str],
    interaction: Any,
    output_dir: Path,
) -> List[str]:
    """Download research reports from the remote sandbox to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded_files: List[str] = []

    # 1. Save full interaction output text
    output_text = getattr(interaction, "output_text", None) or ""
    if output_text:
        (output_dir / "interaction_output.txt").write_text(output_text, encoding="utf-8")
        logger.info(f"Saved raw interaction output to {output_dir / 'interaction_output.txt'}")

    # 2. Try native client.environments.files.download
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

    # 3. Fallback extraction from response text if some files weren't directly downloaded
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
    """Poll the background interaction until it reaches a terminal state."""
    logger.info(f"Monitoring background interaction '{interaction_id}' (Poll interval: {poll_interval}s)...")
    start_time = time.time()
    last_status = "in_progress"

    while True:
        elapsed = int(time.time() - start_time)
        if elapsed > timeout:
            logger.error(f"Polling timed out after {timeout} seconds. Interaction may still be running in background.")
            persist_state(
                state_file,
                interaction_id=interaction_id,
                environment_id=environment_id,
                status="timed_out_monitoring",
                agent=agent_name,
            )
            return

        try:
            interaction = client.interactions.get(id=interaction_id)
            current_status = getattr(interaction, "status", "unknown")
        except Exception as e:
            logger.warning(f"Error querying interaction status: {e}. Retrying in {poll_interval}s...")
            time.sleep(poll_interval)
            continue

        if current_status != last_status:
            logger.info(f"Status changed: {last_status} -> {current_status}")
            last_status = current_status
            persist_state(
                state_file,
                interaction_id=interaction_id,
                environment_id=environment_id,
                status=current_status,
                agent=agent_name,
            )

        if current_status == "completed":
            logger.info("Remote research agent has completed execution!")
            downloaded = download_reports_from_sandbox(
                client=client,
                environment_id=environment_id,
                interaction=interaction,
                output_dir=output_dir,
            )
            logger.info(f"Downloaded {len(downloaded)} reports to {output_dir.resolve()}: {downloaded}")
            persist_state(
                state_file,
                interaction_id=interaction_id,
                environment_id=environment_id,
                status="completed",
                agent=agent_name,
                extra={"downloaded_reports": downloaded},
            )
            break
        elif current_status in ["failed", "cancelled"]:
            logger.error(f"Interaction ended with terminal status: {current_status}")
            error_details = getattr(interaction, "error", None) or "No specific error provided by API."
            logger.error(f"Error details: {error_details}")
            persist_state(
                state_file,
                interaction_id=interaction_id,
                environment_id=environment_id,
                status=current_status,
                agent=agent_name,
                extra={"error": str(error_details)},
            )
            sys.exit(1)
        else:
            logger.info(f"Agent actively working in sandbox... [Elapsed: {elapsed}s | Status: {current_status}]")
            time.sleep(poll_interval)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch Phase 0 Autonomous Research via Google Antigravity Managed Agent"
    )
    parser.add_argument(
        "--agent",
        default=DEFAULT_AGENT,
        help=f"Target Antigravity agent name (default: {DEFAULT_AGENT})",
    )
    parser.add_argument(
        "--environment",
        default=DEFAULT_ENVIRONMENT,
        help=f"Environment mode (default: {DEFAULT_ENVIRONMENT})",
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
        help="Wait and poll for completion rather than exiting immediately after launching background task",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=15,
        help="Polling interval in seconds when --wait is specified (default: 15)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout in seconds when --wait is specified (default: 3600)",
    )
    args = parser.parse_args()

    # Step 1: Verify SDK
    genai = verify_genai_sdk()
    client = initialize_genai_client(genai)

    # Step 2: Load local research context files
    agents_path = Path("AGENTS.md")
    spec_path = Path("research_spec.md")
    prompt_path = Path("prompts/phase0.md")

    logger.info("Loading research specification and prompt files...")
    agents_content = load_file_content(agents_path)
    spec_content = load_file_content(spec_path)
    prompt_content = load_file_content(prompt_path)

    full_prompt = compose_phase0_prompt(agents_content, spec_content, prompt_content)

    # Step 3: Launch remote background interaction
    logger.info(f"Provisioning remote sandbox with agent '{args.agent}'...")
    logger.info("Launching research task with background=True (no local GPU compute)...")

    try:
        interaction = client.interactions.create(
            agent=args.agent,
            environment=args.environment,
            background=True,
            input=full_prompt,
        )
    except Exception as e:
        logger.error(f"Failed to create interaction with Antigravity agent: {e}")
        sys.exit(1)

    interaction_id = getattr(interaction, "id", None) or getattr(interaction, "name", "unknown")
    environment_id = getattr(interaction, "environment_id", None)
    status = getattr(interaction, "status", "in_progress")

    logger.info("=" * 70)
    logger.info(f"Interaction successfully created: {interaction_id}")
    logger.info(f"Remote Environment ID: {environment_id}")
    logger.info(f"Initial Status: {status}")
    logger.info("=" * 70)

    # Step 4: Persist state to runs/state.json
    persist_state(
        state_file=args.state_file,
        interaction_id=interaction_id,
        environment_id=environment_id,
        status=status,
        agent=args.agent,
    )

    # Step 5: Upload files to remote environment if accessible
    if environment_id:
        upload_workspace_files(
            client=client,
            environment_id=environment_id,
            files_to_upload=[agents_path, spec_path, prompt_path],
        )

    # Step 6: Either poll until completion or output resumption guidance
    if args.wait:
        poll_interaction(
            client=client,
            interaction_id=interaction_id,
            environment_id=environment_id,
            state_file=args.state_file,
            output_dir=args.output_dir,
            poll_interval=args.poll_interval,
            timeout=args.timeout,
            agent_name=args.agent,
        )
    else:
        logger.info(
            "\nThe research agent is executing autonomously in Google's cloud sandbox.\n"
            "To monitor progress, resume the conversation, or download final reports, run:\n"
            "    python scripts/resume_research.py --poll\n"
            "or to download reports once completed:\n"
            "    python scripts/resume_research.py --download\n"
        )


if __name__ == "__main__":
    main()
