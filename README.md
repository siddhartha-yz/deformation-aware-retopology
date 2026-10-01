# Mesh-Auto-Research: Cloud-Based Autonomous Research Harness

An autonomous scientific research project investigating **Conditional & Deformation-Aware Mesh Retopology via Parallel Generative Models (Flow Matching / Diffusion)**.

This repository serves as the orchestration harness that launches and manages Google's **Antigravity Managed Agent** (`antigravity-preview-09-2026`) operating inside a secure, remote Google Cloud Linux sandbox.

---

## 1. Project Motivation & Research Hypothesis

### Core Problem
In computer animation, games, and VFX, high-density 3D geometry must be converted into low-poly quad-dominant meshes (retopology). However:
1. **Classical methods** (QuadriFlow, Instant Meshes, MIQ) rely purely on extrinsic static surface curvature. They are completely oblivious to articulation joints, resulting in diagonal edge loops that pinch and tear when rigged and animated.
2. **Autoregressive models** (PolyGen, MeshGPT) treat 3D meshes as 1D token sequences. They suffer from quadratic latency $O(N^2)$, exposure bias, and artificial ordering constraints on inherently 2D spatial surface graphs.

### Core Scientific Hypothesis
> **Core Hypothesis**: A small, specialized parallel generative model (such as continuous or discrete flow matching / diffusion) is fundamentally better suited than a generalist autoregressive model for low-level mesh topology decisions.

### Initial Task
$$\text{Geometry } (\mathcal{M}_{\text{high}}) + \text{Oracle Deformation / Joint Condition } (\mathcal{C}_{\text{def}}) \longrightarrow \text{Parallel Topology Generator} \longrightarrow \text{Animation-Ready Quad Mesh } (\mathcal{M}_{\text{low}})$$

---

## 2. Phase 0 Scope & Scientific Falsification

### Hard Constraint: Strictly No GPU Training in Phase 0
Phase 0 is a rigorous theoretical, bibliographic, and data-feasibility audit. **No GPU compute, model training, or parameter fitting is performed during Phase 0.**

### Deliverables Expected from the Remote Agent
Upon completion of Phase 0, the remote Antigravity agent will produce four markdown reports:
1. **`literature_review.md`**: Exhaustive survey of traditional remeshing, autoregressive mesh models, geometric diffusion/flow matching, and kinematics-aware 3D representations.
2. **`novelty_report.md`**: Investigation of prior art collisions and a structured closest-work comparison matrix.
3. **`dataset_report.md`**: Feasibility analysis of paired datasets (geometry + skeletal rig + quad topology) using Mixamo, DeformingThings4D, SMPL-X, etc.
4. **`phase0_report.md`**: Executive synthesis evaluating the 4 Falsification Gates and providing a decisive **GO / NO-GO** determination.

### Falsification Gates (Kill Criteria)
The agent operates under a Popperian scientific framework where disproving an invalid assumption early is considered a high-value success:
- **Gate 1 (Prior Art Collision)**: Has an identical deformation-conditioned flow matching retopology model already been published?
- **Gate 2 (Topological Infeasibility)**: Can parallel flow generation satisfy manifold closure invariants without sequential token constraints?
- **Gate 3 (Data Scarcity)**: Are there $\ge 1,000$ permissible paired assets with clean quad topology and skeletal deformation data?
- **Gate 4 (Autoregressive Superiority)**: Does conditioning existing autoregressive models on deformation priors eliminate their bottlenecks, making flow matching redundant?

---

## 3. Repository Structure

```
mesh-auto-research/
├── AGENTS.md                 # Agent roles, multi-persona protocols & falsification rules
├── research_spec.md          # Formal mathematical specification, hypothesis & metrics
├── prompts/
│   ├── phase0.md             # Directive prompt for Phase 0 literature audit
│   └── phase1.md             # Directive prompt for Phase 1 math & toy benchmarks
├── scripts/
│   ├── launch_phase0.py      # Launch Antigravity agent for Phase 0 (background=True)
│   ├── launch_phase1.py      # Advance to Phase 1 in the existing remote sandbox
│   └── resume_research.py    # Monitor, download reports, or resume same sandbox session
├── runs/
│   ├── state.json            # Persisted interaction_id, environment_id, and status
│   └── latest/               # Downloaded research reports and code from sandbox
├── requirements.txt          # Python dependencies (google-genai>=2.3.0)
├── .env.example              # Template for API credentials
├── .gitignore                # Git exclusions
└── README.md                 # Project documentation and operational guide
```

---

## 4. Setup & Installation

### 4.1 Prerequisites
- Python 3.10+
- Google GenAI API Key (or Google Cloud Vertex AI credentials)

### 4.2 Virtual Environment & Dependencies
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4.3 Configure Credentials
Set your Gemini API key in your environment:
```bash
export GEMINI_API_KEY="your-api-key-here"
```
*(Alternatively, for Vertex AI enterprise usage, ensure `GOOGLE_APPLICATION_CREDENTIALS` and `GOOGLE_CLOUD_PROJECT` are configured).*

---

## 5. Usage Guide

### 5.1 Launch Phase 0 (Background Execution)
Launch the remote research agent in Google's cloud sandbox:
```bash
python scripts/launch_phase0.py
```
This command will:
1. Load `AGENTS.md`, `research_spec.md`, and `prompts/phase0.md`.
2. Provision a remote Linux sandbox via `client.interactions.create(agent="antigravity-preview-09-2026", environment="remote", background=True, ...)`.
3. Persist `interaction_id`, `environment_id`, and `status: "in_progress"` to `runs/state.json`.
4. Upload research context documents to the remote sandbox.

#### Optional: Synchronous Polling Mode
If you prefer to keep the terminal open and wait for completion in one command:
```bash
python scripts/launch_phase0.py --wait --poll-interval 15
```

### 5.2 Monitor & Check Status
To check the current status of the background task without blocking:
```bash
python scripts/resume_research.py --status
```

To continuously poll the running agent until it finishes:
```bash
python scripts/resume_research.py --poll
```

### 5.3 Download Research Reports
Once the agent finishes (status: `completed`), download all artifacts into `runs/latest/`:
```bash
python scripts/resume_research.py --download
```
Downloaded artifacts will include:
- `runs/latest/literature_review.md`
- `runs/latest/novelty_report.md`
- `runs/latest/dataset_report.md`
- `runs/latest/phase0_report.md`
- `runs/latest/interaction_output.txt`

### 5.4 Resuming Conversations & Follow-Up Turns
You can send follow-up questions or instructions to the **exact same remote sandbox and conversation history** using `--prompt`:
```bash
python scripts/resume_research.py --prompt "Please elaborate on Section 3.2 of the novelty report regarding Riemannian vs. Euclidean flow matching."
```
The script will reference `environment_id` and `previous_interaction_id`, maintaining full filesystem state and context.

### 5.5 Launch Phase 1 (Mathematical Formulation & Representation Sandbox)
Once Phase 0 is approved by human review, launch Phase 1 into the same cloud sandbox:
```bash
python scripts/launch_phase1.py
```
To poll and wait for Phase 1 completion synchronously:
```bash
python scripts/launch_phase1.py --wait
```
Phase 1 generates and downloads:
- `runs/latest/phase1_math_spec.md` (Full ODE flow matching derivation)
- `runs/latest/synthetic_benchmarks.py` (2D hinge & 3D cylinder test suites)
- `runs/latest/toy_flow_sampler.py` (Lightweight parallel flow prototype)
- `runs/latest/phase1_report.md` (Benchmark results and validation report)

---

## 6. Error Handling & Troubleshooting

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| `ImportError: No module named 'google.genai'` | SDK not installed | Run `pip install 'google-genai>=2.3.0'` |
| `Neither GEMINI_API_KEY nor GOOGLE_API_KEY set` | Missing authentication | Run `export GEMINI_API_KEY="your-key"` |
| `State file not found at runs/state.json` | Launch script hasn't run | Run `python scripts/launch_phase0.py` first |
| `Polling timed out` | Research task took longer than timeout | Rerun `python scripts/resume_research.py --poll --timeout 7200` |
| Sandbox connection reset | Ephemeral network hiccup | State is saved in `runs/state.json`. Rerun `python scripts/resume_research.py --poll` |

---

## 7. Human Review Gate

Phase 0 terminates intentionally after the four reports are produced. **Phase 1 must not begin automatically.**

Human researchers must:
1. Inspect `runs/latest/phase0_report.md` for the GO / NO-GO verdict.
2. Review `novelty_report.md` to confirm the absence of prior-art collisions.
3. Review `dataset_report.md` to verify dataset availability and licensing.
4. Provide formal sign-off before any code or experimental models are implemented in Phase 1.
