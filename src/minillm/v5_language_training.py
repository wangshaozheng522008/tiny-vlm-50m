"""Pinned V5 training profile and the shared Pilot launch gates."""

from pathlib import Path

from .assessment_io import read_json
from .corpus_utils import verify_file
from .corpus_versions import V5_SEQ_LEN, V5_TARGET, V5_TOKENS_PER_UPDATE, V5_UPDATES, V5_VERSION
from .evaluation import EvaluationConfig
from .v4_language_scaling import INITIAL, INITIAL_FILE, ROOT
from .v5_language_corpus import OUTPUT, SAMPLER_REPORT, sampler_budget

TRAINING_CONFIG = ROOT / "configs/pilot-v5-language-10m-t1200.json"
EVALUATION_CONFIG = ROOT / "configs/evaluation.v4-language-pilot.json"
DATASET = OUTPUT.with_name(OUTPUT.name + "-reviewed-final")
CONTENT_ID = "2a688379c66d701d41cede75961ca71af0dbb8254c2a5c065220b19ee4ea54ce"
RUN = ROOT / "runs/v5-language-pilot-10m-v1"
WORKSPACE = ROOT / "artifacts/v5-language-pilot-10m-v1/workspace"
REPORT = ROOT / "reports/v5-language-pilot-10m-v1"
NOMINAL_TARGET = 10_000_000
MILESTONES = (0, 245, 489, 733, 978, 1222)


def validate_budget(samples, opts, seq_len, planned):
    if (opts.batch_size, opts.grad_accum, opts.seed, seq_len, opts.max_steps, planned,
            tuple(opts.milestone_steps), int(opts.max_steps * opts.warmup_ratio)) != (
            1, 8, 42, V5_SEQ_LEN, V5_UPDATES, V5_TARGET, MILESTONES, 20):
        raise ValueError("V5 requires the fixed 1,222-step schedule and six milestones")
    if samples != 9789 or opts.max_steps * opts.grad_accum != 9776:
        raise ValueError("V5 requires 9,789 frozen windows and consumes 9,776 without repetition")


def validate_profile(dataset, raw, workspace, evaluation=None):
    manifest = read_json(Path(dataset) / "dataset-manifest.json")
    if (Path(dataset).resolve() != DATASET or manifest["corpus_version"] != V5_VERSION
            or manifest["content_id"] != CONTENT_ID):
        raise ValueError("V5 requires the pinned formal 10M release")
    expected = read_json(TRAINING_CONFIG)
    compared = {key: value for key, value in raw.items()
                if key not in ("resume", "workload_report", "lineage_report")}
    if compared != expected or Path(compared["out"]).resolve() != RUN or Path(workspace).resolve() != WORKSPACE:
        raise ValueError("V5 requires the pinned independent training profile")
    if raw.get("resume") and Path(raw["resume"]).resolve() not in {RUN / "state.pt", RUN / "state.prev.pt"}:
        raise ValueError("V5 resumes only its own recovery state")
    if evaluation is not None and evaluation != EvaluationConfig(**read_json(EVALUATION_CONFIG)):
        raise ValueError("V5 requires the fixed historical evaluation profile")
    verify_file(INITIAL, INITIAL_FILE)


def validate_launch_config(dataset, raw):
    validate_profile(dataset, {key: value for key, value in raw.items() if key != "init_from"}, WORKSPACE)
    initial = raw.get("init_from")
    if raw.get("resume"):
        if initial:
            raise ValueError("V5 cannot resume and reinitialize")
    else:
        if not initial or Path(initial).resolve() != WORKSPACE / "initial.pt":
            raise ValueError("V5 launch requires its pinned PR #16 Step 0 copy")
        verify_file(Path(initial), INITIAL_FILE)


def preview(dataset):
    dataset = Path(dataset)
    measured = sampler_budget(dataset, read_json(dataset / ".v4/plan.json"))
    budget = measured["budgets"]["10M"]
    if (measured != read_json(dataset / SAMPLER_REPORT) or budget["status"] != "FEASIBLE"
            or budget["required_effective_tokens"] != V5_TARGET
            or budget["required_windows"] != 9776):
        raise ValueError("V5 Seed 42 consumption differs from the frozen admission report")
    return budget
