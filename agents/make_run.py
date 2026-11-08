"""Create a clean run directory for one agent attempt.

    uv run python -m agents.make_run --agent claude --model argmedian_4_3 --n 1 \
        --root ~/agent-runs --budget "..." --model-access "..." \
        --internet-rule "..." --packages-rule "..."

The run directory gets only what the agent needs: the weights as .npz (so it
needs neither torch nor alg_zoo), a numpy loader that logs every evaluation,
an empty log and TASK.md. Nothing from notes/, docs/, figures/ or the
explanation code. The exact accuracy goes into a manifest NEXT TO the run
directory, where the scorer finds it and the agent does not.

Contamination: Claude Code reads CLAUDE.md from every parent directory, and
Codex/Gemini look for AGENTS.md/GEMINI.md up to the git root. A run inside
this repo would show the agent our project notes, so the default --root is
outside the repo, and the script refuses a root that has one of those files
above it (override with --unsafe-root, for tests only).
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "TASK.md"
LOADER = HERE / "run_template" / "load_model.py"
ALLOWED_FILES = ["weights.npz", "load_model.py", "TASK.md", "model_calls.jsonl"]
CONTEXT_FILES = ["CLAUDE.md", "AGENTS.md", "GEMINI.md", ".claude", ".codex", ".gemini"]
DEFAULT_ROOT = Path.home() / "agent-runs"
PLACEHOLDER = "[[TO BE DECIDED BY XABIER: {}]]"

SETTINGS = {  # TASK.md field -> what is still undecided (see PROTOCOL.md)
    "BUDGET": "budget per run",
    "MODEL_ACCESS": "may the agent run the model, and how many evaluations",
    "INTERNET_RULE": "is any other internet use allowed",
    "PACKAGES_RULE": "may the agent install packages (scipy?)",
}


def parse_model(model):
    m = re.fullmatch(r"argmedian_(\d+)_(\d+)", model)
    if not m:
        raise ValueError(f"only argmedian_<hidden>_<seq_len> models for now, got {model!r}")
    return int(m.group(1)), int(m.group(2))


def render_task(hidden, seq_len, settings=None):
    settings = settings or {}
    text = TEMPLATE.read_text()
    n_params = hidden + hidden * hidden + seq_len * hidden
    fields = {"HIDDEN": hidden, "SEQ_LEN": seq_len, "SEQ_LEN_MINUS_1": seq_len - 1, "N_PARAMS": n_params}
    for key, what in SETTINGS.items():
        fields[key] = settings.get(key) or PLACEHOLDER.format(what)
    for key, value in fields.items():
        text = text.replace("{{" + key + "}}", str(value))
    assert "{{" not in text, "unfilled field in TASK.md"
    return text


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def context_files_above(path):
    """CLAUDE.md, AGENTS.md... in path or any parent (home-level config dirs excluded)."""
    hits = []
    home = Path.home().resolve()
    for d in [path, *path.parents]:
        for name in CONTEXT_FILES:
            if (d / name).exists() and not (d == home and name.startswith(".")):
                hits.append(d / name)
    return hits


def build_run(root, *, agent, model, n, weights, exact_acc, settings=None, git=True):
    """Write <root>/<agent>-<model>-<n>/ and <root>/<agent>-<model>-<n>.manifest.json."""
    hidden, seq_len = parse_model(model)
    root = Path(root).expanduser().resolve()
    run = root / f"{agent}-{model}-{n}"
    if run.exists():
        raise FileExistsError(run)
    run.mkdir(parents=True)
    w = {k: np.asarray(weights[k], dtype=np.float32) for k in ("w_ih", "w_hh", "w_out")}
    assert w["w_hh"].shape == (hidden, hidden) and w["w_out"].shape == (seq_len, hidden)
    np.savez(run / "weights.npz", **w)
    shutil.copyfile(LOADER, run / "load_model.py")
    (run / "TASK.md").write_text(render_task(hidden, seq_len, settings))
    (run / "model_calls.jsonl").touch()
    assert sorted(p.name for p in run.iterdir()) == sorted(ALLOWED_FILES)

    manifest = {
        "agent": agent, "model": model, "n": n,
        "hidden": hidden, "seq_len": seq_len,
        "exact_accuracy": float(exact_acc),
        "weights_sha256": sha256(run / "weights.npz"),
        "loader_sha256": sha256(run / "load_model.py"),
        "settings": settings or {},
    }
    (root / f"{run.name}.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    if git:  # Codex wants a git repo; a fresh one also stops AGENTS.md/GEMINI.md lookup here
        g = ["git", "-c", "user.name=run", "-c", "user.email=run@localhost"]
        subprocess.run(["git", "init", "-q"], cwd=run, check=True)
        subprocess.run([*g, "add", "-A"], cwd=run, check=True)
        subprocess.run([*g, "commit", "-q", "-m", "initial run directory"], cwd=run, check=True)
    return run


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent", required=True, help="claude, codex, gemini, grok")
    p.add_argument("--model", default="argmedian_4_3")
    p.add_argument("--n", type=int, required=True, help="repetition number")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--budget")
    p.add_argument("--model-access")
    p.add_argument("--internet-rule")
    p.add_argument("--packages-rule")
    p.add_argument("--draft", action="store_true", help="allow undecided placeholders (dry runs only)")
    p.add_argument("--unsafe-root", action="store_true", help="skip the CLAUDE.md/AGENTS.md check")
    a = p.parse_args(argv)

    settings = {k: v for k, v in {
        "BUDGET": a.budget, "MODEL_ACCESS": a.model_access,
        "INTERNET_RULE": a.internet_rule, "PACKAGES_RULE": a.packages_rule,
    }.items() if v}
    missing = [k for k in SETTINGS if k not in settings]
    if missing and not a.draft:
        sys.exit(f"undecided settings {missing}: decide them in PROTOCOL.md or pass --draft")
    root = a.root.expanduser().resolve()
    hits = context_files_above(root)
    if hits and not a.unsafe_root:
        sys.exit(f"refusing {root}: agents would read {[str(h) for h in hits]}")

    from leeuwenhoek.exact import exact_accuracy  # needs torch only through models
    from leeuwenhoek.models import load_argmedian

    hidden, seq_len = parse_model(a.model)
    _, w = load_argmedian(hidden, seq_len)
    if seq_len != 3:
        sys.exit("exact accuracy is only implemented for seq_len=3")
    acc = exact_accuracy(w)[0]
    run = build_run(root, agent=a.agent, model=a.model, n=a.n, weights=w, exact_acc=acc, settings=settings)
    print(run)


if __name__ == "__main__":
    main()
