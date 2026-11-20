import json
from pathlib import Path

base = Path("/content/memfix-short")
root = base / "results" / "colab"
runs = [
    str(path.parent.relative_to(root))
    for path in sorted(root.glob("*/*/*/result.json"))
    if (path.parent / "model.eqx").is_file()
    and list(path.parent.glob("checkpoints/step_*/checkpoint.json"))
]
log = base / "worker.log"
tail = log.read_text().splitlines()[-4:] if log.exists() else []
print(
    "MEMFIX "
    + json.dumps(
        {
            "complete": (root / "complete").exists(),
            "runs": runs,
            "tail": tail,
        }
    )
)
