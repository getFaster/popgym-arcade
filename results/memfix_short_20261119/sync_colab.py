"""Copy completed T4 runs, including optimizer state, without replacing local results."""

import json
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SESSION = "memfix-short-20261119"
REMOTE = "/content/memfix-short"


def colab(*command, timeout=300):
    return subprocess.run(
        ["colab", *command, "-s", SESSION],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    ).stdout


def sync():
    output = colab("exec", "-f", str(ROOT / "colab_inventory.py"))
    status = json.loads(
        next(line[7:] for line in output.splitlines() if line.startswith("MEMFIX "))
    )
    print(json.dumps(status), flush=True)
    for relative in status["runs"]:
        destination = ROOT / "colab" / relative
        if (destination / "result.json").exists():
            continue
        archive = f"{REMOTE}/transfer.tar"
        with tempfile.TemporaryDirectory(dir=ROOT) as work:
            work = Path(work)
            package = work / "package.py"
            package.write_text(
                "import subprocess\n"
                f'subprocess.run(["tar", "-cf", {archive!r}, "-C", '
                f"{(REMOTE + '/results/colab')!r}, {relative!r}], check=True)\n"
            )
            colab("exec", "-f", str(package))
            colab("download", archive, str(work / "transfer.tar"), timeout=600)
            with tarfile.open(work / "transfer.tar") as source:
                source.extractall(work, filter="data")
            run = work / relative
            assert (run / "model.eqx").stat().st_size > 0
            assert (
                json.loads((run / "result.json").read_text())["actual_steps"] == 98304
            )
            assert list(run.glob("checkpoints/step_*/checkpoint.json"))
            destination.parent.mkdir(parents=True, exist_ok=True)
            run.rename(destination)
            print(f"Copied {relative}", flush=True)
    return status["complete"]


if __name__ == "__main__":
    while True:
        try:
            if sync():
                (ROOT / "colab" / "complete").touch()
                break
        except (
            OSError,
            ValueError,
            StopIteration,
            subprocess.SubprocessError,
        ) as error:
            print(f"{type(error).__name__}: {error}", flush=True)
        time.sleep(60)
