"""Drive the OpenFOAM case from the simulator (OpenFOAM v2406 inside WSL).

The case (../openfoam_desalination) is copied to a fresh run directory,
a small set of boundary/control values are overwritten from the simulator
configuration, and buoyantPimpleFoam is launched in WSL in the background.

LIMITS (do not hide these in reports):
* The case solves buoyant heat transfer + flow ONLY. It has no
  evaporation/condensation mass-transfer source, so it yields NO freshwater
  number. It is not CFD validation of the Python model.
* The mesh is fixed (one 3.6 m^2 module); only intensive quantities (fluxes,
  temperatures, h) and run controls are mapped. Changing area needs re-meshing.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASE = PROJECT_ROOT.parent / "openfoam_desalination"
RUNS_DIR = PROJECT_ROOT / "results" / "openfoam_runs"
DEFAULT_DISTRO = "Ubuntu"
FOAM_BASHRC = "/usr/lib/openfoam/openfoam2406/etc/bashrc"
_COPY = ("0", "constant", "system", "case.foam")
_PROCS: Dict[str, subprocess.Popen] = {}


def to_wsl_path(p: Path) -> str:
    p = Path(p).resolve()
    drive = p.drive.rstrip(":").lower()
    return f"/mnt/{drive}/" + "/".join(p.parts[1:])


def openfoam_available(distro: str = DEFAULT_DISTRO) -> bool:
    try:
        r = subprocess.run(["wsl", "-d", distro, "--", "test", "-f", FOAM_BASHRC],
                           capture_output=True, timeout=60)
        return r.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def map_simulator_to_case(cfg, result) -> Dict[str, float]:
    """Intensive quantities from the Python model -> case parameters.

    q_net is the model's useful heat per effective area over the sunlight
    period (the case's evaporator BC prescribes a net flux).
    """
    q = result.Q_useful_J / (result.A_effective * cfg.sunlight_hours * 3600.0)
    return {
        "q_net_W_m2": float(q),
        "T_ambient_K": float(cfg.ambient_temperature + 273.15),
        "h_glass_W_m2K": 18.0,   # case default; not derived from the Python model
        "end_time_s": 600.0,
        "write_interval_s": 60.0,
    }


def _edit_block(text: str, block: str, key: str, value: float) -> str:
    """Replace `key uniform <n>;` inside `block { ... }`."""
    m = re.search(rf"(\n\s*{block}\s*\{{)(.*?)(\n\s*\}})", text, re.S)
    if not m:
        raise ValueError(f"Block '{block}' not found.")
    body, n = re.subn(rf"(\b{key}\s+uniform\s+)[-+0-9.eE]+;", rf"\g<1>{value:g};", m.group(2))
    if n != 1:
        raise ValueError(f"Key '{key}' not found exactly once in block '{block}'.")
    return text[:m.start(2)] + body + text[m.end(2):]


def prepare_case(params: Dict[str, float], case_src: Path = DEFAULT_CASE,
                 runs_dir: Path = RUNS_DIR) -> Path:
    case_src = Path(case_src)
    if not (case_src / "system" / "controlDict").exists():
        raise FileNotFoundError(f"No OpenFOAM case at {case_src}")
    run_dir = Path(runs_dir) / time.strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True)
    for name in _COPY:
        s = case_src / name
        if s.is_dir():
            shutil.copytree(s, run_dir / name)
        elif s.exists():
            shutil.copy2(s, run_dir / name)

    def rw(rel, fn):
        f = run_dir / rel
        with open(f, encoding="utf-8", errors="surrogateescape", newline="") as fh:
            t = fh.read()
        t = fn(t)
        with open(f, "w", encoding="utf-8", errors="surrogateescape", newline="") as fh:
            fh.write(t)

    def edit_T(t: str) -> str:
        t = re.sub(r"(internalField\s+uniform\s+)[-+0-9.eE]+;",
                   rf"\g<1>{params['T_ambient_K']:g};", t, count=1)
        t = _edit_block(t, "evaporatorSurface", "q", params["q_net_W_m2"])
        t = _edit_block(t, "glassCover", "h", params["h_glass_W_m2K"])
        return _edit_block(t, "glassCover", "Ta", params["T_ambient_K"])

    def edit_ctrl(t: str) -> str:
        t = re.sub(r"(?m)^(endTime\s+)\S+;", rf"\g<1>{params['end_time_s']:g};", t, count=1)
        t = re.sub(r"(?m)^(writeInterval\s+)\S+;", rf"\g<1>{params['write_interval_s']:g};", t, count=1)
        # sample the evaporator temperature in simulated time, not step count
        # (steps are variable, so a step-based interval gives a sparse series)
        t, n = re.subn(r"(evaporatorTemp\s*\{[^}]*?)writeControl\s+timeStep;(\s*)writeInterval\s+\d+;",
                       rf"\g<1>writeControl    adjustableRunTime;\g<2>writeInterval   {params['write_interval_s'] / 4:g};",
                       t, count=1, flags=re.S)
        if n != 1:
            raise ValueError("evaporatorTemp function block not found in controlDict.")
        return t

    rw("0/T", edit_T)
    rw("system/controlDict", edit_ctrl)
    (run_dir / "params.txt").write_text(
        "\n".join(f"{k} = {v}" for k, v in params.items()), encoding="utf-8")
    return run_dir


def start_run(run_dir: Path, distro: str = DEFAULT_DISTRO) -> None:
    run_dir = Path(run_dir)
    wp = to_wsl_path(run_dir)
    cmd = (f"source {FOAM_BASHRC}; cd '{wp}'; "
           "buoyantPimpleFoam > log.run 2>&1; echo $? > exit.code")
    _PROCS[str(run_dir)] = subprocess.Popen(["wsl", "-d", distro, "--", "bash", "-lc", cmd])


def stop_run(run_dir: Path, distro: str = DEFAULT_DISTRO) -> None:
    subprocess.run(["wsl", "-d", distro, "--", "pkill", "-f", "buoyantPimpleFoam"],
                   capture_output=True, timeout=30)
    proc = _PROCS.get(str(run_dir))
    if proc and proc.poll() is None:
        proc.terminate()


@dataclass
class RunStatus:
    state: str                     # running | finished | failed | not_started
    sim_time_s: float
    end_time_s: float
    exec_time_s: Optional[float]
    exit_code: Optional[int]
    log_tail: str
    T_evap_series: List[tuple]     # (time_s, T_K) area-average on evaporatorSurface


def run_status(run_dir: Path) -> RunStatus:
    run_dir = Path(run_dir)
    log = run_dir / "log.run"
    text = log.read_text(errors="replace") if log.exists() else ""
    times = re.findall(r"(?m)^Time = ([-+0-9.eE]+)", text)
    sim_t = float(times[-1]) if times else 0.0
    ex = re.findall(r"ExecutionTime = ([0-9.]+) s", text)
    end_m = re.search(r"(?m)^endTime\s+([-+0-9.eE]+);",
                      (run_dir / "system" / "controlDict").read_text(errors="replace"))
    code_f = run_dir / "exit.code"
    code = None
    if code_f.exists() and code_f.read_text().strip():
        code = int(code_f.read_text().strip())
    if code is None:
        proc = _PROCS.get(str(run_dir))
        if proc is not None:
            state = "running" if proc.poll() is None else "failed"
        else:
            state = "running" if log.exists() else "not_started"
    else:
        state = "finished" if code == 0 else "failed"

    series: List[tuple] = []
    for dat in run_dir.glob("postProcessing/evaporatorTemp/*/surfaceFieldValue.dat"):
        for line in dat.read_text(errors="replace").splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            try:
                series.append((float(parts[0]), float(parts[-1])))
            except ValueError:
                pass
    series.sort()
    return RunStatus(state, sim_t, float(end_m.group(1)) if end_m else 0.0,
                     float(ex[-1]) if ex else None, code, "\n".join(text.splitlines()[-25:]), series)
