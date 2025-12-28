from pathlib import Path

def find_project_root(start: Path | None = None) -> Path:
    start = start or Path.cwd()
    start = start.resolve()

    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p

    raise RuntimeError("Could not find project root")

def build_paths(root: Path) -> dict[str, Path]:
    data = root / "data"
    src = root / "src"
    return {
        "ROOT": root,
        "SRC": src,
        "DATA": data,
        "MODELS": src / "models",
        "CONFIG": root / "config"
    }
