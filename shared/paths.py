from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def data(self) -> Path:
        return self.root / "data"


def get_project_root(start: Path | None = None) -> Path:
    cur = (start or Path.cwd()).resolve()
    for p in [cur, *cur.parents]:
        if (p / ".git").exists():
            return p
    return cur


def get_paths(start: Path | None = None) -> ProjectPaths:
    root = get_project_root(start)
    return ProjectPaths(root=root)

