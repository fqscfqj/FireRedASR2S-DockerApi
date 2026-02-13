from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def ensure_firered_source(repo_dir: Path, repo_url: str, auto_clone: bool = True) -> None:
    package_dir = repo_dir / "fireredasr2s"
    if not package_dir.exists():
        if not auto_clone:
            raise FileNotFoundError(
                f"FireRedASR2S source not found at {repo_dir}. "
                "Set AUTO_CLONE_FIRERED=1 or mount repository to FIRERED_REPO_DIR."
            )
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        logger.info("FireRedASR2S source missing, cloning from %s", repo_url)
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(repo_dir)],
            check=True,
        )

    path = str(repo_dir)
    if path not in sys.path:
        sys.path.insert(0, path)

