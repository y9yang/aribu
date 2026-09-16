from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[2]
REPO_ID = "Gwynpleina/aribu"

api = HfApi()
api.upload_folder(
    repo_id=REPO_ID,
    repo_type="space",
    folder_path=ROOT,
    allow_patterns=[
        "Dockerfile", "pyproject.toml",
        "deploy/requirements.txt", "src/*", "app/*", ".streamlit/config.toml",
        "data/demo/*", "data/live/*",
        "models/fusion.pt", "models/radar-only.pt",
    ],
    ignore_patterns=["*__pycache__*", "*.egg-info*", "*.key.json"],
    delete_patterns=["src/*", "app/*", "data/demo/*", "data/live/*"],
    commit_message="deploy v2",
)
# The Space reads its settings from the README at its root, kept in deploy/hf-space/ so it does not clash with ours.
api.upload_file(
    path_or_fileobj=ROOT / "deploy/hf-space/README.md",
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="space",
    commit_message="update space readme",
)
