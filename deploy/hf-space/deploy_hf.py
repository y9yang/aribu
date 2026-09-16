from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[2]

HfApi().upload_folder(
    repo_id="Gwynpleina/aribu",
    repo_type="space",
    folder_path=ROOT,
    allow_patterns=[
        "Dockerfile", ".dockerignore", "pyproject.toml",
        "deploy/requirements.txt", "src/*", "app/*", ".streamlit/*",
        "data/demo/*", "data/live/*",
        "models/fusion.pt", "models/radar-only.pt",
    ],
    ignore_patterns=["*__pycache__*", "*.egg-info*", "*.key.json"],
    delete_patterns=["src/*", "app/*", "data/demo/*", "data/live/*"],
    commit_message="deploy v2",
)
