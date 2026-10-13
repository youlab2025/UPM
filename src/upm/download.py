"""Download the complete UPM inference checkpoint from Hugging Face."""
import argparse
from pathlib import Path
import shutil

from .bundle import BUNDLE_FILES, verify_bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="zongting/UPM", help="Hugging Face model repository")
    parser.add_argument("--revision", help="HF commit hash, tag, or branch")
    parser.add_argument("--output", default="checkpoints/upm", help="Checkpoint directory")
    parser.add_argument("--local-files-only", action="store_true", help="Use existing local/cache files")
    args = parser.parse_args(argv)
    from huggingface_hub import snapshot_download
    snapshot = Path(snapshot_download(
        repo_id=args.repo_id, revision=args.revision,
        allow_patterns=list(BUNDLE_FILES), local_files_only=args.local_files_only,
    ))
    verify_bundle(snapshot)
    root = Path(args.output).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    if root.resolve() != snapshot.resolve():
        for name in BUNDLE_FILES:
            shutil.copyfile(snapshot / name, root / name)
    verify_bundle(root)
    print(f"Verified UPM checkpoint: {root.resolve()}")


if __name__ == "__main__":
    main()
