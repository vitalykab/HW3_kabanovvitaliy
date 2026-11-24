"""Download the exact tokenizer, config and safetensors files used in the experiment."""

from pathlib import Path
from hashlib import file_digest

from huggingface_hub import snapshot_download

MODELS = {
    "bert": (
        "cointegrated/rubert-tiny2",
        "e8ed3b0c8bbf4fb6984c3de043bf7d2f4e5969ae",
        [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.txt",
            "model.safetensors",
        ],
    ),
    "qwen": (
        "Qwen/Qwen2.5-0.5B-Instruct",
        "7ae557604adf67be50417f59c2c2f167def9a775",
        [
            "config.json",
            "generation_config.json",
            "merges.txt",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "model.safetensors",
        ],
    ),
}

WEIGHT_SHA256 = {
    "bert": "26ebb6db2a68593c54c74902d7a74f332da66297693f965cc9f1b0af4abf3894",
    "qwen": "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
}

if __name__ == "__main__":
    for name, (repo, revision, files) in MODELS.items():
        directory = Path("models") / name
        if not all((directory / filename).is_file() for filename in files):
            snapshot_download(
                repo, revision=revision, allow_patterns=files, local_dir=directory
            )
        with (directory / "model.safetensors").open("rb") as weights:
            if file_digest(weights, "sha256").hexdigest() != WEIGHT_SHA256[name]:
                raise ValueError(f"Weight checksum mismatch: {directory}")
