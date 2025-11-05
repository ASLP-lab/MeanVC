from huggingface_hub import snapshot_download
from pathlib import Path

def download_ckpt(dest_dir: str = "src/ckpt") -> None:
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="ASLP-lab/MeanVC",
        allow_patterns=["model_200ms.safetensors", "meanvc_200ms.pt", "fastu2++.pt", "vocos.pt"],
        local_dir=dest_dir,
        local_dir_use_symlinks=False,
        repo_type="model",
    )

if __name__ == "__main__":
    download_ckpt()