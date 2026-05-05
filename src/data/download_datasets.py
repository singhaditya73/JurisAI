"""
JurisAI - Dataset Downloader
Downloads Indian legal datasets from Hugging Face.
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_dataset
from rich.console import Console
from rich.table import Table

from src.data.data_utils import ensure_dirs, load_config

console = Console()


# ============================================================================
# Dataset Registry
# ============================================================================

DATASETS = {
    "indian-legal-texts": {
        "hf_name": "Techmaestro369/indian-legal-texts-finetuning",
        "description": "Q&A pairs from IPC, CrPC, Constitution of India",
        "type": "instruction",
        "split": "train",
    },
    "indian-legal-sft": {
        "hf_name": "Prarabdha/indian-legal-supervised-fine-tuning-data",
        "description": "Large-scale Indian legal SFT dataset",
        "type": "instruction",
        "split": "train",
    },
}


def download_dataset(
    name: str,
    hf_name: str,
    split: str,
    output_dir: str,
    token: str = None,
) -> int:
    """Download a single dataset from Hugging Face.

    Returns:
        Number of rows downloaded.
    """
    output_path = Path(output_dir)
    ensure_dirs(str(output_path))

    console.print(f"\n[bold blue]📥 Downloading:[/bold blue] {hf_name}")
    console.print(f"   Split: {split}", markup=False)
    console.print(f"   Output: {output_path}", markup=False)

    try:
        # Load from Hugging Face
        ds = load_dataset(
            hf_name,
            split=split,
            token=token,
        )

        # Save locally
        ds.save_to_disk(str(output_path))

        num_rows = len(ds)
        console.print(f"   ✓ Downloaded {num_rows:,} rows", style="green", markup=False)

        # Show sample
        if num_rows > 0:
            console.print(
                f"   Columns: {list(ds.column_names)}", style="dim", markup=False
            )
            console.print(
                f"   Sample: {str(ds[0])[:200]}...", style="dim", markup=False
            )

        return num_rows

    except Exception as e:
        console.print(
            f"   ✗ Error downloading {hf_name}: {e}", style="red", markup=False
        )
        console.print(
            "   Tip: Check if you need to accept the dataset's license on HF",
            style="yellow",
        )
        return 0


def download_all_datasets(token: str = None) -> None:
    """Download all registered datasets."""

    console.print(
        "\n[bold cyan]═══════════════════════════════════════════════════[/bold cyan]"
    )
    console.print("[bold cyan]  JurisAI - Dataset Downloader[/bold cyan]")
    console.print(
        "[bold cyan]═══════════════════════════════════════════════════[/bold cyan]"
    )

    data_config = load_config("data_config.yaml")
    raw_dir = Path(PROJECT_ROOT) / "data" / "raw" / "huggingface"

    # Summary table
    table = Table(title="Datasets to Download")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Description")
    table.add_column("Rows", justify="right")

    total_rows = 0
    results = []

    for name, info in DATASETS.items():
        output_dir = str(raw_dir / name)
        rows = download_dataset(
            name=name,
            hf_name=info["hf_name"],
            split=info["split"],
            output_dir=output_dir,
            token=token,
        )
        total_rows += rows
        results.append((name, info["type"], info["description"], rows))

    # Print summary
    for name, dtype, desc, rows in results:
        table.add_row(
            name, dtype, desc, f"{rows:,}" if rows > 0 else "[red]FAILED[/red]"
        )

    console.print("\n")
    console.print(table)
    console.print(f"\n[bold green]Total rows downloaded: {total_rows:,}[/bold green]")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download JurisAI training datasets")
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token (or set HF_TOKEN env var)",
    )
    args = parser.parse_args()

    # Try env var if no token provided
    token = args.token or os.environ.get("HF_TOKEN")

    if not token:
        console.print(
            "[yellow]⚠ No HF token provided. Some datasets may require authentication.[/yellow]"
        )
        console.print("[yellow]  Set HF_TOKEN env var or pass --token[/yellow]")

    download_all_datasets(token=token)
