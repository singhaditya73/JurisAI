"""
JurisAI - Data Preprocessor
Cleans, filters, and chunks raw legal text for training.

Optimized for large datasets (6M+ rows):
- Samples a manageable subset FIRST, then processes
- Quality > Quantity for fine-tuning
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_from_disk, Dataset
from rich.console import Console
from rich.progress import track

from src.data.data_utils import (
    load_config, clean_text, has_legal_keywords,
    compute_text_hash, ensure_dirs, save_jsonl
)

console = Console()

# ── Sampling limits ──
# V2: Increased for better accuracy
MAX_INSTRUCT_SAMPLES = 30_000   # Max instruction examples to keep (was 10K)
MAX_PRETRAIN_SAMPLES = 40_000   # Max pretraining text chunks to keep (was 20K)
SAMPLE_POOL_SIZE = 150_000      # How many rows to sample from the full dataset (was 50K)


def load_raw_datasets(raw_dir: str) -> List[Dataset]:
    """Load all raw datasets from disk."""
    datasets = []
    raw_path = Path(raw_dir)
    
    if not raw_path.exists():
        console.print(f"[red]✗ Raw data directory not found: {raw_path}[/red]")
        console.print("[yellow]  Run download_datasets.py first![/yellow]")
        return datasets
    
    for ds_dir in raw_path.iterdir():
        if ds_dir.is_dir() and (ds_dir / "dataset_info.json").exists():
            try:
                ds = load_from_disk(str(ds_dir))
                console.print(f"[green]✓ Loaded {ds_dir.name}: {len(ds):,} rows[/green]")
                console.print(f"  Columns: {ds.column_names}")
                datasets.append(ds)
            except Exception as e:
                console.print(f"[red]✗ Failed to load {ds_dir.name}: {e}[/red]")
    
    return datasets


def smart_sample(dataset: Dataset, n: int, seed: int = 42) -> Dataset:
    """Randomly sample n rows from a large dataset efficiently."""
    if len(dataset) <= n:
        return dataset
    
    console.print(f"[yellow]  Sampling {n:,} from {len(dataset):,} rows (seed={seed})[/yellow]")
    return dataset.shuffle(seed=seed).select(range(n))


def detect_columns(dataset: Dataset) -> Dict[str, str]:
    """Auto-detect which columns map to instruction/context/output."""
    columns = dataset.column_names
    mapping = {}
    
    # Priority-ordered patterns
    instruction_patterns = ["instruction", "question", "query", "prompt"]
    context_patterns = ["context", "input", "passage", "document"]
    output_patterns = ["output", "answer", "response", "completion", "target"]
    
    for col in columns:
        col_lower = col.lower().strip()
        
        if col_lower in instruction_patterns and "instruction" not in mapping:
            mapping["instruction"] = col
        elif col_lower in context_patterns and "context" not in mapping:
            mapping["context"] = col
        elif col_lower in output_patterns and "output" not in mapping:
            mapping["output"] = col
    
    # Fallback: 'text' only = pretraining
    if not mapping and "text" in columns:
        mapping["text"] = "text"
    
    # If we only have context + output but no instruction, promote context
    if "context" in mapping and "instruction" not in mapping and "output" in mapping:
        mapping["instruction"] = mapping.pop("context")
    
    return mapping


def clean_and_filter(
    datasets: List[Dataset],
    min_length: int = 50,
    max_length: int = 8000,
    require_legal: bool = True,
    max_instruct: int = MAX_INSTRUCT_SAMPLES,
    max_pretrain: int = MAX_PRETRAIN_SAMPLES,
) -> Tuple[List[Dict], List[Dict]]:
    """Clean datasets and split into pretraining vs instruction data.
    
    Samples first, then filters for quality.
    
    Returns:
        (pretrain_data, instruct_data)
    """
    pretrain_data = []
    instruct_data = []
    seen_hashes = set()
    
    stats = {
        "total_sampled": 0,
        "too_short": 0,
        "too_long_truncated": 0,
        "no_legal_keywords": 0,
        "duplicates": 0,
        "empty_output": 0,
        "kept_instruct": 0,
        "kept_pretrain": 0,
    }
    
    for ds in datasets:
        col_mapping = detect_columns(ds)
        console.print(f"[cyan]  Column mapping: {col_mapping}[/cyan]")
        
        # Sample a manageable subset from large datasets
        sampled = smart_sample(ds, SAMPLE_POOL_SIZE)
        
        for row in track(sampled, description="Processing rows..."):
            stats["total_sampled"] += 1
            
            # Check if we already have enough
            if len(instruct_data) >= max_instruct and len(pretrain_data) >= max_pretrain:
                console.print("[yellow]  Reached sample limits, stopping early.[/yellow]")
                break
            
            # ── Instruction-format data ──
            if "instruction" in col_mapping and "output" in col_mapping:
                instruction = clean_text(str(row.get(col_mapping["instruction"], "") or ""))
                context = clean_text(str(row.get(col_mapping.get("context", ""), "") or ""))
                output = clean_text(str(row.get(col_mapping["output"], "") or ""))
                
                # Skip empty outputs
                if not output or len(output) < 20:
                    stats["empty_output"] += 1
                    continue
                
                # Skip empty instructions
                if not instruction or len(instruction) < 10:
                    stats["too_short"] += 1
                    continue
                
                combined = f"{instruction} {context} {output}".strip()
                
                if len(combined) < min_length:
                    stats["too_short"] += 1
                    continue
                
                if len(combined) > max_length:
                    stats["too_long_truncated"] += 1
                    output = output[:max_length - len(instruction) - len(context) - 100]
                
                if require_legal and not has_legal_keywords(combined):
                    stats["no_legal_keywords"] += 1
                    continue
                
                # Dedup
                text_hash = compute_text_hash(instruction + output)
                if text_hash in seen_hashes:
                    stats["duplicates"] += 1
                    continue
                seen_hashes.add(text_hash)
                
                # Add to instruction data
                if len(instruct_data) < max_instruct:
                    instruct_data.append({
                        "instruction": instruction,
                        "input": context if context else "",
                        "output": output,
                    })
                    stats["kept_instruct"] += 1
                
                # Also add to pretrain corpus
                if len(pretrain_data) < max_pretrain:
                    pretrain_data.append({"text": combined})
                    stats["kept_pretrain"] += 1
            
            # ── Raw text data (pretraining only) ──
            elif "text" in col_mapping:
                text = clean_text(str(row.get(col_mapping["text"], "") or ""))
                
                if len(text) < min_length:
                    stats["too_short"] += 1
                    continue
                
                if require_legal and not has_legal_keywords(text):
                    stats["no_legal_keywords"] += 1
                    continue
                
                text_hash = compute_text_hash(text)
                if text_hash in seen_hashes:
                    stats["duplicates"] += 1
                    continue
                seen_hashes.add(text_hash)
                
                if len(pretrain_data) < max_pretrain:
                    pretrain_data.append({"text": text})
                    stats["kept_pretrain"] += 1
    
    # Print stats
    console.print("\n[bold]═══ Preprocessing Stats ═══[/bold]")
    for key, val in stats.items():
        color = "green" if "kept" in key else "dim"
        console.print(f"  [{color}]{key}: {val:,}[/{color}]")
    
    return pretrain_data, instruct_data


def create_splits(
    data: List[Dict],
    train_ratio: float = 0.85,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> Dict[str, List[Dict]]:
    """Split data into train/validation/test sets."""
    random.seed(seed)
    
    shuffled = data.copy()
    random.shuffle(shuffled)
    
    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    splits = {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }
    
    for name, split in splits.items():
        console.print(f"  {name}: {len(split):,} examples")
    
    return splits


def preprocess_all() -> None:
    """Run the full preprocessing pipeline."""
    console.print("\n[bold cyan]═══════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]  JurisAI - Data Preprocessor[/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════[/bold cyan]")
    
    data_config = load_config("data_config.yaml")
    
    raw_dir = str(PROJECT_ROOT / "data" / "raw" / "huggingface")
    pretrain_dir = str(PROJECT_ROOT / "data" / "processed" / "pretrain")
    instruct_dir = str(PROJECT_ROOT / "data" / "processed" / "instruct")
    eval_dir = str(PROJECT_ROOT / "data" / "evaluation")
    
    ensure_dirs(pretrain_dir, instruct_dir, eval_dir)
    
    # Step 1: Load raw datasets
    console.print("\n[bold]Step 1: Loading raw datasets...[/bold]")
    datasets = load_raw_datasets(raw_dir)
    
    if not datasets:
        console.print("[red]No datasets found. Run download_datasets.py first![/red]")
        return
    
    # Step 2: Sample, clean, and filter
    console.print("\n[bold]Step 2: Sampling, cleaning, and filtering...[/bold]")
    console.print(f"  Target: {MAX_INSTRUCT_SAMPLES:,} instruction + {MAX_PRETRAIN_SAMPLES:,} pretrain examples")
    
    prep = data_config.get("preprocessing", {})
    pretrain_data, instruct_data = clean_and_filter(
        datasets,
        min_length=prep.get("min_text_length", 50),
        max_length=prep.get("max_text_length", 8000),
        require_legal=prep.get("require_legal_keywords", True),
    )
    
    # Step 3: Create splits
    console.print("\n[bold]Step 3: Creating data splits...[/bold]")
    splits_config = data_config.get("splits", {})
    
    if pretrain_data:
        console.print("\n[blue]Pretrain splits:[/blue]")
        pretrain_splits = create_splits(
            pretrain_data,
            train_ratio=splits_config.get("train", 0.85),
            val_ratio=splits_config.get("validation", 0.10),
            seed=splits_config.get("seed", 42),
        )
        
        for split_name, split_data in pretrain_splits.items():
            save_jsonl(split_data, f"{pretrain_dir}/{split_name}.jsonl")
    
    if instruct_data:
        console.print("\n[blue]Instruction splits:[/blue]")
        instruct_splits = create_splits(
            instruct_data,
            train_ratio=splits_config.get("train", 0.85),
            val_ratio=splits_config.get("validation", 0.10),
            seed=splits_config.get("seed", 42),
        )
        
        for split_name, split_data in instruct_splits.items():
            save_jsonl(split_data, f"{instruct_dir}/{split_name}.jsonl")
        
        # Save test set for evaluation
        save_jsonl(instruct_splits["test"], f"{eval_dir}/test_questions.json")
    
    console.print("\n[bold green]═══════════════════════════════════════════════════[/bold green]")
    console.print(f"[bold green]  ✓ Preprocessing complete![/bold green]")
    console.print(f"[bold green]═══════════════════════════════════════════════════[/bold green]")
    console.print(f"  Pretrain examples: {len(pretrain_data):,}")
    console.print(f"  Instruct examples: {len(instruct_data):,}")
    console.print(f"\n  Next: python -m src.data.prepare_instruct")


if __name__ == "__main__":
    preprocess_all()
