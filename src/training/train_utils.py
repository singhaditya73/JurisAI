"""
JurisAI - Training Utilities
Shared helpers for training pipeline.
Uses Unsloth's FastLanguageModel for 2-5x faster QLoRA training.
"""

import sys
import gc
import torch
from pathlib import Path
from typing import Dict, Any, Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.table import Table

from src.data.data_utils import load_config

console = Console()


def print_gpu_info() -> None:
    """Print GPU status and memory."""
    if not torch.cuda.is_available():
        console.print("[red]✗ No CUDA GPU detected![/red]")
        return
    
    gpu_name = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    free = total_mem - reserved
    
    table = Table(title="GPU Status")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Device", gpu_name)
    table.add_row("Total VRAM", f"{total_mem:.1f} GB")
    table.add_row("Allocated", f"{allocated:.2f} GB")
    table.add_row("Reserved", f"{reserved:.2f} GB")
    table.add_row("Free", f"{free:.1f} GB")
    table.add_row("CUDA Version", torch.version.cuda or "N/A")
    table.add_row("PyTorch", torch.__version__)
    console.print(table)


def clear_gpu_memory() -> None:
    """Aggressively clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_model_and_tokenizer(
    model_config: Dict[str, Any],
    lora_config: Optional[Dict[str, Any]] = None,
):
    """Load model with Unsloth FastLanguageModel for QLoRA.
    
    Returns:
        (model, tokenizer) tuple
    """
    from unsloth import FastLanguageModel
    
    base = model_config["base_model"]
    
    console.print(f"\n[bold blue]Loading model: {base['name']}[/bold blue]")
    console.print(f"  4-bit quantization: {base.get('load_in_4bit', True)}")
    console.print(f"  Max seq length: {base.get('max_seq_length', 2048)}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base["name"],
        max_seq_length=base.get("max_seq_length", 2048),
        dtype=None,
        load_in_4bit=base.get("load_in_4bit", True),
        trust_remote_code=base.get("trust_remote_code", True),
    )
    
    console.print("[green]✓ Model loaded successfully[/green]")
    print_gpu_info()
    
    if lora_config:
        console.print(f"\n[bold blue]Applying LoRA adapters[/bold blue]")
        console.print(f"  Rank: {lora_config.get('r', 16)}")
        console.print(f"  Alpha: {lora_config.get('alpha', 32)}")
        
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_config.get("r", 16),
            lora_alpha=lora_config.get("alpha", 32),
            lora_dropout=lora_config.get("dropout", 0.05),
            target_modules=lora_config.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]),
            bias=lora_config.get("bias", "none"),
            use_gradient_checkpointing=lora_config.get(
                "use_gradient_checkpointing", "unsloth"
            ),
            random_state=42,
        )
        
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        pct = 100 * trainable / total
        console.print(f"[green]✓ LoRA applied — Trainable: {trainable:,} / {total:,} ({pct:.2f}%)[/green]")
        print_gpu_info()
    
    return model, tokenizer


def save_checkpoint(model, tokenizer, output_dir: str, tag: str = "final") -> None:
    """Save LoRA adapter checkpoint."""
    save_path = Path(output_dir) / tag
    save_path.mkdir(parents=True, exist_ok=True)
    console.print(f"\n[blue]Saving checkpoint to {save_path}...[/blue]")
    model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    console.print(f"[green]✓ Checkpoint saved[/green]")


def merge_and_export(model, tokenizer, output_dir: str, gguf: bool = True, gguf_quant: str = "q4_k_m") -> None:
    """Merge LoRA adapters and optionally export GGUF."""
    merged_path = Path(output_dir)
    merged_path.mkdir(parents=True, exist_ok=True)
    
    console.print(f"\n[bold blue]Merging adapters...[/bold blue]")
    model.save_pretrained_merged(str(merged_path), tokenizer, save_method="merged_16bit")
    console.print(f"[green]✓ Merged model saved to {merged_path}[/green]")
    
    if gguf:
        console.print(f"\n[bold blue]Exporting GGUF ({gguf_quant})...[/bold blue]")
        try:
            model.save_pretrained_gguf(str(merged_path / "gguf"), tokenizer, quantization_method=gguf_quant)
            console.print(f"[green]✓ GGUF exported[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠ GGUF export failed: {e}[/yellow]")
            console.print(f"[yellow]  You can export manually later with llama.cpp[/yellow]")
