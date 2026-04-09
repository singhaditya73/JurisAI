"""
JurisAI - Stage 2: Instruction Fine-Tuning (SFT)
Teaches the model legal Q&A behavior using ChatML-formatted instruction pairs.

This is the CORE training step. Uses completion-only loss (trains only on
assistant responses, masking user prompts).

Usage:
    python -m src.training.finetune
    python -m src.training.finetune --from-pretrained ./models/adapters/pretrain_v1/final
    python -m src.training.finetune --export-gguf
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import unsloth FIRST for optimizations
import unsloth

from rich.console import Console

console = Console()


def formatting_func(examples, tokenizer, system_prompt: str = None):
    """Format examples into chat template strings for the trainer.
    
    Handles both:
    - Pre-formatted {"messages": [...]} format
    - Raw {"instruction": ..., "input": ..., "output": ...} format
    """
    texts = []
    
    for i in range(len(examples.get("messages", examples.get("instruction", [])))):
        # Check if already in messages format
        if "messages" in examples:
            messages = examples["messages"][i]
        else:
            # Convert from instruction format
            instruction = examples["instruction"][i]
            inp = examples.get("input", [""] * len(examples["instruction"]))[i]
            output = examples["output"][i]
            
            user_msg = f"{instruction}\n\n{inp}".strip() if inp else instruction
            
            messages = [
                {"role": "system", "content": system_prompt or "You are JurisAI, an expert Indian legal assistant."},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": output},
            ]
        
        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)
    
    return texts


def run_finetuning(
    pretrained_adapter: Optional[str] = None,
    export_gguf: bool = False,
) -> str:
    """Run Stage 2: Instruction Fine-Tuning.
    
    Args:
        pretrained_adapter: Path to Stage 1 adapter (optional, for continued training)
        export_gguf: Whether to export merged model as GGUF
    
    Returns:
        Path to the saved adapter directory.
    """
    import json
    from datasets import load_dataset, DatasetDict
    from trl import SFTTrainer, SFTConfig
    from unsloth import FastLanguageModel
    
    from src.data.data_utils import load_config
    from src.training.train_utils import (
        load_model_and_tokenizer, print_gpu_info,
        clear_gpu_memory, save_checkpoint, merge_and_export
    )
    
    console.print("\n[bold cyan]═══════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]  JurisAI - Stage 2: Instruction Fine-Tuning[/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════[/bold cyan]")
    
    # Load configs
    model_config = load_config("model_config.yaml")
    train_config = load_config("training_config.yaml")
    data_config = load_config("data_config.yaml")
    
    lora_config = train_config["lora"]
    sft_args = train_config["instruct"]
    export_config = train_config.get("export", {})
    output_dir = sft_args["output_dir"]
    
    # Step 1: Load model
    console.print("\n[bold]Step 1: Loading model...[/bold]")
    clear_gpu_memory()
    
    if pretrained_adapter and Path(pretrained_adapter).exists():
        console.print(f"[blue]  Loading from pretrained adapter: {pretrained_adapter}[/blue]")
        # Load base + apply pretrained LoRA, then add new LoRA on top
        # For simplicity with Unsloth, load base fresh and apply LoRA
        # The pretrained adapter will have already adapted the weights
        model_config_copy = model_config.copy()
        model_config_copy["base_model"] = model_config["base_model"].copy()
        model_config_copy["base_model"]["name"] = pretrained_adapter
        model, tokenizer = load_model_and_tokenizer(model_config_copy, lora_config)
    else:
        if pretrained_adapter:
            console.print(f"[yellow]⚠ Pretrained adapter not found: {pretrained_adapter}[/yellow]")
            console.print("[yellow]  Training from base model instead[/yellow]")
        model, tokenizer = load_model_and_tokenizer(model_config, lora_config)
    
    # Step 2: Load instruction data
    console.print("\n[bold]Step 2: Loading instruction dataset...[/bold]")
    
    # Try formatted data first, then raw instruction data
    formatted_dir = PROJECT_ROOT / "data" / "processed" / "instruct" / "formatted"
    raw_dir = PROJECT_ROOT / "data" / "processed" / "instruct"
    
    train_path = None
    val_path = None
    
    # Check for formatted data
    if (formatted_dir / "train_formatted.jsonl").exists():
        train_path = str(formatted_dir / "train_formatted.jsonl")
        val_path = str(formatted_dir / "validation_formatted.jsonl")
        console.print("[green]  Using pre-formatted ChatML data[/green]")
    elif (raw_dir / "train.jsonl").exists():
        train_path = str(raw_dir / "train.jsonl")
        val_path = str(raw_dir / "validation.jsonl")
        console.print("[yellow]  Using raw instruction data (will format on-the-fly)[/yellow]")
    else:
        console.print("[red]✗ No instruction data found![/red]")
        console.print("[yellow]  Run the data pipeline first:[/yellow]")
        console.print("[yellow]    python -m src.data.download_datasets[/yellow]")
        console.print("[yellow]    python -m src.data.preprocess[/yellow]")
        console.print("[yellow]    python -m src.data.prepare_instruct[/yellow]")
        return ""
    
    train_dataset = load_dataset("json", data_files=train_path, split="train")
    console.print(f"[green]✓ Training set: {len(train_dataset):,} examples[/green]")
    
    eval_dataset = None
    if val_path and Path(val_path).exists():
        eval_dataset = load_dataset("json", data_files=val_path, split="train")
        console.print(f"[green]✓ Validation set: {len(eval_dataset):,} examples[/green]")
    
    # Step 3: Configure trainer  
    console.print("\n[bold]Step 3: Configuring SFT trainer...[/bold]")
    
    # Get system prompt from config
    inst_config = data_config.get("instruction_format", {})
    system_prompt = inst_config.get("system_prompt", "You are JurisAI, an expert Indian legal assistant.")
    
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=sft_args.get("num_train_epochs", 3),
        per_device_train_batch_size=sft_args.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=sft_args.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=sft_args.get("gradient_accumulation_steps", 8),
        learning_rate=sft_args.get("learning_rate", 2e-4),
        lr_scheduler_type=sft_args.get("lr_scheduler_type", "cosine"),
        warmup_steps=sft_args.get("warmup_steps", 50),
        weight_decay=sft_args.get("weight_decay", 0.01),
        max_grad_norm=sft_args.get("max_grad_norm", 1.0),
        fp16=False,
        bf16=True,
        logging_steps=sft_args.get("logging_steps", 10),
        save_steps=sft_args.get("save_steps", 100),
        save_total_limit=sft_args.get("save_total_limit", 3),
        eval_strategy=sft_args.get("eval_strategy", "steps") if eval_dataset else "no",
        eval_steps=sft_args.get("eval_steps", 50) if eval_dataset else None,
        load_best_model_at_end=sft_args.get("load_best_model_at_end", True) if eval_dataset else False,
        metric_for_best_model=sft_args.get("metric_for_best_model", "eval_loss") if eval_dataset else None,
        greater_is_better=sft_args.get("greater_is_better", False),
        neftune_noise_alpha=sft_args.get("neftune_noise_alpha", 5),
        dataloader_num_workers=sft_args.get("dataloader_num_workers", 2),
        dataloader_pin_memory=sft_args.get("dataloader_pin_memory", True),
        max_seq_length=sft_args.get("max_seq_length", 2048),
        packing=sft_args.get("packing", False),
        report_to="none",
    )
    
    # Convert all data to text field using chat template
    def _format_row(row):
        # Handle 'messages' format (pre-formatted ChatML)
        if "messages" in row:
            messages = row["messages"]
        else:
            # Convert from instruction/input/output format
            instruction = row.get("instruction", "")
            inp = row.get("input", "")
            output = row.get("output", "")
            user_msg = f"{instruction}\n\n{inp}".strip() if inp else instruction
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": output},
            ]
        
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        return {"text": text}
    
    console.print("  Formatting data with chat template...")
    train_dataset = train_dataset.map(_format_row, remove_columns=train_dataset.column_names)
    if eval_dataset:
        eval_dataset = eval_dataset.map(_format_row, remove_columns=eval_dataset.column_names)
    console.print(f"[green]  ✓ Formatted {len(train_dataset):,} training examples[/green]")
    
    sft_config.dataset_text_field = "text"
    
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
    )
    
    # Step 4: Train!
    console.print("\n[bold]Step 4: Starting instruction fine-tuning...[/bold]")
    console.print(f"  Epochs: {sft_args.get('num_train_epochs', 3)}")
    console.print(f"  Batch size: {sft_args.get('per_device_train_batch_size', 1)}")
    console.print(f"  Grad accumulation: {sft_args.get('gradient_accumulation_steps', 8)}")
    console.print(f"  Effective batch: {sft_args.get('per_device_train_batch_size', 1) * sft_args.get('gradient_accumulation_steps', 8)}")
    console.print(f"  Learning rate: {sft_args.get('learning_rate', 2e-4)}")
    console.print(f"  NEFTune alpha: {sft_args.get('neftune_noise_alpha', 5)}")
    console.print("")
    
    print_gpu_info()
    
    # Check for existing checkpoints to resume from
    resume_from = None
    checkpoint_dirs = sorted(Path(output_dir).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    if checkpoint_dirs:
        resume_from = str(checkpoint_dirs[-1])
        console.print(f"[yellow]  ⚡ Resuming from {resume_from}[/yellow]")
    
    trainer_stats = trainer.train(resume_from_checkpoint=resume_from)
    
    # Step 5: Save adapter
    console.print("\n[bold]Step 5: Saving fine-tuned adapter...[/bold]")
    save_checkpoint(model, tokenizer, output_dir, tag="final")
    
    # Step 6: Export (optional)
    if export_gguf or export_config.get("merge_adapters", False):
        console.print("\n[bold]Step 6: Merging & exporting...[/bold]")
        
        merge_path = export_config.get("save_merged", "./models/merged/jurisai-v1")
        merge_and_export(
            model, tokenizer,
            output_dir=merge_path,
            gguf=export_gguf or export_config.get("gguf_export", False),
            gguf_quant=export_config.get("gguf_quantization", "q4_k_m"),
        )
    
    # Print results
    console.print("\n[bold green]═══════════════════════════════════════════════════[/bold green]")
    console.print("[bold green]  Fine-Tuning Complete! 🎉[/bold green]")
    console.print("[bold green]═══════════════════════════════════════════════════[/bold green]")
    console.print(f"  Training loss: {trainer_stats.training_loss:.4f}")
    console.print(f"  Runtime: {trainer_stats.metrics.get('train_runtime', 0):.0f}s")
    console.print(f"  Samples/sec: {trainer_stats.metrics.get('train_samples_per_second', 0):.2f}")
    console.print(f"  Adapter saved: {output_dir}")
    console.print("")
    console.print("[bold]Next steps:[/bold]")
    console.print("  1. Test: python -m src.inference.generate")
    console.print("  2. Evaluate: python -m src.evaluation.evaluate")
    
    # Cleanup
    clear_gpu_memory()
    
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JurisAI Stage 2: Instruction Fine-Tuning")
    parser.add_argument(
        "--from-pretrained",
        type=str,
        default=None,
        help="Path to Stage 1 pretrained adapter (optional)"
    )
    parser.add_argument(
        "--export-gguf",
        action="store_true",
        help="Export merged model as GGUF after training"
    )
    args = parser.parse_args()
    
    run_finetuning(
        pretrained_adapter=args.from_pretrained,
        export_gguf=args.export_gguf,
    )
