"""Command-line interface for the Vision Inspection Pipeline."""

import json
import os
from pathlib import Path
from typing import List, Optional

import cv2
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .config import RunConfig
from .io.loader import iter_images
from .io.writer import save_results
from .pipeline import Pipeline

app = typer.Typer(help="Vision Inspection Pipeline for defect detection")
console = Console()


@app.command()
def run(
    input_dir: str = typer.Option("input", "--input", "-i", help="Input directory containing images"),
    output_dir: str = typer.Option("output", "--output", "-o", help="Output directory for results"),
    defects: str = typer.Option("scratches,contamination,discoloration,cracks", "--defects", "-d", 
                               help="Comma-separated list of defect types to detect"),
    resize_width: Optional[int] = typer.Option(None, "--resize-width", "-w", 
                                              help="Optional width to resize images to"),
    save_overlay: bool = typer.Option(True, "--save-overlay/--no-save-overlay", 
                                    help="Whether to save overlay images"),
    save_json: bool = typer.Option(True, "--save-json/--no-save-json", 
                                 help="Whether to save JSON results")
):
    """Run defect detection on images in the input directory."""
    
    # Parse defect types
    defect_list = [d.strip() for d in defects.split(",")]
    
    # Create configuration
    config = RunConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        defects=defect_list,
        resize_width=resize_width,
        save_overlay=save_overlay,
        save_json=save_json
    )
    
    console.print(f"[bold blue]Running VIP with configuration:[/bold blue]")
    console.print(f"  Input: {config.input_dir}")
    console.print(f"  Output: {config.output_dir}")
    console.print(f"  Defects: {', '.join(config.defects)}")
    if config.resize_width:
        console.print(f"  Resize width: {config.resize_width}")
    
    # Initialize pipeline
    try:
        pipeline = Pipeline(config)
        console.print(f"[green]✓[/green] Pipeline initialized with {len(pipeline.detectors)} detectors")
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to initialize pipeline: {e}")
        raise typer.Exit(1)
    
    # Process images
    image_count = 0
    total_detections = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task("Processing images...", total=None)
        
        try:
            for stem, image in iter_images(config.input_dir):
                progress.update(task, description=f"Processing {stem}...")
                
                # Process image
                detections, overlay_image = pipeline.process_image(image, stem)
                
                # Save results
                save_results(
                    config.output_dir,
                    stem,
                    detections,
                    overlay_image,
                    config.save_overlay,
                    config.save_json
                )
                
                image_count += 1
                total_detections += len(detections)
                
                progress.update(task, description=f"Processed {stem} ({len(detections)} defects)")
            
            progress.update(task, description="Complete!")
            
        except Exception as e:
            console.print(f"[red]✗[/red] Error processing images: {e}")
            raise typer.Exit(1)
    
    # Summary
    console.print(f"\n[bold green]Processing complete![/bold green]")
    console.print(f"  Images processed: {image_count}")
    console.print(f"  Total defects detected: {total_detections}")
    console.print(f"  Results saved to: {config.output_dir}")


@app.command()
def eval_simple(
    dataset_root: str = typer.Option("data/datasets/kaggle_tarros/1_front/test", "--dataset-root", 
                                    help="Path to Kaggle dataset test directory")
):
    """Evaluate detection performance on Kaggle Tarros dataset."""
    
    console.print(f"[bold blue]Evaluating on dataset:[/bold blue] {dataset_root}")
    
    # Check dataset structure
    dataset_path = Path(dataset_root)
    true_dir = dataset_path / "1_true"
    false_dir = dataset_path / "2_false"
    
    if not true_dir.exists() or not false_dir.exists():
        console.print(f"[red]✗[/red] Invalid dataset structure. Expected:")
        console.print(f"  {dataset_root}/")
        console.print(f"    ├── 1_true/     (defective images)")
        console.print(f"    └── 2_false/    (non-defective images)")
        raise typer.Exit(1)
    
    # Count images
    true_images = list(true_dir.glob("*.jpg")) + list(true_dir.glob("*.png"))
    false_images = list(false_dir.glob("*.jpg")) + list(false_dir.glob("*.png"))
    
    console.print(f"  Defective images: {len(true_images)}")
    console.print(f"  Non-defective images: {len(false_images)}")
    
    if not true_images or not false_images:
        console.print(f"[red]✗[/red] No images found in dataset directories")
        raise typer.Exit(1)
    
    # Initialize pipeline with all detectors
    config = RunConfig(
        defects=["scratches", "contamination", "discoloration", "cracks", "flash"],
        resize_width=1024  # Resize for consistent processing
    )
    
    try:
        pipeline = Pipeline(config)
        console.print(f"[green]✓[/green] Pipeline initialized")
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to initialize pipeline: {e}")
        raise typer.Exit(1)
    
    # Evaluation metrics
    tp = 0  # True positives: defective image with detections
    fp = 0  # False positives: non-defective image with detections
    tn = 0  # True negatives: non-defective image without detections
    fn = 0  # False negatives: defective image without detections
    
    # Process defective images
    console.print("\n[bold]Processing defective images...[/bold]")
    for img_path in true_images:
        try:
            image = cv2.imread(str(img_path))
            if image is not None:
                detections, _ = pipeline.process_image(image, img_path.stem)
                if detections:
                    tp += 1
                else:
                    fn += 1
        except Exception as e:
            console.print(f"Warning: Error processing {img_path.name}: {e}")
            continue
    
    # Process non-defective images
    console.print("\n[bold]Processing non-defective images...[/bold]")
    for img_path in false_images:
        try:
            image = cv2.imread(str(img_path))
            if image is not None:
                detections, _ = pipeline.process_image(image, img_path.stem)
                if detections:
                    fp += 1
                else:
                    tn += 1
        except Exception as e:
            console.print(f"Warning: Error processing {img_path.name}: {e}")
            continue
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Display results
    results = {
        "dataset": dataset_root,
        "metrics": {
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4)
        },
        "total_images": len(true_images) + len(false_images)
    }
    
    # Create results table
    table = Table(title="Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("True Positives", str(tp))
    table.add_row("False Positives", str(fp))
    table.add_row("True Negatives", str(tn))
    table.add_row("False Negatives", str(fn))
    table.add_row("Precision", f"{precision:.4f}")
    table.add_row("Recall", f"{recall:.4f}")
    table.add_row("F1 Score", f"{f1:.4f}")
    
    console.print(table)
    
    # Save results to JSON
    output_file = f"evaluation_results_{dataset_path.name}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    console.print(f"\n[green]✓[/green] Results saved to {output_file}")


@app.command()
def list_detectors():
    """List all available defect detectors."""
    
    config = RunConfig()
    pipeline = Pipeline(config)
    
    detectors = pipeline.list_available_detectors()
    
    console.print(f"[bold blue]Available detectors:[/bold blue]")
    for detector in detectors:
        console.print(f"  • {detector}")


if __name__ == "__main__":
    app()
