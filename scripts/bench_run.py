#!/usr/bin/env python3
"""Benchmark runner script."""

import argparse
import json
import os
import sys
import time
import yaml
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bench.benchcore.utils.environment import (
    set_random_seeds, capture_environment, generate_run_id
)
from bench.benchcore.eval.metrics import (
    ClassificationMetrics, DetectionMetrics, PerformanceMetrics
)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_technique(technique_name: str, technique_config: Dict[str, Any]):
    """Load a technique by name."""
    if technique_name == "baseline":
        from bench.benchcore.techniques.baseline.technique import BaselineTechnique
        return BaselineTechnique(**technique_config)
    elif technique_name == "improved":
        from bench.benchcore.techniques.improved.technique import ImprovedTechnique
        return ImprovedTechnique(**technique_config)
    elif technique_name == "bg_removal":
        from bench.benchcore.techniques.improved.technique import BackgroundRemovalTechnique
        return BackgroundRemovalTechnique(**technique_config)
    elif technique_name == "glare_removal":
        from bench.benchcore.techniques.improved.technique import GlareRemovalTechnique
        return GlareRemovalTechnique(**technique_config)
    elif technique_name == "region_growing":
        from bench.benchcore.techniques.improved.technique import RegionGrowingTechnique
        return RegionGrowingTechnique(**technique_config)
    elif technique_name == "hog_enhancement":
        from bench.benchcore.techniques.improved.technique import HOGEnhancementTechnique
        return HOGEnhancementTechnique(**technique_config)
    else:
        raise ValueError(f"Unknown technique: {technique_name}")


def load_dataset_from_manifest(dataset_name: str, max_images: int = None) -> tuple:
    """
    Load dataset from manifest file.
    
    Args:
        dataset_name: Name of dataset to load
        max_images: Maximum number of images to load (None for all)
    
    Returns:
        Tuple of (images, ground_truths, image_ids)
    """
    import numpy as np
    import cv2
    import json
    
    manifest_file = Path(f"data/processed/{dataset_name}/manifest.jsonl")
    
    images = []
    ground_truths = []
    image_ids = []
    
    if manifest_file.exists():
        print(f"Loading dataset {dataset_name} from manifest...")
        
        with open(manifest_file, 'r') as f:
            entries = [json.loads(line) for line in f]
        
        # Limit number of images if specified
        if max_images:
            entries = entries[:max_images]
        
        for entry in entries:
            image_path = Path("data/raw") / entry['image_path']
            
            try:
                img = cv2.imread(str(image_path))
                if img is not None:
                    images.append(img)
                    image_ids.append(entry['id'])
                    
                    # Convert labels to ground truth format
                    gt_labels = entry.get('gt_labels', [])
                    ground_truths.append(gt_labels)
                else:
                    print(f"Warning: Could not load image {image_path}")
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
        
        print(f"Loaded {len(images)} images from {dataset_name}")
        return images, ground_truths, image_ids
    
    else:
        print(f"Manifest not found for {dataset_name}: {manifest_file}")
        return load_dummy_dataset({})


def load_dummy_dataset(dataset_config: dict[str, Any]) -> tuple:
    """
    Load a dummy dataset for testing.
    
    Returns:
        Tuple of (images, ground_truths, image_ids)
    """
    import numpy as np
    import cv2
    
    # For demo purposes, create some synthetic test images
    images = []
    ground_truths = []
    image_ids = []
    
    # Load actual test images if they exist
    input_dir = Path("input")
    if input_dir.exists():
        image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
        
        for i, img_file in enumerate(image_files[:5]):  # Limit to 5 for demo
            try:
                img = cv2.imread(str(img_file))
                if img is not None:
                    images.append(img)
                    image_ids.append(f"{img_file.stem}_{i:03d}")
                    
                    # Create dummy ground truth (no defects for simplicity)
                    ground_truths.append([])
            except Exception as e:
                print(f"Failed to load {img_file}: {e}")
    
    # If no images found, create synthetic ones
    if not images:
        print("No test images found, creating synthetic images...")
        for i in range(3):
            # Create a simple synthetic image
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            images.append(img)
            image_ids.append(f"synthetic_{i:03d}")
            ground_truths.append([])  # No defects in synthetic images
    
    print(f"Loaded {len(images)} test images")
    return images, ground_truths, image_ids


def run_benchmark(experiment_config: Dict[str, Any], run_id: str) -> Dict[str, Any]:
    """
    Run a benchmark experiment.
    
    Args:
        experiment_config: Experiment configuration
        run_id: Unique run identifier
        
    Returns:
        Benchmark results
    """
    # Set up results directory
    results_dir = Path("results") / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Capture environment
    print("📋 Capturing environment...")
    env_info = capture_environment()
    with open(results_dir / "environment.json", 'w') as f:
        json.dump(env_info, f, indent=2)
    
    # Set random seeds
    seed = experiment_config.get('seed', 42)
    set_random_seeds(seed)
    print(f"🎲 Set random seed to {seed}")
    
    # Load dataset(s)
    dataset_name = experiment_config.get('dataset', 'demo_dataset')
    dataset_config = experiment_config.get('dataset_config', {})
    max_images = experiment_config.get('max_images_per_dataset', None)
    
    print(f"📊 Loading dataset: {dataset_name}")
    
    # Try loading from manifest first, fallback to dummy data
    images, ground_truths, image_ids = load_dataset_from_manifest(dataset_name, max_images)
    
    if not images:
        print("Falling back to dummy dataset...")
        images, ground_truths, image_ids = load_dummy_dataset(dataset_config)
    
    # Load techniques
    techniques = []
    technique_names = experiment_config['techniques']
    
    for tech_name in technique_names:
        tech_config = experiment_config.get('technique_configs', {}).get(tech_name, {})
        print(f"🔧 Loading technique: {tech_name}")
        technique = load_technique(tech_name, tech_config)
        technique.setup()
        techniques.append(technique)
    
    # Run benchmarks
    all_results = []
    summary_data = []
    
    for technique in techniques:
        print(f"🚀 Running technique: {technique.name}")
        
        start_time = time.time()
        results = technique.predict_batch(images)
        total_time = time.time() - start_time
        
        # Extract latencies
        latencies = [r.get('latency_ms', 0.0) for r in results]
        
        # Calculate performance metrics
        perf_metrics = PerformanceMetrics.calculate(latencies)
        
        # For now, we'll skip detailed accuracy metrics since we don't have proper ground truth
        # In a real implementation, this would calculate mAP, F1, etc.
        
        # Create summary entry
        summary_entry = {
            'run_id': run_id,
            'timestamp': env_info['timestamp'],
            'git_commit': env_info['git']['commit'],
            'git_branch': env_info['git']['branch'],
            'dataset': dataset_name,
            'technique': technique.name,
            'num_images': len(images),
            'total_detections': sum(r.get('detection_count', 0) for r in results),
            'seed': seed,
            **perf_metrics
        }
        
        summary_data.append(summary_entry)
        
        # Store detailed results
        technique_results = {
            'technique': technique.name,
            'summary': summary_entry,
            'per_image_results': results
        }
        all_results.append(technique_results)
        
        print(f"  ✅ Processed {len(images)} images in {total_time:.2f}s")
        print(f"  📊 Found {summary_entry['total_detections']} total detections")
        print(f"  ⚡ {perf_metrics['images_per_second']:.2f} images/sec")
    
    # Save results
    print("💾 Saving results...")
    
    # Save summary CSV
    import pandas as pd
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(results_dir / "summary.csv", index=False)
    
    # Save per-image results as JSONL
    with open(results_dir / "per_image.jsonl", 'w') as f:
        for result_set in all_results:
            for img_result in result_set['per_image_results']:
                img_result['technique'] = result_set['technique']
                f.write(json.dumps(img_result) + '\n')
    
    # Save complete results
    with open(results_dir / "results.json", 'w') as f:
        json.dump({
            'experiment_config': experiment_config,
            'environment': env_info,
            'results': all_results,
            'summary': summary_data
        }, f, indent=2)
    
    # Generate report
    print("📊 Generating report...")
    try:
        from bench.benchcore.viz.report import generate_benchmark_report
        report_files = generate_benchmark_report(str(results_dir))
        print(f"📄 Report generated: {report_files['report_md']}")
        print(f"📊 Generated {len(report_files['plots'])} plots")
    except Exception as e:
        print(f"⚠️  Report generation failed: {e}")
    
    # Cleanup techniques
    for technique in techniques:
        technique.teardown()
    
    print(f"✅ Benchmark complete! Results saved to: {results_dir}")
    
    return {
        'run_id': run_id,
        'results_dir': str(results_dir),
        'summary': summary_data,
        'num_images': len(images),
        'num_techniques': len(techniques)
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run VIP benchmarking experiment")
    parser.add_argument('--config', '-c', required=True, 
                       help='Path to experiment configuration file')
    parser.add_argument('--run-id', default='auto',
                       help='Run ID (use "auto" to generate automatically)')
    parser.add_argument('--output-dir', default='results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Load experiment configuration
    if not os.path.exists(args.config):
        print(f"❌ Configuration file not found: {args.config}")
        sys.exit(1)
    
    experiment_config = load_config(args.config)
    
    # Generate run ID if needed
    if args.run_id == 'auto':
        run_id = generate_run_id("bench")
    else:
        run_id = args.run_id
    
    print(f"🎯 Starting benchmark run: {run_id}")
    print(f"📋 Config: {args.config}")
    
    try:
        results = run_benchmark(experiment_config, run_id)
        
        # Update run registry
        registry_file = "bench_runs.jsonl"
        registry_entry = {
            'run_id': run_id,
            'timestamp': capture_environment()['timestamp'],
            'config_file': args.config,
            'git_commit': capture_environment()['git']['commit'],
            'git_branch': capture_environment()['git']['branch'],
            'results_dir': results['results_dir'],
            'num_images': results['num_images'],
            'num_techniques': results['num_techniques']
        }
        
        with open(registry_file, 'a') as f:
            f.write(json.dumps(registry_entry) + '\n')
        
        print(f"📝 Updated run registry: {registry_file}")
        print(f"🎉 Benchmark completed successfully!")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
