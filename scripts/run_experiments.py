#!/usr/bin/env python3
"""
Phase 4: Experiment Engine - CLI Entry Point
Run reproducible experiments comparing SAF-RAG vs baselines
"""

import sys
import argparse
import logging
from pathlib import Path

import torch
from src.phase4 import ExperimentRunner, ExperimentRegistry

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SAF-RAG Experiment Engine: Run reproducible benchmark experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full experiment from config
  python scripts/run_experiments.py --config configs/saf_rag_experiment.yaml
  
  # Quick validation (small sample, no plots)
  python scripts/run_experiments.py --config configs/saf_rag_experiment.yaml --quick
  
  # Run specific models only
  python scripts/run_experiments.py --config configs/saf_rag_experiment.yaml --models saf-rag,vanilla-rag
  
  # Verbose output
  python scripts/run_experiments.py --config configs/saf_rag_experiment.yaml --verbose
  
  # List all experiments
  python scripts/run_experiments.py --list-experiments
        """
    )
    
    # Config file
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file for experiments"
    )
    
    # Model selection
    parser.add_argument(
        "--models",
        type=str,
        help="Comma-separated list of models to run (default: all)"
    )
    
    # Dataset selection
    parser.add_argument(
        "--datasets",
        type=str,
        help="Comma-separated list of datasets to use"
    )
    
    # Flags
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick validation run (small sample sizes, no plots)"
    )
    
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip plot generation"
    )
    
    parser.add_argument(
        "--skip-tables",
        action="store_true",
        help="Skip table generation"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging output"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device to use (default: cuda)"
    )
    
    # List experiments
    parser.add_argument(
        "--list-experiments",
        action="store_true",
        help="List all available experiments"
    )
    
    parser.add_argument(
        "--list-ablations",
        action="store_true",
        help="List all ablation experiments"
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results",
        help="Directory to save results (default: experiments/results)"
    )
    
    args = parser.parse_args()
    
    # List experiments mode
    if args.list_experiments:
        registry = ExperimentRegistry()
        print("\nAvailable Experiments:")
        print("=" * 70)
        for name in registry.list_experiments():
            config = registry.get_experiment(name)
            print(f"  {name:30s} — {config.description}")
        print("=" * 70)
        return 0
    
    if args.list_ablations:
        registry = ExperimentRegistry()
        print("\nAblation Experiments:")
        print("=" * 70)
        for name in registry.list_ablations():
            config = registry.get_experiment(name)
            print(f"  {name:30s} — {config.description}")
        print("=" * 70)
        return 0
    
    # Run experiments mode
    if not args.config:
        print("Error: --config is required (unless using --list-experiments)")
        parser.print_help()
        return 1
    
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return 1
    
    try:
        # Load config
        logger.info(f"Loading config: {config_path}")
        runner = ExperimentRunner(config_path)
        
        # Apply CLI overrides
        if args.quick:
            runner.config.sample_size = 20
            runner.config.save_plots = False
            runner.config.device = "cpu"
            logger.info("Quick mode: Using sample_size=20, skipping plots, running on CPU")
        
        if args.skip_plots:
            runner.config.save_plots = False
        
        if args.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available; falling back to CPU")
            runner.config.device = "cpu"
        
        if args.skip_tables:
            runner.config.save_tables = False
        
        if args.device:
            runner.config.device = args.device
        
        if args.verbose:
            runner.config.verbose = True
            logging.getLogger().setLevel(logging.DEBUG)
        
        runner.config.results_dir = Path(args.output_dir)
        
        # Parse models
        models = None
        if args.models:
            models = [m.strip() for m in args.models.split(",")]
            logger.info(f"Running specific models: {models}")
        
        # Run experiments
        logger.info("Starting experiment run...")
        results = runner.run_all_experiments(models=models)
        logger.info(f"Completed {len(results)} experiments")
        
        # Generate report
        logger.info("Generating report...")
        report = runner.generate_report(save=True)
        
        # Print summary
        print("\n" + "=" * 70)
        print("EXPERIMENT SUMMARY")
        print("=" * 70)
        print(f"Experiments run: {len(results)}")
        print(f"Datasets: {len(runner.config.datasets)}")
        print(f"Output directory: {runner.config.results_dir}")
        print()
        
        # Print aggregated results
        print("Aggregated Results:")
        for model_name in sorted(results.keys()):
            result = results[model_name]
            if "aggregated" in result:
                agg = result["aggregated"]
                print(f"\n  {model_name}:")
                for metric, value in agg.items():
                    if isinstance(value, (int, float)):
                        print(f"    {metric:5s}: {value:.3f}")
        
        print("\n" + "=" * 70)
        logger.info("Experiment run completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
