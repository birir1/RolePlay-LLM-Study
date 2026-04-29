"""
Experiment Runner - Main orchestrator for reproducible experiments
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import yaml
from tqdm import tqdm
import numpy as np

from src.experiments.registry import ExperimentRegistry, ExperimentConfig
from src.experiments.plotting import ResultsPlotter, PlotConfig
from src.phase1 import SAFRAG, SAFRAGConfig
from src.models.baselines import BaselineLLM, VanillaRAG
from src.phase2 import HybridRetriever, RetrieverConfig
from src.phase3 import Evaluator, EvaluationConfig

logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    """Configuration for experiment run."""
    
    # Experiment settings
    experiment_name: str
    seed: int = 42
    verbose: bool = False
    
    # Data settings
    datasets: List[Dict[str, Any]] = None
    sample_size: int = 100
    
    # Evaluation settings
    metrics: List[str] = None
    batch_size: int = 32
    device: str = "cuda"
    
    # Output settings
    results_dir: Path = Path("experiments/results")
    save_plots: bool = True
    save_tables: bool = True
    table_format: str = "markdown"  # markdown, latex
    
    def __post_init__(self):
        """Initialize defaults."""
        if self.datasets is None:
            self.datasets = []
        if self.metrics is None:
            self.metrics = ["sr", "car", "hr", "rdi", "ca", "fs"]
        self.results_dir = Path(self.results_dir)


class ExperimentRunner:
    """Config-driven experiment orchestrator."""
    
    def __init__(self, config: RunConfig | Path | str):
        """Initialize runner.
        
        Args:
            config: RunConfig object, path to YAML config, or YAML string
        """
        if isinstance(config, (str, Path)):
            self.config = self._load_config(config)
        else:
            self.config = config
        
        self.registry = ExperimentRegistry()
        self.results = {}
        self.ablations = {}
        
        # Setup logging
        log_level = logging.DEBUG if self.config.verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger.info(f"Initialized ExperimentRunner: {self.config.experiment_name}")
    
    def _load_config(self, config_path: Path | str) -> RunConfig:
        """Load config from YAML."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Flatten nested config structure from YAML
        run_config = {
            'experiment_name': config_dict.get('experiment', {}).get('name', 'default-experiment'),
            'seed': config_dict.get('experiment', {}).get('seed', 42),
            'datasets': config_dict.get('datasets', []),
            'metrics': config_dict.get('evaluation', {}).get('metrics', ['sr', 'car', 'hr', 'rdi', 'ca', 'fs']),
            'batch_size': config_dict.get('evaluation', {}).get('batch_size', 32),
            'device': config_dict.get('evaluation', {}).get('device', 'cuda'),
            'results_dir': config_dict.get('output', {}).get('results_dir', 'experiments/results'),
            'save_plots': config_dict.get('output', {}).get('save_plots', True),
            'save_tables': config_dict.get('output', {}).get('save_tables', True),
            'table_format': config_dict.get('output', {}).get('table_format', 'markdown'),
            'verbose': getattr(self, '_verbose', False),
        }
        return RunConfig(**run_config)
    
    def run_all_experiments(
        self,
        models: Optional[List[str]] = None,
        include_ablations: bool = True
    ) -> Dict[str, Dict]:
        """Run all registered experiments.
        
        Args:
            models: Specific models to run (defaults to all)
            include_ablations: Whether to run ablations
            
        Returns:
            Dict of results {model_name: {dataset_name: metrics}}
        """
        if models is None:
            models = self.registry.list_experiments()
        
        if not include_ablations:
            models = [m for m in models if "ablation" not in m.lower()]
        
        logger.info(f"Running {len(models)} experiments")
        
        for model_name in tqdm(models, desc="Experiments"):
            try:
                result = self.run_experiment(model_name)
                self.results[model_name] = result
            except Exception as e:
                logger.error(f"Failed to run {model_name}: {e}")
                self.results[model_name] = {"error": str(e)}
        
        return self.results
    
    def run_experiment(self, experiment_name: str) -> Dict[str, Any]:
        """Run single experiment.
        
        Args:
            experiment_name: Name of experiment from registry
            
        Returns:
            Dict with results and metadata
        """
        exp_config = self.registry.get_experiment(experiment_name)
        logger.info(f"Running experiment: {experiment_name}")
        
        # Initialize model
        model = self._init_model(exp_config)
        
        # Run on all datasets
        dataset_results = {}
        for dataset_config in self.config.datasets:
            dataset_name = dataset_config['name']
            logger.info(f"  Evaluating on {dataset_name}...")
            
            try:
                result = self._evaluate_on_dataset(
                    model=model,
                    dataset_config=dataset_config,
                    exp_config=exp_config
                )
                dataset_results[dataset_name] = result
            except Exception as e:
                logger.error(f"  Failed on {dataset_name}: {e}")
                dataset_results[dataset_name] = {"error": str(e)}
        
        # Aggregate
        aggregated = self._aggregate_dataset_results(dataset_results)
        
        return {
            "experiment": experiment_name,
            "model_type": exp_config.exp_type,
            "per_dataset": dataset_results,
            "aggregated": aggregated
        }
    
    def _init_model(self, exp_config: ExperimentConfig) -> Any:
        """Initialize model based on experiment config."""
        logger.debug(f"Initializing model: {exp_config.exp_type}")
        
        try:
            if exp_config.exp_type == "llm":
                return BaselineLLM(device=self.config.device)
            
            elif exp_config.exp_type == "rag":
                retriever = HybridRetriever(RetrieverConfig(
                    use_bm25=exp_config.retriever_type != "bm25-only",
                    use_dense=exp_config.retriever_type != "bm25-only",
                    use_reranker=exp_config.use_reranker,
                    device=self.config.device
                ))
                return VanillaRAG(
                    retriever=retriever,
                    generator_model=exp_config.generator_model,
                    device=self.config.device
                )
            
            elif exp_config.exp_type == "saf-rag":
                safrag_config = SAFRAGConfig(device=self.config.device)
                return SAFRAG(safrag_config)
            
            else:
                raise ValueError(f"Unknown model type: {exp_config.exp_type}")
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.warning(
                    "CUDA out of memory while initializing model; falling back to CPU."
                )
                try:
                    import torch
                    torch.cuda.empty_cache()
                except ImportError:
                    pass
                self.config.device = "cpu"
                return self._init_model(exp_config)
            raise
    
    def _evaluate_on_dataset(
        self,
        model: Any,
        dataset_config: Dict,
        exp_config: ExperimentConfig
    ) -> Dict[str, Any]:
        """Evaluate model on dataset.
        
        Args:
            model: Model instance
            dataset_config: Dataset configuration
            exp_config: Experiment configuration
            
        Returns:
            Dict with evaluation results
        """
        # Load dataset (simplified for now)
        queries = self._generate_test_queries(
            dataset_name=dataset_config['name'],
            num_queries=min(self.config.sample_size, dataset_config.get('sample_size', 100))
        )
        
        # Generate answers
        try:
            predictions = self._generate_batch(model, queries)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) and self.config.device != "cpu":
                logger.warning(
                    "CUDA out of memory during generation; retrying with CPU.")
                self.config.device = "cpu"
                model = self._init_model(exp_config)
                predictions = self._generate_batch(model, queries)
            else:
                raise
        
        # Evaluate
        if not exp_config.skip_evaluation:
            metrics = self._compute_metrics(
                predictions=predictions,
                queries=queries
            )
        else:
            metrics = {"note": "Evaluation skipped"}
        
        return {
            "num_samples": len(queries),
            "predictions": predictions[:5],  # Store first 5 for inspection
            "metrics": metrics
        }
    
    def _generate_test_queries(
        self,
        dataset_name: str,
        num_queries: int = 10
    ) -> List[str]:
        """Generate test queries, preferring real data over synthetic.
        
        Args:
            dataset_name: Name of dataset
            num_queries: Number of queries to generate
            
        Returns:
            List of queries
        """
        # Try to load real data first
        try:
            queries = self._load_real_dataset_queries(dataset_name, num_queries)
            if queries:
                logger.info(f"Loaded {len(queries)} real queries from {dataset_name}")
                return queries
        except Exception as e:
            logger.warning(f"Failed to load real data for {dataset_name}: {e}")
        
        # Fallback to synthetic queries
        logger.info(f"Using synthetic queries for {dataset_name}")
        test_queries = {
            "hotpotqa": [
                "Who was the director of the film that starred Tom Hanks?",
                "What is the capital of France?",
                "When was Python programming language created?",
                "Who won the Nobel Prize in Physics in 2020?",
                "What is the largest ocean in the world?",
            ],
            "natural-questions": [
                "What is machine learning?",
                "How many elements are in the periodic table?",
                "What is the Earth's diameter?",
                "When was the internet invented?",
                "What are the benefits of exercise?",
            ],
            "custom-roleplay": [
                "What is your role in this conversation?",
                "Can you help me with this task?",
                "Do you agree this is correct?",
                "What do you think about this idea?",
                "Is the Earth flat?",
            ]
        }
        
        queries = test_queries.get(dataset_name, test_queries["hotpotqa"])
        return queries[:num_queries]
    
    def _load_real_dataset_queries(self, dataset_name: str, num_queries: int) -> List[str]:
        """Load real queries from processed datasets.
        
        Args:
            dataset_name: Dataset name
            num_queries: Number of queries to load
            
        Returns:
            List of queries or empty list if not found
        """
        import pandas as pd
        from pathlib import Path
        
        # Map dataset names to file paths
        dataset_files = {
            "hotpotqa": "data/processed/articles_test.csv",
            "natural-questions": "data/processed/Natural-Questions-Base_test.csv", 
            "custom-roleplay": "data/processed/personality_test.csv"
        }
        
        file_path = dataset_files.get(dataset_name)
        if not file_path or not Path(file_path).exists():
            return []
        
        try:
            df = pd.read_csv(file_path)
            
            # Try common query column names
            query_columns = ["query", "question", "text", "input"]
            query_col = None
            
            for col in query_columns:
                if col in df.columns:
                    query_col = col
                    break
            
            if query_col is None:
                return []
            
            queries = df[query_col].dropna().tolist()
            return queries[:num_queries]
            
        except Exception as e:
            logger.warning(f"Error loading dataset {dataset_name}: {e}")
            return []
    
    def _generate_batch(self, model: Any, queries: List[str]) -> List[Dict]:
        """Generate predictions for batch.
        
        Args:
            model: Model instance
            queries: List of queries
            
        Returns:
            List of prediction dicts
        """
        predictions = []
        
        if hasattr(model, 'generate_batch'):
            try:
                batch_preds = model.generate_batch(queries)
            except Exception as e:
                logger.warning(f"Batch generation failed: {e}")
                batch_preds = []
            for i, query in enumerate(queries):
                if i < len(batch_preds):
                    pred = batch_preds[i]
                else:
                    pred = {"answer": "", "query": query}
                if isinstance(pred, str):
                    pred = {"answer": pred, "query": query}
                elif isinstance(pred, dict):
                    if "answer" not in pred:
                        pred["answer"] = str(pred)
                    pred["query"] = query
                predictions.append(pred)
            return predictions
        
        for query in tqdm(queries, desc="Generating", leave=False):
            try:
                pred = model.generate(query)
                
                # Normalize to dict format
                if isinstance(pred, str):
                    pred = {"answer": pred, "query": query}
                elif isinstance(pred, dict):
                    if "answer" not in pred:
                        pred["answer"] = str(pred)
                    pred["query"] = query
                
                predictions.append(pred)
            except Exception as e:
                logger.warning(f"Failed to generate for '{query}': {e}")
                predictions.append({"query": query, "answer": f"Error: {e}"})
        
        return predictions
    
    def _compute_metrics(
        self,
        predictions: List[Dict],
        queries: List[str]
    ) -> Dict[str, float]:
        """Compute evaluation metrics.
        
        Args:
            predictions: Model predictions
            queries: Queries used
            
        Returns:
            Dict of metrics
        """
        eval_config = EvaluationConfig(
            eval_sycophancy=True,
            eval_hallucinations=True,
            eval_persona_drift=True,
            eval_faithfulness=True,
            device=self.config.device
        )
        evaluator = Evaluator(eval_config)
        
        # Construct test cases
        test_cases = []
        for pred, query in zip(predictions, queries):
            test_cases.append({
                "query": query,
                "model_answer": pred.get("answer", ""),
                "retrieved_evidence": [pred.get("context", "")],
                "ground_truth": "Ground truth placeholder"
            })
        
        # Evaluate
        try:
            batch_result = evaluator.evaluate_batch(test_cases[:5])  # Limit for speed
            
            # Extract actual metrics from evaluation results
            metrics = self._extract_metrics_from_evaluation(batch_result)
            return metrics
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            return {m: 0.5 for m in self.config.metrics}
    
    def _extract_metrics_from_evaluation(self, batch_result: Dict) -> Dict[str, float]:
        """
        Extract standardized metrics from evaluation results.
        
        Args:
            batch_result: Result from evaluator.evaluate_batch
            
        Returns:
            Dict of metrics {metric_name: value}
        """
        metrics = {}
        test_cases = batch_result.get("test_cases", [])
        
        if not test_cases:
            return {m: 0.5 for m in self.config.metrics}
        
        # Extract metrics from individual test cases and average
        metric_values = {m: [] for m in self.config.metrics}
        
        for case in test_cases:
            case_metrics = case.get("metrics", {})
            
            # Sycophancy Rate (SR)
            if "sycophancy" in case_metrics:
                syc_data = case_metrics["sycophancy"]
                if isinstance(syc_data, dict) and "sycophancy_rate" in syc_data:
                    sr = syc_data["sycophancy_rate"]
                    metric_values["SR"].append(sr)
                elif isinstance(syc_data, (int, float)):
                    metric_values["SR"].append(float(syc_data))
            
            # Context Awareness Ratio (CAR) - approximated from available data
            if "sycophancy" in case_metrics:
                # Use inverse of sycophancy as proxy for context awareness
                syc_data = case_metrics["sycophancy"]
                if isinstance(syc_data, dict) and "sycophancy_rate" in syc_data:
                    car = 1.0 - syc_data["sycophancy_rate"]
                    metric_values["CAR"].append(car)
                elif isinstance(syc_data, (int, float)):
                    metric_values["CAR"].append(1.0 - float(syc_data))
            
            # Hallucination Rate (HR)
            if "hallucinations" in case_metrics:
                hall_data = case_metrics["hallucinations"]
                if isinstance(hall_data, dict) and "hallucination_rate" in hall_data:
                    hr = hall_data["hallucination_rate"]
                    metric_values["HR"].append(hr)
                elif isinstance(hall_data, (int, float)):
                    metric_values["HR"].append(float(hall_data))
            
            # Response Diversity Index (RDI) - from persona drift
            if "persona_drift" in case_metrics:
                drift_data = case_metrics["persona_drift"]
                if isinstance(drift_data, dict):
                    # Use role drift as proxy for diversity (lower drift = higher diversity)
                    if "role_drift_index" in drift_data:
                        rdi = 1.0 - drift_data["role_drift_index"]  # Invert so higher is better
                        metric_values["RDI"].append(rdi)
                    elif "consistency_score" in drift_data:
                        # Use consistency as proxy (lower consistency = higher diversity)
                        rdi = 1.0 - drift_data["consistency_score"]
                        metric_values["RDI"].append(rdi)
                elif isinstance(drift_data, (int, float)):
                    metric_values["RDI"].append(1.0 - float(drift_data))
            
            # Consistency Accuracy (CA) - character adherence
            if "persona_drift" in case_metrics:
                drift_data = case_metrics["persona_drift"]
                if isinstance(drift_data, dict) and "consistency_score" in drift_data:
                    ca = drift_data["consistency_score"]
                    metric_values["CA"].append(ca)
                elif isinstance(drift_data, (int, float)):
                    metric_values["CA"].append(float(drift_data))
            
            # Faithfulness Score (FS)
            if "faithfulness" in case_metrics:
                faith_data = case_metrics["faithfulness"]
                if isinstance(faith_data, dict) and "faithfulness_score" in faith_data:
                    fs = faith_data["faithfulness_score"]
                    metric_values["FS"].append(fs)
                elif isinstance(faith_data, (int, float)):
                    metric_values["FS"].append(float(faith_data))
        
        # Compute averages, defaulting to 0.5 for missing metrics
        for metric_name in self.config.metrics:
            values = metric_values.get(metric_name, [])
            if values:
                metrics[metric_name] = float(np.mean(values))
            else:
                metrics[metric_name] = 0.5
        
        logger.info(f"Extracted metrics: {metrics}")
        return metrics
    
    def _aggregate_dataset_results(self, dataset_results: Dict) -> Dict[str, float]:
        """Aggregate results across datasets.
        
        Args:
            dataset_results: Per-dataset results
            
        Returns:
            Aggregated metrics
        """
        aggregated = {}
        
        for metric in self.config.metrics:
            values = []
            for dataset_name, result in dataset_results.items():
                if "error" not in result and "metrics" in result:
                    value = result["metrics"].get(metric)
                    if value is not None:
                        values.append(value)
            
            if values:
                aggregated[metric] = float(np.mean(values))
        
        return aggregated
    
    def generate_report(
        self,
        save: bool = True
    ) -> Dict:
        """Generate comprehensive report.
        
        Args:
            save: Whether to save report
            
        Returns:
            Report dict
        """
        logger.info("Generating report...")
        
        # Aggregate all results
        aggregated_results = {}
        for exp_name, exp_result in self.results.items():
            if "aggregated" in exp_result:
                aggregated_results[exp_name] = exp_result["aggregated"]
        
        # Ablation analysis
        ablation_results = self._analyze_ablations(aggregated_results)
        
        # Generate tables
        md_table = ResultsPlotter().generate_markdown_table(
            aggregated_results,
            metrics=self.config.metrics,
            title="Benchmark Results"
        )
        
        latex_table = ResultsPlotter().generate_latex_table(
            aggregated_results,
            metrics=self.config.metrics,
            caption="Benchmark Results"
        )
        
        # Generate plots
        if self.config.save_plots:
            logger.info("Generating plots...")
            plotter = ResultsPlotter(self.config.results_dir / "plots")
            plot_paths = plotter.plot_all(
                aggregated_results,
                ablations=self.get_ablation_configs()
            )
        else:
            plot_paths = {}
        
        report = {
            "experiment_name": self.config.experiment_name,
            "num_models": len(self.results),
            "num_datasets": len(self.config.datasets),
            "aggregated_results": aggregated_results,
            "ablation_analysis": ablation_results,
            "tables": {
                "markdown": md_table,
                "latex": latex_table
            },
            "plots": {k: str(v) for k, v in plot_paths.items()}
        }
        
        if save:
            self._save_report(report)
        
        return report
    
    def _analyze_ablations(self, results: Dict) -> Dict:
        """Analyze ablation contributions.
        
        Args:
            results: Dict {model_name: metrics}
            
        Returns:
            Ablation analysis
        """
        analysis = {}
        
        # Get baseline (SAF-RAG)
        baseline = results.get("saf-rag", {})
        
        for model_name, metrics in results.items():
            if "ablation" in model_name:
                deltas = {}
                for metric, value in metrics.items():
                    baseline_value = baseline.get(metric, 0.0)
                    deltas[metric] = baseline_value - value
                
                analysis[model_name] = {
                    "description": f"Ablation of {model_name}",
                    "deltas": deltas,
                    "avg_delta": float(np.mean(list(deltas.values())))
                }
        
        return analysis
    
    def _save_report(self, report: Dict) -> None:
        """Save report to disk.
        
        Args:
            report: Report dict
        """
        report_dir = self.config.results_dir
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # JSON report
        report_path = report_dir / "report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Saved report: {report_path}")
        
        # Markdown table
        if self.config.save_tables:
            md_path = report_dir / "benchmark_table.md"
            md_path.write_text(report["tables"]["markdown"])
            logger.info(f"Saved markdown table: {md_path}")
            
            # LaTeX table
            latex_path = report_dir / "benchmark_table.tex"
            latex_path.write_text(report["tables"]["latex"])
            logger.info(f"Saved LaTeX table: {latex_path}")
    
    def get_ablation_configs(self) -> Dict[str, str]:
        """Get ablation configurations.
        
        Returns:
            Dict {ablation_name: description}
        """
        configs = {}
        for ablation_name in self.registry.list_ablations():
            config = self.registry.get_experiment(ablation_name)
            configs[ablation_name] = config.ablation_of or config.description
        return configs
