"""
Automated interpretability validation and testing framework.

This module provides comprehensive validation for SHAP explanations and model interpretability:
- SHAP consistency tests across different background datasets
- Feature importance stability checks with bootstrap confidence intervals
- Automated explanation quality metrics (accuracy, robustness)
- Counterfactual explanation generation for individual predictions
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import shap
from scipy import stats

from .data import PROCESSED_DIR
from .model import MODELS_DIR
from .utils import ensure_dir, load_json, save_json, set_seeds


@dataclass
class ExplanationQualityMetrics:
    """Quality metrics for SHAP explanations."""
    # Consistency metrics
    background_consistency: float  # Consistency across different background datasets
    stability_score: float         # Stability of explanations across bootstrap samples

    # Accuracy metrics
    feature_importance_rank_stability: float  # Stability of feature importance rankings
    explanation_sparsity: float     # Fraction of features with non-zero importance

    # Robustness metrics
    perturbation_robustness: float  # Robustness to small input perturbations
    cross_validation_consistency: float  # Consistency across CV folds

    # Summary metrics
    overall_quality_score: float    # Composite quality score

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class CounterfactualExplanation:
    """Counterfactual explanation for a single prediction."""
    original_prediction: float
    counterfactual_prediction: float
    feature_changes: Dict[str, Tuple[float, float]]  # (original_value, counterfactual_value)
    feature_importance: Dict[str, float]
    validity_score: float  # How well the counterfactual achieves the desired outcome
    proximity_score: float  # How close the counterfactual is to the original


@dataclass
class InterpretabilityValidationReport:
    """Comprehensive interpretability validation report."""
    model_version: str
    dataset_hash: str
    validation_timestamp: str

    # Quality metrics
    explanation_quality: ExplanationQualityMetrics

    # Stability analysis
    feature_importance_ci: Dict[str, Dict[str, float]]  # feature -> {lower, upper, mean}

    # Consistency analysis
    background_consistency_scores: Dict[str, float]  # background_size -> consistency_score

    # Robustness analysis
    perturbation_analysis: Dict[str, float]

    # Counterfactual examples
    counterfactual_examples: List[CounterfactualExplanation]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_version": self.model_version,
            "dataset_hash": self.dataset_hash,
            "validation_timestamp": self.validation_timestamp,
            "explanation_quality": self.explanation_quality.to_dict(),
            "feature_importance_ci": self.feature_importance_ci,
            "background_consistency_scores": self.background_consistency_scores,
            "perturbation_analysis": self.perturbation_analysis,
            "counterfactual_examples": [
                {
                    "original_prediction": ex.original_prediction,
                    "counterfactual_prediction": ex.counterfactual_prediction,
                    "feature_changes": ex.feature_changes,
                    "feature_importance": ex.feature_importance,
                    "validity_score": ex.validity_score,
                    "proximity_score": ex.proximity_score,
                }
                for ex in self.counterfactual_examples
            ],
        }


class InterpretabilityValidator:
    """Main class for automated interpretability validation."""

    def __init__(
        self,
        background_sizes: List[int] = [50, 100, 200],
        n_bootstrap: int = 100,
        perturbation_magnitude: float = 0.1,
        random_seed: int = 42,
    ):
        """
        Initialize the interpretability validator.

        Args:
            background_sizes: Different background dataset sizes to test
            n_bootstrap: Number of bootstrap samples for stability testing
            perturbation_magnitude: Magnitude of perturbations for robustness testing
            random_seed: Random seed for reproducibility
        """
        self.background_sizes = background_sizes
        self.n_bootstrap = n_bootstrap
        self.perturbation_magnitude = perturbation_magnitude
        self.random_seed = random_seed

        set_seeds(random_seed)

    def validate_explanations(
        self,
        processed_dir: Path = PROCESSED_DIR,
        models_dir: Path = MODELS_DIR,
        output_dir: Path = Path("reports/interpretability_validation"),
    ) -> InterpretabilityValidationReport:
        """
        Run comprehensive interpretability validation.

        Args:
            processed_dir: Directory containing processed data
            models_dir: Directory containing trained models
            output_dir: Directory to save validation results

        Returns:
            Comprehensive validation report
        """
        ensure_dir(output_dir)

        # Load training state
        features, splits, pipeline = self._load_training_state(processed_dir, models_dir)
        train_genes = splits["train_genes"]
        test_genes = splits["test_genes"]

        background = features.loc[train_genes]
        target = features.loc[test_genes]

        # Run validation tests
        print("🔍 Running interpretability validation...")

        # 1. Test background consistency
        print("  📊 Testing background consistency...")
        background_consistency = self._test_background_consistency(pipeline, background, target)

        # 2. Test explanation stability
        print("  🔄 Testing explanation stability...")
        stability_metrics = self._test_explanation_stability(pipeline, background, target)

        # 3. Test feature importance stability
        print("  📈 Testing feature importance stability...")
        feature_stability = self._test_feature_importance_stability(pipeline, background, target)

        # 4. Test perturbation robustness
        print("  🛡️ Testing perturbation robustness...")
        perturbation_robustness = self._test_perturbation_robustness(pipeline, background, target)

        # 5. Generate counterfactual examples
        print("  🔄 Generating counterfactual examples...")
        counterfactuals = self._generate_counterfactual_examples(pipeline, background, target)

        # 6. Compute cross-validation consistency
        print("  ✅ Computing cross-validation consistency...")
        cv_consistency = self._test_cross_validation_consistency(pipeline, features, splits)

        # Compile quality metrics
        explanation_quality = ExplanationQualityMetrics(
            background_consistency=background_consistency["overall_score"],
            stability_score=stability_metrics["stability_score"],
            feature_importance_rank_stability=feature_stability["rank_stability"],
            explanation_sparsity=stability_metrics["sparsity"],
            perturbation_robustness=perturbation_robustness["robustness_score"],
            cross_validation_consistency=cv_consistency,
            overall_quality_score=self._compute_overall_quality_score(
                background_consistency["overall_score"],
                stability_metrics["stability_score"],
                feature_stability["rank_stability"],
                perturbation_robustness["robustness_score"],
                cv_consistency,
            ),
        )

        # Create validation report
        report = InterpretabilityValidationReport(
            model_version="latest",  # Would extract from model metadata
            dataset_hash=splits.get("dataset_hash", "unknown"),
            validation_timestamp=pd.Timestamp.now().isoformat(),
            explanation_quality=explanation_quality,
            feature_importance_ci=feature_stability["confidence_intervals"],
            background_consistency_scores=background_consistency["scores_by_size"],
            perturbation_analysis=perturbation_robustness["detailed_scores"],
            counterfactual_examples=counterfactuals,
        )

        # Save report
        report_path = output_dir / "validation_report.json"
        save_json(report_path, report.to_dict())

        print(f"✅ Interpretability validation completed. Report saved to {report_path}")
        return report

    def _load_training_state(self, processed_dir: Path, models_dir: Path):
        """Load training data and model."""
        from joblib import load

        features = pd.read_parquet(processed_dir / "features.parquet")
        splits = load_json(processed_dir / "splits.json")
        model_path = models_dir / "logreg_pipeline.pkl"

        if not model_path.exists():
            raise FileNotFoundError("Trained model not found; run train before validation")

        pipeline = load(model_path)
        return features, splits, pipeline

    def _test_background_consistency(
        self, pipeline, background: pd.DataFrame, target: pd.DataFrame
    ) -> Dict:
        """Test SHAP explanation consistency across different background sizes."""

        def predict_fn(data: pd.DataFrame) -> np.ndarray:
            return pipeline.predict_proba(data)[:, 1]

        consistency_scores = {}
        explanations = {}

        for bg_size in self.background_sizes:
            # Sample background data
            bg_sample = background.sample(min(bg_size, len(background)), random_state=self.random_seed)

            # Create explainer and compute SHAP values
            explainer = shap.Explainer(predict_fn, bg_sample, seed=self.random_seed)
            shap_values = explainer(target.iloc[:10])  # Use subset for speed

            explanations[bg_size] = np.array(shap_values.values)
            consistency_scores[bg_size] = self._compute_consistency_score(explanations[bg_size])

        # Compute overall consistency across background sizes
        all_explanations = list(explanations.values())
        overall_score = np.mean([
            self._compute_pairwise_consistency(exp1, exp2)
            for i, exp1 in enumerate(all_explanations)
            for exp2 in all_explanations[i+1:]
        ])

        return {
            "scores_by_size": consistency_scores,
            "overall_score": float(overall_score),
        }

    def _compute_consistency_score(self, shap_values: np.ndarray) -> float:
        """Compute consistency score for a set of SHAP values."""
        if shap_values.shape[0] <= 1:
            return 1.0

        # Compute correlation between explanations for different samples
        correlations = []
        for i in range(shap_values.shape[0]):
            for j in range(i + 1, shap_values.shape[0]):
                corr = np.corrcoef(shap_values[i], shap_values[j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))

        return float(np.mean(correlations)) if correlations else 0.0

    def _compute_pairwise_consistency(self, exp1: np.ndarray, exp2: np.ndarray) -> float:
        """Compute consistency between two explanation sets."""
        if exp1.shape != exp2.shape:
            return 0.0

        # Average correlation across all sample pairs
        correlations = []
        for i in range(min(exp1.shape[0], exp2.shape[0])):
            corr = np.corrcoef(exp1[i], exp2[i])[0, 1]
            if not np.isnan(corr):
                correlations.append(abs(corr))

        return float(np.mean(correlations)) if correlations else 0.0

    def _test_explanation_stability(
        self, pipeline, background: pd.DataFrame, target: pd.DataFrame
    ) -> Dict:
        """Test stability of explanations across bootstrap samples."""

        def predict_fn(data: pd.DataFrame) -> np.ndarray:
            return pipeline.predict_proba(data)[:, 1]

        # Use a fixed background for stability testing
        bg_sample = background.sample(100, random_state=self.random_seed)
        explainer = shap.Explainer(predict_fn, bg_sample, seed=self.random_seed)

        # Compute SHAP values multiple times with bootstrap resampling
        bootstrap_explanations = []
        n_samples = min(20, len(target))  # Limit for computational efficiency

        for _ in range(self.n_bootstrap):
            # Bootstrap sample of target data
            bootstrap_indices = np.random.choice(len(target), size=n_samples, replace=True)
            bootstrap_target = target.iloc[bootstrap_indices]

            try:
                shap_values = explainer(bootstrap_target)
                bootstrap_explanations.append(np.array(shap_values.values))
            except Exception as e:
                print(f"Warning: Bootstrap explanation failed: {e}")
                continue

        if not bootstrap_explanations:
            return {"stability_score": 0.0, "sparsity": 0.0}

        # Compute stability as average pairwise consistency
        stability_scores = []
        for i, exp1 in enumerate(bootstrap_explanations):
            for j, exp2 in enumerate(bootstrap_explanations[i+1:]):
                consistency = self._compute_pairwise_consistency(exp1, exp2)
                stability_scores.append(consistency)

        stability_score = float(np.mean(stability_scores)) if stability_scores else 0.0

        # Compute sparsity (fraction of features with non-zero importance)
        mean_abs_shap = np.mean(np.abs(np.concatenate(bootstrap_explanations, axis=0)), axis=0)
        sparsity = float(np.mean(mean_abs_shap > 0.01))  # Threshold for "meaningful" importance

        return {
            "stability_score": stability_score,
            "sparsity": sparsity,
        }

    def _test_feature_importance_stability(
        self, pipeline, background: pd.DataFrame, target: pd.DataFrame
    ) -> Dict:
        """Test stability of feature importance rankings."""

        def predict_fn(data: pd.DataFrame) -> np.ndarray:
            return pipeline.predict_proba(data)[:, 1]

        bg_sample = background.sample(100, random_state=self.random_seed)
        explainer = shap.Explainer(predict_fn, bg_sample, seed=self.random_seed)

        # Compute multiple explanations
        n_bootstrap = min(50, self.n_bootstrap)  # Limit for computational efficiency
        importance_matrices = []

        for _ in range(n_bootstrap):
            # Bootstrap sample
            sample_size = min(50, len(target))
            sample_indices = np.random.choice(len(target), size=sample_size, replace=True)
            sample_target = target.iloc[sample_indices]

            try:
                shap_values = explainer(sample_target)
                mean_abs_importance = np.mean(np.abs(shap_values.values), axis=0)
                importance_matrices.append(mean_abs_importance)
            except Exception:
                continue

        if not importance_matrices:
            return {"rank_stability": 0.0, "confidence_intervals": {}}

        # Compute rank stability using Kendall tau correlation
        rank_stability_scores = []
        importance_matrix = np.array(importance_matrices)

        for i in range(importance_matrix.shape[0]):
            for j in range(i + 1, importance_matrix.shape[0]):
                # Convert to ranks
                rank_i = stats.rankdata(importance_matrix[i])
                rank_j = stats.rankdata(importance_matrix[j])

                # Kendall tau correlation
                tau, _ = stats.kendalltau(rank_i, rank_j)
                rank_stability_scores.append(abs(tau))

        rank_stability = float(np.mean(rank_stability_scores)) if rank_stability_scores else 0.0

        # Compute confidence intervals for feature importance
        feature_names = list(target.columns)
        confidence_intervals = {}

        for i, feature in enumerate(feature_names):
            importance_values = importance_matrix[:, i]
            ci_lower = float(np.percentile(importance_values, 2.5))
            ci_upper = float(np.percentile(importance_values, 97.5))
            mean_importance = float(np.mean(importance_values))

            confidence_intervals[feature] = {
                "mean": mean_importance,
                "lower": ci_lower,
                "upper": ci_upper,
            }

        return {
            "rank_stability": rank_stability,
            "confidence_intervals": confidence_intervals,
        }

    def _test_perturbation_robustness(
        self, pipeline, background: pd.DataFrame, target: pd.DataFrame
    ) -> Dict:
        """Test robustness of explanations to input perturbations."""

        def predict_fn(data: pd.DataFrame) -> np.ndarray:
            return pipeline.predict_proba(data)[:, 1]

        bg_sample = background.sample(100, random_state=self.random_seed)
        explainer = shap.Explainer(predict_fn, bg_sample, seed=self.random_seed)

        # Test on a subset for computational efficiency
        test_size = min(10, len(target))
        test_data = target.iloc[:test_size]

        original_explanations = []
        perturbed_explanations = []

        try:
            original_shap = explainer(test_data)
            original_explanations = np.array(original_shap.values)
        except Exception as e:
            print(f"Warning: Failed to compute original explanations: {e}")
            return {"robustness_score": 0.0, "detailed_scores": {}}

        # Apply perturbations and recompute explanations
        for _ in range(10):  # Number of perturbation trials
            # Add small random noise to features
            noise = np.random.normal(0, self.perturbation_magnitude, test_data.shape)
            perturbed_data = test_data + noise

            try:
                perturbed_shap = explainer(perturbed_data)
                perturbed_explanations.append(np.array(perturbed_shap.values))
            except Exception:
                continue

        if not perturbed_explanations:
            return {"robustness_score": 0.0, "detailed_scores": {}}

        perturbed_matrix = np.array(perturbed_explanations)

        # Compute robustness as consistency between original and perturbed explanations
        robustness_scores = []
        for i in range(perturbed_matrix.shape[0]):
            consistency = self._compute_pairwise_consistency(original_explanations, perturbed_matrix[i])
            robustness_scores.append(consistency)

        robustness_score = float(np.mean(robustness_scores)) if robustness_scores else 0.0

        return {
            "robustness_score": robustness_score,
            "detailed_scores": {
                "mean_robustness": robustness_score,
                "robustness_std": float(np.std(robustness_scores)) if robustness_scores else 0.0,
                "n_perturbations": len(perturbed_explanations),
            },
        }

    def _test_cross_validation_consistency(
        self, pipeline, features: pd.DataFrame, splits: Dict
    ) -> float:
        """Test consistency of explanations across cross-validation folds."""
        # Simple implementation: split data and compare explanations
        from sklearn.model_selection import KFold

        train_genes = splits["train_genes"]
        train_features = features.loc[train_genes]

        # Create 3-fold CV
        kf = KFold(n_splits=3, shuffle=True, random_state=self.random_seed)

        fold_explanations = []

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_features)):
            if fold_idx >= 2:  # Limit to 2 folds for speed
                break

            fold_train = train_features.iloc[train_idx]
            fold_val = train_features.iloc[val_idx]

            # Sample background for this fold
            bg_size = min(50, len(fold_train))
            bg_sample = fold_train.sample(bg_size, random_state=self.random_seed)

            def predict_fn(data: pd.DataFrame) -> np.ndarray:
                return pipeline.predict_proba(data)[:, 1]

            try:
                explainer = shap.Explainer(predict_fn, bg_sample, seed=self.random_seed)
                shap_values = explainer(fold_val.iloc[:5])  # Small sample for speed
                fold_explanations.append(np.mean(np.abs(shap_values.values), axis=0))
            except Exception:
                continue

        if len(fold_explanations) < 2:
            return 0.0

        # Compute consistency across folds
        consistency_scores = []
        for i, exp1 in enumerate(fold_explanations):
            for j, exp2 in enumerate(fold_explanations[i+1:]):
                corr = np.corrcoef(exp1, exp2)[0, 1]
                if not np.isnan(corr):
                    consistency_scores.append(abs(corr))

        return float(np.mean(consistency_scores)) if consistency_scores else 0.0

    def _generate_counterfactual_examples(
        self, pipeline, background: pd.DataFrame, target: pd.DataFrame, n_examples: int = 3
    ) -> List[CounterfactualExplanation]:
        """Generate counterfactual explanations for sample predictions."""

        counterfactuals = []
        sample_indices = np.random.choice(len(target), size=min(n_examples, len(target)), replace=False)

        for idx in sample_indices:
            sample = target.iloc[[idx]]
            original_pred = pipeline.predict_proba(sample)[0, 1]

            # Generate counterfactual by perturbing features
            # This is a simplified counterfactual generation
            perturbed_sample = sample.copy()

            # Find most important features (simplified)
            feature_names = list(target.columns)
            n_features_to_change = min(3, len(feature_names))

            # Randomly select features to change
            features_to_change = np.random.choice(feature_names, size=n_features_to_change, replace=False)

            for feature in features_to_change:
                # Apply perturbation in direction that would change prediction
                original_value = sample[feature].iloc[0]

                # Simple perturbation strategy
                if original_pred > 0.5:
                    # Try to decrease prediction (make less likely to be positive)
                    perturbation = -abs(np.random.normal(0, 0.5))
                else:
                    # Try to increase prediction (make more likely to be positive)
                    perturbation = abs(np.random.normal(0, 0.5))

                perturbed_sample[feature] = original_value + perturbation

            counterfactual_pred = pipeline.predict_proba(perturbed_sample)[0, 1]

            # Compute feature importance (simplified)
            feature_importance = {}
            for feature in features_to_change:
                change = perturbed_sample[feature].iloc[0] - sample[feature].iloc[0]
                # Simple importance based on magnitude of change and prediction change
                importance = abs(change) * abs(counterfactual_pred - original_pred)
                feature_importance[feature] = float(importance)

            # Compute validity (did we achieve desired outcome?)
            if original_pred > 0.5:
                desired_outcome = counterfactual_pred < original_pred
            else:
                desired_outcome = counterfactual_pred > original_pred

            validity_score = 1.0 if desired_outcome else 0.0

            # Compute proximity (how close is counterfactual to original?)
            feature_changes = {}
            for feature in features_to_change:
                original_val = sample[feature].iloc[0]
                counterfactual_val = perturbed_sample[feature].iloc[0]
                feature_changes[feature] = (float(original_val), float(counterfactual_val))

            # Simple proximity based on normalized feature changes
            total_change = sum(abs(orig - cf) for orig, cf in feature_changes.values())
            proximity_score = max(0.0, 1.0 - total_change / len(features_to_change))

            counterfactual = CounterfactualExplanation(
                original_prediction=float(original_pred),
                counterfactual_prediction=float(counterfactual_pred),
                feature_changes=feature_changes,
                feature_importance=feature_importance,
                validity_score=validity_score,
                proximity_score=proximity_score,
            )

            counterfactuals.append(counterfactual)

        return counterfactuals

    def _compute_overall_quality_score(self, *scores: float) -> float:
        """Compute overall interpretability quality score."""
        valid_scores = [s for s in scores if not np.isnan(s) and not np.isinf(s)]
        if not valid_scores:
            return 0.0

        # Weighted average of different quality metrics
        weights = [0.25, 0.20, 0.20, 0.15, 0.20]  # Weights for different metrics
        return float(np.average(valid_scores, weights=weights[:len(valid_scores)]))

    def generate_validation_summary(
        self, report: InterpretabilityValidationReport, output_path: Optional[Path] = None
    ) -> str:
        """Generate a human-readable summary of validation results."""

        summary = f"""
# Interpretability Validation Summary

**Model Version:** {report.model_version}
**Validation Date:** {report.validation_timestamp}
**Dataset Hash:** {report.dataset_hash}

## Explanation Quality Metrics

- **Overall Quality Score:** {report.explanation_quality.overall_quality_score:.3f}
- **Background Consistency:** {report.explanation_quality.background_consistency:.3f}
- **Stability Score:** {report.explanation_quality.stability_score:.3f}
- **Feature Importance Rank Stability:** {report.explanation_quality.feature_importance_rank_stability:.3f}
- **Explanation Sparsity:** {report.explanation_quality.explanation_sparsity:.3f}
- **Perturbation Robustness:** {report.explanation_quality.perturbation_robustness:.3f}
- **Cross-Validation Consistency:** {report.explanation_quality.cross_validation_consistency:.3f}

## Feature Importance Confidence Intervals

Top 5 features with 95% confidence intervals:
"""

        # Add top features with confidence intervals
        sorted_features = sorted(
            report.feature_importance_ci.items(),
            key=lambda x: x[1]["mean"],
            reverse=True
        )[:5]

        for feature, ci in sorted_features:
            mean_val = ci["mean"]
            lower_val = ci["lower"]
            upper_val = ci["upper"]
            summary += f"- **{feature}:** {mean_val:.4f} (95% CI: [{lower_val:.4f}, {upper_val:.4f}])\n"

        # Add background consistency results
        summary += "\n## Background Consistency Analysis\n\n"
        for bg_size, score in report.background_consistency_scores.items():
            summary += f"- Background size {bg_size}: {score:.3f}\n"

        # Add perturbation analysis
        summary += "\n## Perturbation Robustness\n\n"
        for metric, value in report.perturbation_analysis.items():
            summary += f"- {metric.replace('_', ' ').title()}: {value:.3f}\n"

        # Add counterfactual examples
        if report.counterfactual_examples:
            summary += "\n## Counterfactual Examples\n\n"
            for i, cf in enumerate(report.counterfactual_examples):
                summary += f"\n### Example {i+1}\n"
                summary += f"- **Original Prediction:** {cf.original_prediction:.3f}\n"
                summary += f"- **Counterfactual Prediction:** {cf.counterfactual_prediction:.3f}\n"
                summary += f"- **Validity Score:** {cf.validity_score:.3f}\n"
                summary += f"- **Proximity Score:** {cf.proximity_score:.3f}\n\n"
                summary += "**Feature Changes:**\n"
                for feature, (orig, cf_val) in cf.feature_changes.items():
                    summary += f"- {feature}: {orig:.3f} → {cf_val:.3f}\n"

        # Add recommendations
        summary += "\n## Recommendations\n\n"
        quality = report.explanation_quality

        if quality.overall_quality_score < 0.7:
            summary += "- ⚠️ Overall explanation quality is below acceptable threshold\n"
        if quality.background_consistency < 0.8:
            summary += "- 🔄 Consider using larger or more diverse background datasets\n"
        if quality.stability_score < 0.7:
            summary += "- 📊 Explanations may be unstable; consider increasing background size\n"
        if quality.perturbation_robustness < 0.8:
            summary += "- 🛡️ Explanations may be sensitive to input perturbations\n"
        if quality.feature_importance_rank_stability < 0.7:
            summary += "- 📈 Feature importance rankings may be unstable\n"

        if quality.overall_quality_score >= 0.8:
            summary += "- ✅ Explanations appear reliable and consistent\n"

        if output_path:
            output_path.write_text(summary)
            summary += f"\n\nFull report saved to: {output_path}"

        return summary
