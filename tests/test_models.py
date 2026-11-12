"""
Fraud Detection System (FDS) - Machine Learning Models & Evaluation Tests
Author: Alpha
"""

import os
import numpy as np
import pytest

from ml.evaluation.cost_optimizer import CostBasedThresholdOptimizer
from ml.evaluation.metrics import compute_fraud_metrics
from ml.explainers.shap_explainer import FraudSHAPExplainer
from ml.models.baselines import LOFFraudModel, OneClassSVMFraudModel
from ml.models.ensemble import HybridAnomalyEnsemble
from ml.models.isolation_forest import IsolationForestFraudModel


@pytest.fixture
def synthetic_features():
    """Generates synthetic normal feature vectors with injected anomalous outliers."""
    np.random.seed(42)
    # 400 normal samples (clustered around 0)
    normal = np.random.normal(loc=0.0, scale=1.0, size=(400, 8))
    # 20 anomalous samples (extreme values)
    outliers = np.random.uniform(low=4.0, high=9.0, size=(20, 8))

    X = np.vstack([normal, outliers])
    y = np.array([0] * 400 + [1] * 20)

    # Random permutation
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]


def test_isolation_forest_fit_predict_and_calibration(synthetic_features):
    """Verify Isolation Forest training, calibrated risk scores (0-100), and thresholding."""
    X, y = synthetic_features

    model = IsolationForestFraudModel(
        contamination=0.05,
        n_estimators=50,
        max_samples=128,
        random_state=42,
        decision_threshold=60.0,
    )
    model.fit(X)

    assert model.is_fitted_

    # Check risk score range
    risk_scores = model.predict_risk_scores(X)
    assert len(risk_scores) == len(X)
    assert np.all(risk_scores >= 0.0) and np.all(risk_scores <= 100.0)

    # Check predictions
    preds = model.predict(X)
    assert set(preds).issubset({0, 1})
    assert np.sum(preds == 1) > 0  # Should detect some anomalies

    # Verify score_single method
    single_vec = X[0]
    raw, risk, flagged = model.score_single(single_vec)
    assert isinstance(raw, float)
    assert 0.0 <= risk <= 100.0
    assert flagged == (risk >= 60.0)


def test_isolation_forest_save_and_load(tmp_path, synthetic_features):
    """Verify model serialization and deserialization preserves identical predictions."""
    X, _ = synthetic_features
    model = IsolationForestFraudModel(n_estimators=30, random_state=42)
    model.fit(X)

    scores_orig = model.predict_risk_scores(X)

    artifact_path = os.path.join(tmp_path, "test_if_model.joblib")
    model.save(artifact_path)
    assert os.path.exists(artifact_path)

    loaded_model = IsolationForestFraudModel.load(artifact_path)
    assert loaded_model.is_fitted_

    scores_loaded = loaded_model.predict_risk_scores(X)
    np.testing.assert_allclose(scores_orig, scores_loaded, rtol=1e-5)


def test_baseline_models_and_ensemble(synthetic_features):
    """Verify LOF, One-Class SVM baselines, and Hybrid Anomaly Ensemble."""
    X, y = synthetic_features

    # 1. LOF
    lof = LOFFraudModel(n_neighbors=15, contamination=0.05)
    lof.fit(X)
    lof_scores = lof.predict_risk_scores(X)
    assert np.all(lof_scores >= 0.0) and np.all(lof_scores <= 100.0)

    # 2. OC-SVM
    ocsvm = OneClassSVMFraudModel(nu=0.05)
    ocsvm.fit(X)
    svm_scores = ocsvm.predict_risk_scores(X)
    assert np.all(svm_scores >= 0.0) and np.all(svm_scores <= 100.0)

    # 3. Hybrid Ensemble
    ensemble = HybridAnomalyEnsemble(decision_threshold=60.0)
    ensemble.fit(X)
    ens_scores = ensemble.predict_risk_scores(X)
    assert len(ens_scores) == len(X)
    assert np.all(ens_scores >= 0.0) and np.all(ens_scores <= 100.0)


def test_cost_based_threshold_optimizer():
    """Verify threshold optimizer minimizes economic misclassification loss."""
    y_true = np.array([0] * 950 + [1] * 50)
    # Generate scores where fraud tends to have higher scores
    scores = np.concatenate([
        np.random.normal(loc=30.0, scale=10.0, size=950),
        np.random.normal(loc=75.0, scale=10.0, size=50),
    ])
    scores = np.clip(scores, 0.0, 100.0)

    optimizer = CostBasedThresholdOptimizer(cost_fn=500.0, cost_fp=15.0)
    res = optimizer.optimize(y_true, scores, search_range=(40.0, 80.0), step=1.0)

    assert "optimal_threshold" in res
    assert 40.0 <= res["optimal_threshold"] <= 80.0
    assert res["net_cost_savings"] > 0
    assert res["roi_percentage"] > 0


def test_metrics_computation():
    """Verify PR-AUC, ROC-AUC, and confusion matrix computations."""
    y_true = np.array([0, 0, 0, 1, 1, 0, 1, 0, 0, 1])
    scores = np.array([10.0, 20.0, 35.0, 85.0, 92.0, 15.0, 78.0, 30.0, 25.0, 88.0])

    metrics = compute_fraud_metrics(y_true, scores, threshold=70.0)

    assert metrics["pr_auc"] > 0.8
    assert metrics["roc_auc"] > 0.8
    assert metrics["true_positives"] == 4
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0


def test_shap_explainer_reason_codes(synthetic_features):
    """Verify TreeSHAP explanation and automated reason code generation."""
    X, _ = synthetic_features
    feature_names = [
        "amount",
        "log_amount",
        "hour",
        "day_of_week",
        "is_night",
        "is_new_device",
        "tx_velocity_1h",
        "distance_from_last_tx_km",
    ]

    model = IsolationForestFraudModel(n_estimators=30, random_state=42)
    model.fit(X)

    explainer = FraudSHAPExplainer(model, feature_names=feature_names)

    # An extreme anomalous transaction
    test_vec = np.array([8.5, 3.2, 3.0, 2.0, 1.0, 1.0, 5.0, 850.0])
    raw_feats = {
        "amount": 1250.0,
        "amount_zscore": 6.8,
        "user_rolling_mean_amount": 75.0,
        "is_night": 1,
        "hour": 3,
        "is_new_device": 1,
        "tx_velocity_1h": 5,
        "distance_from_last_tx_km": 850.0,
    }

    result = explainer.explain_transaction(test_vec, raw_feats, top_k=3)

    assert "feature_attributions" in result
    assert "reason_codes" in result
    assert len(result["reason_codes"]) >= 1

    # Check that reasons have code and description
    for r in result["reason_codes"]:
        assert "code" in r
        assert "description" in r
        assert "impact" in r
