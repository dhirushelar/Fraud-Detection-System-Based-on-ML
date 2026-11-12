-- =============================================================================
-- Fraud Detection System (FDS) - Core PostgreSQL Schema
-- Author: Alpha
-- Version: 1.0.0
-- Database: PostgreSQL 14+ compatible
-- Description: Production DDL schema with primary keys, foreign keys, 
--              check constraints, and composite indexes for low-latency queries.
-- =============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Clean drop for idempotent local migration runs (wrapped in dependency order)
DROP TABLE IF EXISTS drift_metrics CASCADE;
DROP TABLE IF EXISTS analyst_feedback CASCADE;
DROP TABLE IF EXISTS alerts CASCADE;
DROP TABLE IF EXISTS predictions CASCADE;
DROP TABLE IF EXISTS model_versions CASCADE;
DROP TABLE IF EXISTS transactions CASCADE;
DROP TABLE IF EXISTS merchants CASCADE;
DROP TABLE IF EXISTS users CASCADE;
DROP TABLE IF EXISTS auth_users CASCADE;

-- -----------------------------------------------------------------------------
-- 1. AUTH USERS (System RBAC: Analysts and Administrators)
-- -----------------------------------------------------------------------------
CREATE TABLE auth_users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) NOT NULL CHECK (role IN ('analyst', 'admin')),
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_auth_users_username ON auth_users(username);
CREATE INDEX idx_auth_users_role ON auth_users(role);

-- -----------------------------------------------------------------------------
-- 2. CUSTOMERS / USERS (Financial Account Holders)
-- -----------------------------------------------------------------------------
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    external_id VARCHAR(64) UNIQUE NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    email VARCHAR(255),
    account_created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    risk_status VARCHAR(30) NOT NULL DEFAULT 'NORMAL' 
        CHECK (risk_status IN ('NORMAL', 'WATCHLIST', 'SUSPENDED', 'BLOCKED')),
    home_lat NUMERIC(9, 6),
    home_lon NUMERIC(9, 6),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_users_external_id ON users(external_id);
CREATE INDEX idx_users_risk_status ON users(risk_status);

-- -----------------------------------------------------------------------------
-- 3. MERCHANTS (Receiving Parties / Payees)
-- -----------------------------------------------------------------------------
CREATE TABLE merchants (
    id SERIAL PRIMARY KEY,
    external_id VARCHAR(64) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    category VARCHAR(100) NOT NULL,
    risk_rating NUMERIC(3, 2) NOT NULL DEFAULT 1.00 
        CHECK (risk_rating >= 0.0 AND risk_rating <= 10.0),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_merchants_external_id ON merchants(external_id);
CREATE INDEX idx_merchants_category ON merchants(category);

-- -----------------------------------------------------------------------------
-- 4. TRANSACTIONS (Financial Event Ledger)
-- -----------------------------------------------------------------------------
CREATE TABLE transactions (
    id BIGSERIAL PRIMARY KEY,
    transaction_uid VARCHAR(64) UNIQUE NOT NULL,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    merchant_id INTEGER NOT NULL REFERENCES merchants(id) ON DELETE RESTRICT,
    amount NUMERIC(12, 2) NOT NULL CHECK (amount > 0),
    currency VARCHAR(3) NOT NULL DEFAULT 'USD',
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    location_lat NUMERIC(9, 6),
    location_lon NUMERIC(9, 6),
    device_id VARCHAR(128) NOT NULL,
    ip_address VARCHAR(45),
    channel VARCHAR(30) NOT NULL DEFAULT 'ONLINE' 
        CHECK (channel IN ('ONLINE', 'POS', 'ATM', 'MOBILE')),
    is_ground_truth_fraud BOOLEAN DEFAULT NULL,
    status VARCHAR(30) NOT NULL DEFAULT 'PROCESSED' 
        CHECK (status IN ('PENDING', 'PROCESSED', 'FLAGGED', 'BLOCKED', 'REVERSED')),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Compound index for fast temporal rolling lookups per account holder
CREATE INDEX idx_transactions_user_timestamp ON transactions(user_id, timestamp DESC);
CREATE INDEX idx_transactions_merchant_id ON transactions(merchant_id);
CREATE INDEX idx_transactions_timestamp ON transactions(timestamp DESC);
CREATE INDEX idx_transactions_device_id ON transactions(device_id);
CREATE INDEX idx_transactions_uid ON transactions(transaction_uid);

-- -----------------------------------------------------------------------------
-- 5. MODEL VERSIONS (MLflow / Artifact Register Metadata)
-- -----------------------------------------------------------------------------
CREATE TABLE model_versions (
    id SERIAL PRIMARY KEY,
    version_tag VARCHAR(50) UNIQUE NOT NULL,
    algorithm VARCHAR(50) NOT NULL,
    parameters JSONB NOT NULL DEFAULT '{}'::jsonb,
    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    decision_threshold NUMERIC(5, 4) NOT NULL DEFAULT 0.5000,
    artifact_path VARCHAR(255) NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT FALSE,
    trained_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_model_versions_active ON model_versions(is_active);

-- -----------------------------------------------------------------------------
-- 6. PREDICTIONS (Real-Time and Batch Anomaly Scoring Results)
-- -----------------------------------------------------------------------------
CREATE TABLE predictions (
    id BIGSERIAL PRIMARY KEY,
    transaction_id BIGINT NOT NULL REFERENCES transactions(id) ON DELETE CASCADE,
    model_version_id INTEGER NOT NULL REFERENCES model_versions(id) ON DELETE RESTRICT,
    raw_anomaly_score DOUBLE PRECISION NOT NULL,
    calibrated_risk_score NUMERIC(5, 2) NOT NULL 
        CHECK (calibrated_risk_score >= 0.0 AND calibrated_risk_score <= 100.0),
    is_flagged BOOLEAN NOT NULL DEFAULT FALSE,
    latency_ms NUMERIC(7, 2) NOT NULL,
    feature_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_predictions_transaction_id ON predictions(transaction_id);
CREATE INDEX idx_predictions_model_version ON predictions(model_version_id);
CREATE INDEX idx_predictions_risk_score ON predictions(calibrated_risk_score DESC);
CREATE INDEX idx_predictions_is_flagged ON predictions(is_flagged);

-- -----------------------------------------------------------------------------
-- 7. ALERTS (Analyst Queue for High-Risk Transactions)
-- -----------------------------------------------------------------------------
CREATE TABLE alerts (
    id BIGSERIAL PRIMARY KEY,
    alert_uid VARCHAR(64) UNIQUE NOT NULL,
    transaction_id BIGINT NOT NULL REFERENCES transactions(id) ON DELETE CASCADE,
    prediction_id BIGINT NOT NULL REFERENCES predictions(id) ON DELETE CASCADE,
    severity VARCHAR(20) NOT NULL 
        CHECK (severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
    status VARCHAR(30) NOT NULL DEFAULT 'OPEN' 
        CHECK (status IN ('OPEN', 'INVESTIGATING', 'RESOLVED_FRAUD', 'RESOLVED_FALSE_POSITIVE', 'DISMISSED')),
    trigger_reasons JSONB NOT NULL DEFAULT '[]'::jsonb,
    assigned_to_user_id INTEGER REFERENCES auth_users(id) ON DELETE SET NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_alerts_status_severity ON alerts(status, severity);
CREATE INDEX idx_alerts_created_at ON alerts(created_at DESC);
CREATE INDEX idx_alerts_transaction_id ON alerts(transaction_id);
CREATE INDEX idx_alerts_assigned_user ON alerts(assigned_to_user_id);

-- -----------------------------------------------------------------------------
-- 8. ANALYST FEEDBACK (Closed-Loop Ground Truth for Retraining)
-- -----------------------------------------------------------------------------
CREATE TABLE analyst_feedback (
    id BIGSERIAL PRIMARY KEY,
    alert_id BIGINT NOT NULL REFERENCES alerts(id) ON DELETE CASCADE,
    transaction_id BIGINT NOT NULL REFERENCES transactions(id) ON DELETE CASCADE,
    analyst_id INTEGER NOT NULL REFERENCES auth_users(id) ON DELETE RESTRICT,
    feedback_verdict VARCHAR(30) NOT NULL 
        CHECK (feedback_verdict IN ('TRUE_FRAUD', 'FALSE_POSITIVE', 'SUSPICIOUS_UNCONFIRMED')),
    reason_tags JSONB NOT NULL DEFAULT '[]'::jsonb,
    notes TEXT,
    reviewed_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_analyst_feedback_alert ON analyst_feedback(alert_id);
CREATE INDEX idx_analyst_feedback_verdict ON analyst_feedback(feedback_verdict);
CREATE INDEX idx_analyst_feedback_reviewed ON analyst_feedback(reviewed_at DESC);

-- -----------------------------------------------------------------------------
-- 9. DRIFT METRICS (Data & Concept Drift Telemetry: PSI & KS-Tests)
-- -----------------------------------------------------------------------------
CREATE TABLE drift_metrics (
    id SERIAL PRIMARY KEY,
    model_version_id INTEGER NOT NULL REFERENCES model_versions(id) ON DELETE CASCADE,
    feature_name VARCHAR(100) NOT NULL,
    metric_type VARCHAR(50) NOT NULL 
        CHECK (metric_type IN ('PSI', 'KS_STATISTIC', 'WASSERSTEIN')),
    metric_value DOUBLE PRECISION NOT NULL,
    p_value DOUBLE PRECISION,
    threshold DOUBLE PRECISION NOT NULL,
    is_drift_detected BOOLEAN NOT NULL DEFAULT FALSE,
    sample_size INTEGER NOT NULL,
    evaluated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_drift_metrics_version_time ON drift_metrics(model_version_id, evaluated_at DESC);
CREATE INDEX idx_drift_metrics_detected ON drift_metrics(is_drift_detected);
