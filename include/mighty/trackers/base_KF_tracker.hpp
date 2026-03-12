#pragma once

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>
#include <iostream>
#include "base_IMM_tracker.hpp"

// ==========================================
// 1. ABSTRACT BASE LINEAR KF MODEL
// ==========================================
class BaseKFModel : public IMMTrackerModel {
public:
    BaseKFModel(int state_dim = 9, int meas_dim = 3) 
        : IMMTrackerModel(state_dim, meas_dim) {}

    virtual ~BaseKFModel() = default;

    // Derived classes must implement prediction
    virtual void predict(double dt) = 0;

    // Centralized, Pure Linear Update (Identical for all models)
    void update(const Eigen::VectorXd& z) {
        // 1. Predict Measurement
        Eigen::VectorXd z_pred = H * x;
        Eigen::VectorXd y_res = z - z_pred;

        // 2. Kalman Gain
        Eigen::MatrixXd S = H * P * H.transpose() + R;
        Eigen::MatrixXd K = P * H.transpose() * S.inverse();

        // 3. Update State (No angle normalization or 180-flip hacks needed!)
        x = x + K * y_res;

        // 4. Update Covariance
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(state_dim_, state_dim_);
        P = (I - K * H) * P;

        // 5. Calculate Likelihood for IMM Mixing
        double det_S = S.determinant();
        if (det_S < 1e-9) det_S = 1e-9; // Prevent singularity

        double mahalanobis = (y_res.transpose() * S.inverse() * y_res).value();
        double norm_factor = 1.0 / std::sqrt(std::pow(2.0 * M_PI, meas_dim_) * det_S);
        
        double likelihood_val = norm_factor * std::exp(-0.5 * mahalanobis);
        this->likelihood = (likelihood_val < 1e-30) ? 1e-30 : likelihood_val; // Clamp
    }
    
    void setP(const Eigen::MatrixXd& P_in) { this->P = P_in; }
    void setR(const Eigen::MatrixXd& R_in) { this->R = R_in; }
};

// ==========================================
// 2. SPECIFIC MODELS
// ==========================================
namespace KFTrackers {
/*
 * Constant Velocity (CV)
 * Acceleration is 0. 
 */
class CVModel : public BaseKFModel {
private:
    double var_v_;
public:
    CVModel(int state_dim, int meas_dim, double sigma_v) 
    : BaseKFModel(state_dim, meas_dim), var_v_(sigma_v * sigma_v) {}

    void predict(double dt) override {
        double dt2 = dt * dt;
        double dt3 = dt2 * dt;

        // 1. Process Noise (Q)
        Q.setZero();
        for (int i = 0; i < 3; ++i) {
            Q(i, i)     = 0.5 * dt3 * var_v_; // Pos noise
            Q(i+3, i+3) = dt2 * var_v_;       // Vel noise
            Q(i+6, i+6) = 1e-6;               // Acc noise (minimal)
        }

        // 2. Transition Matrix (F)
        Eigen::MatrixXd F = Eigen::MatrixXd::Identity(state_dim_, state_dim_);
        for (int i = 0; i < 3; ++i) {
            F(i, i+3) = dt; // Pos = Pos + Vel * dt
        }

        // 3. Predict Kinematics linearly
        x = F * x;

        // Force acceleration strictly to 0
        x(6) = 0.0; x(7) = 0.0; x(8) = 0.0;

        // 4. Predict Covariance
        P = F * P * F.transpose() + Q;
    }
};

/*
 * Constant Acceleration (CA)
 * Smoothly accelerating or decelerating targets.
 */
class CAModel : public BaseKFModel {
private:
    double var_a_;
public:
    CAModel(int state_dim, int meas_dim, double sigma_a) 
    : BaseKFModel(state_dim, meas_dim), var_a_(sigma_a * sigma_a) {}

    void predict(double dt) override {
        double dt2 = dt * dt;
        double dt3 = dt2 * dt;
        double dt4 = dt3 * dt;

        // 1. Process Noise (Q)
        Q.setZero();
        for (int i = 0; i < 3; ++i) {
            Q(i, i)     = 0.25 * dt4 * var_a_;
            Q(i+3, i+3) = dt2 * var_a_;
            Q(i+6, i+6) = var_a_;
            // Correlations
            Q(i, i+3)   = Q(i+3, i) = 0.5 * dt3 * var_a_;
        }

        // 2. Transition Matrix (F)
        Eigen::MatrixXd F = Eigen::MatrixXd::Identity(state_dim_, state_dim_);
        for (int i = 0; i < 3; ++i) {
            F(i, i+3) = dt;               // Pos = Pos + Vel * dt
            F(i, i+6) = 0.5 * dt * dt;    // Pos = Pos + 0.5 * Acc * dt^2
            F(i+3, i+6) = dt;             // Vel = Vel + Acc * dt
        }

        // 3. Predict Kinematics linearly
        x = F * x;

        // 4. Predict Covariance
        P = F * P * F.transpose() + Q;
    }
};

/*
 * High Maneuver Model (Replaces CT)
 * Uses the CA kinematics but injects massive process noise to allow the 
 * filter to "trust the measurement" during sharp, unpredictable 180 turns.
 */
class ManeuverModel : public BaseKFModel {
private:
    double var_maneuver_;
public:
    ManeuverModel(int state_dim, int meas_dim, double sigma_maneuver) 
    : BaseKFModel(state_dim, meas_dim), var_maneuver_(sigma_maneuver * sigma_maneuver) {}

    void predict(double dt) override {
        double dt2 = dt * dt;
        double dt3 = dt2 * dt;
        double dt4 = dt3 * dt;

        Q.setZero();
        for (int i = 0; i < 3; ++i) {
            Q(i, i)     = 0.25 * dt4 * var_maneuver_;
            Q(i+3, i+3) = dt2 * var_maneuver_;
            Q(i+6, i+6) = var_maneuver_;
            Q(i, i+3)   = Q(i+3, i) = 0.5 * dt3 * var_maneuver_;
        }

        Eigen::MatrixXd F = Eigen::MatrixXd::Identity(state_dim_, state_dim_);
        for (int i = 0; i < 3; ++i) {
            F(i, i+3) = dt;
            F(i, i+6) = 0.5 * dt * dt;
            F(i+3, i+6) = dt;
        }

        x = F * x;
        P = F * P * F.transpose() + Q;
    }
};

/*
 * Stationary Model
 * Target is completely stopped. v = 0, a = 0.
 */
class StationaryModel : public BaseKFModel {
private:
    double var_pos_;
public:
    StationaryModel(int state_dim, int meas_dim, double sigma_pos) 
    : BaseKFModel(state_dim, meas_dim), var_pos_(sigma_pos * sigma_pos) {}

    void predict(double dt) override {
        // 1. Process Noise (Q)
        Q.setZero();
        for (int i = 0; i < 3; ++i) {
            Q(i, i) = var_pos_;  // Allow slight position sway
            Q(i+3, i+3) = 1e-6;  // Vel noise ~0
            Q(i+6, i+6) = 1e-6;  // Acc noise ~0
        }

        // 2. Transition Matrix (F)
        Eigen::MatrixXd F = Eigen::MatrixXd::Identity(state_dim_, state_dim_);
        // Cut off all connections from velocity and acceleration
        for (int i = 0; i < 3; ++i) {
            F(i, i+3) = 0.0;
            F(i, i+6) = 0.0;
            F(i+3, i+6) = 0.0;
        }

        // 3. Predict Kinematics (x, y, z remain unchanged)
        x = F * x;
        
        // Strictly force velocities and accelerations to zero
        for (int i = 3; i < 9; ++i) {
            x(i) = 0.0;
        }

        // 4. Predict Covariance
        P = F * P * F.transpose() + Q;
    }
};

} // namespace KFTrackers