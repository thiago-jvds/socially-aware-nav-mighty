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
class CVModel : public BaseKFModel
{
private:
  double process_noise_;
public:
  CVModel(int state_dim, int meas_dim, double process_noise)
      : BaseKFModel(state_dim, meas_dim), process_noise_(process_noise)
  {}

  void predict(double dt) override
  {
    double dt2 = dt * dt;
    double dt3 = dt2 * dt;

    // 1. Process Noise (Q): continuous white-acceleration model for [p, v].
    // For each axis, Q_axis = qa *
    // [ dt^3/3  dt^2/2
    //   dt^2/2  dt ]
    // Keep a tiny diffusion on the acceleration states since CV clamps them to 0.
    Q.setZero();
    for (int i = 0; i < 3; ++i)
    {
      const int p = i;
      const int v = i + 3;
      const int a = i + 6;
      const double qa = process_noise_;

      Q(p, p) = (dt3 / 3.0) * qa;
      Q(p, v) = (dt2 / 2.0) * qa;
      Q(v, p) = Q(p, v);
      Q(v, v) = dt * qa;
      Q(a, a) = 1e-6;
    }

    // 2. Transition Matrix (F)
    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(state_dim_, state_dim_);
    for (int i = 0; i < 3; ++i) {
        F(i, i+3) = dt;     // Pos = Pos + Vel * dt
    }

    // 3. Predict Kinematics linearly
    x = F * x;
    
    // Force acceleration strictly to 0
    x(6) = 0.0; x(7) = 0.0; x(8) = 0.0;

    // 4. Update Covariance
    P = F * P * F.transpose() + Q;
  }
};

class CAModel : public BaseKFModel
{
private:
  double process_noise_;
public:
  CAModel(int state_dim, int meas_dim, double process_noise)
      : BaseKFModel(state_dim, meas_dim), process_noise_(process_noise)
  {}

  void predict(double dt) override
  {
    double dt2 = dt * dt;
    double dt3 = dt2 * dt;
    double dt4 = dt3 * dt;
    double dt5 = dt4 * dt;

    // 1. Process Noise (Q): continuous white-jerk model for [p, v, a].
    // For each axis, Q_axis = qj *
    // [ dt^5/20  dt^4/8  dt^3/6
    //   dt^4/8   dt^3/3  dt^2/2
    //   dt^3/6   dt^2/2  dt ]
    Q.setZero();
    for (int i = 0; i < 3; ++i)
    {
      const int p = i;
      const int v = i + 3;
      const int a = i + 6;
      const double qj = process_noise_;

      Q(p, p) = (dt5 / 20.0) * qj;
      Q(p, v) = (dt4 / 8.0) * qj;
      Q(v, p) = Q(p, v);
      Q(p, a) = (dt3 / 6.0) * qj;
      Q(a, p) = Q(p, a);
      Q(v, v) = (dt3 / 3.0) * qj;
      Q(v, a) = (dt2 / 2.0) * qj;
      Q(a, v) = Q(v, a);
      Q(a, a) = dt * qj;
    }

    // 2. Transition Matrix (F)
    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(state_dim_, state_dim_);
    for (int i = 0; i < 3; ++i) {
        F(i, i+3) = dt;               // Pos = Pos + Vel * dt
        F(i, i+6) = 0.5 * dt2;        // Pos = Pos + 0.5 * Acc * dt^2
        F(i+3, i+6) = dt;             // Vel = Vel + Acc * dt
    }

    // 3. Predict Kinematics linearly
    x = F * x;

    // 4. Predict Covariance
    P = F * P * F.transpose() + Q;
  }
};

class StationaryModel : public BaseKFModel
{
private:
  double process_noise_;
public:
  StationaryModel(int state_dim, int meas_dim, double process_noise)
      : BaseKFModel(state_dim, meas_dim), process_noise_(process_noise)
  {
    Q.setZero();
    const double q_pos = process_noise;
    Q(0, 0) = q_pos;
    Q(1, 1) = q_pos;
    Q(2, 2) = q_pos;
  }

  void predict(double dt) override
  {
    (void)dt;
    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(state_dim_, state_dim_);
    F(3, 3) = 0.0;
    F(4, 4) = 0.0;
    F(5, 5) = 0.0;
    F(6, 6) = 0.0;
    F(7, 7) = 0.0;
    F(8, 8) = 0.0;

    x = F * x;
    P = F * P * F.transpose() + Q;
  }
};

} // namespace KFTrackers