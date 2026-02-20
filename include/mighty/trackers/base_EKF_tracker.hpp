#pragma once

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <cmath>
#include <std_msgs/msg/color_rgba.hpp>

// Helpers
// Source: https://docs.ros.org/en/jazzy/p/angles/generated/program_listing_file_include_angles_angles.h.html
static inline double normalize_angle(double angle)
{
    const double result = fmod(angle + M_PI, 2.0*M_PI);
    if(result <= 0.0) return result + M_PI;
    return result - M_PI;
}

// ==========================================
// 1. ABSTRACT BASE MODEL
// ==========================================
class BaseEKFModel {
public:
    int state_dim_;
    int meas_dim_;

    Eigen::VectorXd x;
    Eigen::MatrixXd P;
    Eigen::MatrixXd Q;
    Eigen::MatrixXd R;
    Eigen::MatrixXd H;
    double likelihood;

    BaseEKFModel(int state_dim, int meas_dim) 
        : state_dim_(state_dim), meas_dim_(meas_dim) {
        x = Eigen::VectorXd::Zero(state_dim_);
        P = Eigen::MatrixXd::Identity(state_dim_, state_dim_);
        Q = Eigen::MatrixXd::Identity(state_dim_, state_dim_);
        R = Eigen::MatrixXd::Identity(meas_dim_, meas_dim_);
        H = Eigen::MatrixXd::Zero(meas_dim_, state_dim_);
        
        // Standard 3D Position Measurement Matrix
        H(0, 0) = 1.0; 
        H(1, 1) = 1.0; 
        H(2, 2) = 1.0;
    }

    virtual ~BaseEKFModel() = default;

    // The ONLY function that derived classes must implement!
    virtual void predict(double dt) = 0;

    // Centralized Update (Identical for all models)
    void update(const Eigen::VectorXd& z) {
        Eigen::VectorXd z_pred = H * x;
        Eigen::VectorXd y_res = z - z_pred;

        Eigen::MatrixXd S = H * P * H.transpose() + R;
        Eigen::MatrixXd K = P * H.transpose() * S.inverse();

        x = x + K * y_res;
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(state_dim_, state_dim_);
        P = (I - K * H) * P;

        double det_S = S.determinant();
        if (det_S < 1e-9) det_S = 1e-9; // Prevent singularity

        double mahalanobis = (y_res.transpose() * S.inverse() * y_res).value();
        double norm_factor = 1.0 / std::sqrt(std::pow(2.0 * M_PI, meas_dim_) * det_S);
        
        double likelihood = norm_factor * std::exp(-0.5 * mahalanobis);
        this->likelihood =  (likelihood < 1e-30) ? 1e-30 : likelihood; // Clamp
    }
    
    void setP(const Eigen::MatrixXd& P) {
        this->P = P;
    }

    void setR(const Eigen::MatrixXd& R) {
        this->R = R;
    }

};

class CAModel : public BaseEKFModel {
private:
    double var_a_;
    double var_yaw_;
public:
    CAModel(int state_dim, int meas_dim, double sigma_a, double sigma_yaw_rate) 
    : BaseEKFModel(state_dim, meas_dim),
      var_a_ (sigma_a * sigma_a),
      var_yaw_ (sigma_yaw_rate * sigma_yaw_rate) {}

    void setQ(double dt) {
        Q.setZero();
        double dt2 = dt * dt;
        double dt3 = dt2 * dt;
        double dt4 = dt3 * dt;

        Q(0,0) = 0.25 * dt4 * var_a_;    // x
        Q(0,3) = 0.5 * dt3 * var_a_;     // x-v correlation
        Q(3,0) = 0.5 * dt3 * var_a_;
        Q(3,3) = dt2 * var_a_;           // v
        
        Q(1,1) = 0.25 * dt4 * var_a_;    // y
        Q(1,3) = 0.5 * dt3 * var_a_;     // y-v correlation
        Q(3,1) = 0.5 * dt3 * var_a_;

        Q(2,2) = 0.25 * dt4 * var_a_;    // z
        Q(2,3) = 0.5 * dt3 * var_a_;     // z-v correlation
        Q(3,2) = 0.5 * dt3 * var_a_;
        
        // Angular Rate Coupling
        Q(4,4) = dt2 * var_yaw_;     // theta
        Q(7,7) = var_yaw_;           // theta_dot
    };

    void predict(double dt) override {
        double px = x(0), py = x(1), pz = x(2);
        double v  = x(3), th = x(4), ph = x(5);
        double a  = x(6); // Allowing mixed acceleration

        setQ(dt);

        // 1. Predict Kinematics
        double dist_step = v * dt + 0.5 * a * dt * dt;
        
        x(0) = px + dist_step * cos(ph) * cos(th);
        x(1) = py + dist_step * cos(ph) * sin(th);
        x(2) = pz + dist_step * sin(ph);
        x(3) = v + a * dt;
        x(7) = 0.0; // Force Turn Rate to 0
        x(8) = 0.0;

        // 2. Jacobian Matrix (F)
        Eigen::MatrixXd F = Eigen::MatrixXd::Identity(state_dim_, state_dim_);
        
        double s_th = sin(th), c_th = cos(th);
        double s_ph = sin(ph), c_ph = cos(ph);
        double d_dist_dv = dt;
        double d_dist_da = 0.5 * dt * dt;

        F(0, 3) = d_dist_dv * c_ph * c_th;      
        F(0, 6) = d_dist_da * c_ph * c_th;      
        F(0, 4) = -dist_step * c_ph * s_th;     
        
        F(1, 3) = d_dist_dv * c_ph * s_th;
        F(1, 6) = d_dist_da * c_ph * s_th;
        F(1, 4) = dist_step * c_ph * c_th;

        F(3, 6) = dt;

        // 3. Update Covariance
        P = F * P * F.transpose() + Q;
    }
};

class CTRVModel : public BaseEKFModel {
private:
    bool is_left_;
    double threshold_;
    double var_a_;
    double var_yaw_;

public:
    CTRVModel(int state_dim, int meas_dim, bool is_left, double threshold,
                double sigma_a, double sigma_yaw_accel) 
        : BaseEKFModel(state_dim, meas_dim),
        is_left_(is_left),
        threshold_(threshold),
        var_a_ (sigma_a * sigma_a),
        var_yaw_ (sigma_yaw_accel * sigma_yaw_accel) {}

    void predict(double dt) override {
        double px = x(0), py = x(1), pz = x(2);
        double v  = x(3), th = x(4), ph = x(5);
        double a  = x(6), th_d = x(7), ph_d = x(8);

        setQ(dt, th);

        // 2. Apply Turn Rate Constraints
        // If we expect to turn left, ensure yaw rate is sufficiently positive
        if (is_left_) {
            if (th_d < threshold_) th_d = threshold_;
        } 
        // If we expect to turn right, ensure yaw rate is sufficiently negative
        else {
            if (th_d > -threshold_) th_d = -threshold_;
        }

        // 3. Predict Kinematics
        double dist_step = v * dt + 0.5 * a * dt * dt;
        
        // Update Position
        x(0) = px + dist_step * cos(ph) * cos(th);
        x(1) = py + dist_step * cos(ph) * sin(th);
        x(2) = pz + dist_step * sin(ph);
        
        // Update Kinematics & Angles
        x(3) = v + a * dt;
        x(4) = normalize_angle(th + th_d * dt);
        x(5) = normalize_angle(ph + ph_d * dt);
        
        // Note: a, th_d, and ph_d remain constant 
        x(6) = a;
        x(7) = th_d;
        x(8) = ph_d;

        // 4. Compute Jacobian Matrix (F)
        Eigen::MatrixXd F = Eigen::MatrixXd::Identity(state_dim_, state_dim_);
        
        double s_th = sin(th), c_th = cos(th);
        double s_ph = sin(ph), c_ph = cos(ph);
        
        double d_dist_dv = dt;
        double d_dist_da = 0.5 * dt * dt;

        // Derivatives for Position X
        F(0, 3) = d_dist_dv * c_ph * c_th;      
        F(0, 6) = d_dist_da * c_ph * c_th;      
        F(0, 4) = -dist_step * c_ph * s_th;     
        F(0, 5) = -dist_step * s_ph * c_th;

        // Derivatives for Position Y
        F(1, 3) = d_dist_dv * c_ph * s_th;
        F(1, 6) = d_dist_da * c_ph * s_th;
        F(1, 4) = dist_step * c_ph * c_th;
        F(1, 5) = -dist_step * s_ph * s_th;

        // Derivatives for Position Z
        F(2, 3) = d_dist_dv * s_ph;
        F(2, 6) = d_dist_da * s_ph;
        F(2, 5) = dist_step * c_ph;

        // Derivatives for Velocity & Angles
        F(3, 6) = dt;    // dv/da
        F(4, 7) = dt;    // dth/dth_d
        F(5, 8) = dt;    // dph/dph_d

        // 5. Update Covariance
        P = F * P * F.transpose() + Q;
    }

    void setQ(double dt, double theta) {
        Q.setZero();
        double dt2 = dt * dt;
        double dt3 = dt2 * dt;
        double dt4 = dt3 * dt;

        // Trig values for current heading
        double c_th = cos(theta);
        double s_th = sin(theta);

        // =======================================================
        // 1. Longitudinal Acceleration Noise
        // (Pushes velocity, which in turn pushes X and Y based on heading)
        // =======================================================
        
        // Position-Position Covariance
        Q(0, 0) = 0.25 * dt4 * var_a_ * c_th * c_th; // x, x
        Q(1, 1) = 0.25 * dt4 * var_a_ * s_th * s_th; // y, y
        Q(0, 1) = 0.25 * dt4 * var_a_ * c_th * s_th; // x, y (Cross-correlation)
        Q(1, 0) = Q(0, 1);

        // Position-Velocity Covariance
        Q(0, 3) = 0.5 * dt3 * var_a_ * c_th;         // x, v
        Q(3, 0) = Q(0, 3);
        Q(1, 3) = 0.5 * dt3 * var_a_ * s_th;         // y, v
        Q(3, 1) = Q(1, 3);

        // Velocity-Velocity Covariance
        Q(3, 3) = dt2 * var_a_;                      // v, v

        // =======================================================
        // 2. Yaw Acceleration Noise
        // (Pushes the turn rate, which in turn pushes the heading)
        // =======================================================
        
        Q(4, 4) = 0.25 * dt4 * var_yaw_;             // theta, theta
        Q(4, 7) = 0.5 * dt3 * var_yaw_;              // theta, theta_dot
        Q(7, 4) = Q(4, 7);
        Q(7, 7) = dt2 * var_yaw_;                    // theta_dot, theta_dot

        // =======================================================
        // 3. Baseline Noise for Uncorrelated States
        // (Prevents matrix singularities in Z, Accel, and Phi)
        // =======================================================
        Q(2, 2) = 0.01;  // z
        Q(5, 5) = 0.01;  // phi
        Q(6, 6) = 0.01;  // a 
        Q(8, 8) = 0.01;  // phi_dot
    }
};
