#pragma once

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>
#include <std_msgs/msg/color_rgba.hpp>
#include <iostream>


enum Mode {
    MODE_FWD = 1,
    MODE_LEFT = 2,
    MODE_RIGHT = 3
};

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

        // force positive velocity
        if (x(3) < -0.1) {
            x(3) = std::abs(x(3));
            x(4) = x(4) + M_PI;                  // Instant 180 flip

            // Clip Acc and Th_d
            x(6) = 0.0;
            x(7) = 0.0;
        } else if (x(3) < 0.0) {
            x(3) = std::abs(x(3));
            
            // Clip Acc and Th_d
            x(6) = 0.0;
            x(7) = 0.0;
        }

        // Normalize the angles to prevent IMM mixing explosions
        x(4) = normalize_angle(x(4)); 
        x(5) = normalize_angle(x(5));

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

    Eigen::MatrixXd getF(
        double th, double ph, double d_dist_dv, double d_dist_da, 
        double d_v_da, double d_a_da, double dist_step, double dt) {
        Eigen::MatrixXd F = Eigen::MatrixXd::Identity(state_dim_, state_dim_);
        
        double s_th = sin(th), c_th = cos(th);
        double s_ph = sin(ph), c_ph = cos(ph);

        F(0, 3) = d_dist_dv * c_ph * c_th;          // dx_new/dv
        F(0, 6) = d_dist_da * c_ph * c_th;          // dx_new/da
        F(0, 4) = -dist_step * c_ph * s_th;         // dx_new/dtheta
        F(0, 5) = -dist_step * s_ph * c_th;         // dx_new/dphi
        
        F(1, 3) = d_dist_dv * c_ph * s_th;          // dy_new/dv
        F(1, 6) = d_dist_da * c_ph * s_th;          // dy_new/da
        F(1, 4) = dist_step * c_ph * c_th;          // dy_new/dtheta
        F(1, 5) = -dist_step * s_ph * s_th;         // dy_new/dphi

        F(2, 3) = d_dist_dv * s_ph;                 // dz_new/dv
        F(2, 6) = d_dist_da * s_ph;                 // dz_new/da
        F(2, 5) = dist_step * c_ph;                 // dz_new/dphi

        F(3, 6) = d_v_da;                           // dv_new/da
        F(6, 6) = d_a_da;                           // da_new/da

        F(4, 7) = dt;                               // dtheta_new/dtheta_d
        F(5, 8) = dt;                               // dphi_new/dphi_d

        return F;
    }
};

/*
 * a = 0, θ̇ = 0
 */
class CVModel : public BaseEKFModel {
private:
    double var_v_;
    double var_yaw_;
    Mode mode_;
public:
    CVModel(int state_dim, int meas_dim, double sigma_v, double sigma_yaw_rate, Mode mode) 
    : BaseEKFModel(state_dim, meas_dim),
      var_v_ (sigma_v * sigma_v),
      var_yaw_ (sigma_yaw_rate * sigma_yaw_rate),
      mode_ (mode) {}

    void setQ(double dt) {
        Q.setZero();
        double dt2 = dt * dt;
        double dt3 = dt2 * dt;

        // Position noise 
        Q(0,0) = 0.5 * dt3 * var_v_;    // x
        Q(1,1) = 0.5 * dt3 * var_v_;    // y
        Q(2,2) = 0.5 * dt3 * var_v_;    // z
        
        // Velocity noise
        Q(3,3) = dt2 * var_v_;          // v
        
        // Acceleration noise is strictly ZERO for a pure CV model
        Q(6,6) = 1e-6;

        // Angular Rate Coupling
        Q(4,4) = dt2 * var_yaw_;     // theta
        Q(7,7) = 1e-6;               // theta_dot
    };

    void predict(double dt) override {
        double px = x(0), py = x(1), pz = x(2);
        double v  = x(3), th = x(4), ph = x(5);

        if (v <= 0.0) {
            std::cout << "CV got negative velocity" << std::endl;
        }

        setQ(dt);

        // 1. Predict Kinematics
        double dist_step = v * dt;
        
        x(0) = px + dist_step * cos(ph) * cos(th);
        x(1) = py + dist_step * cos(ph) * sin(th);
        x(2) = pz + dist_step * sin(ph);
        x(3) = v;                    // v constant
        x(4) = normalize_angle(th);  // th constant
        x(5) = normalize_angle(ph);  // ph constant

        // a = 0
        x(6) = 0.0; 

        // θ̇ = 0
        x(7) = 0.0;

        // No phi for now
        x(8) = 0.0;

        // 2. Jacobian Matrix (F)
        double d_dist_dv = dt;
        double d_dist_da = 0.0;      
        double d_v_da    = 0.0;      
        double d_a_da    = 0.0;     

        Eigen::MatrixXd F = getF(th, ph, d_dist_dv, d_dist_da, d_v_da, d_a_da, dist_step, dt);        

        F(3, 6) = 0.0; // dv/da     = 0
        F(4, 7) = 0.0; // dth/dth_d = 0
        F(5, 8) = 0.0; // dph/dph_d = 0
        F(6, 6) = 0.0; // da/da     = 0

        // 3. Update Covariance
        P = F * P * F.transpose() + Q;
    }
};


/*
* a ≠ 0, θ = 0
*/
class CAModel : public BaseEKFModel {
private:
    double var_a_;
    double var_yaw_;
    Mode mode_;
public:
    CAModel(int state_dim, int meas_dim, double sigma_a, double sigma_yaw_rate, Mode mode) 
    : BaseEKFModel(state_dim, meas_dim),
      var_a_ (sigma_a * sigma_a),
      var_yaw_ (sigma_yaw_rate * sigma_yaw_rate),
      mode_ (mode) {}

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

        // Acceleration noise
        Q(6,6) = var_a_;
        
        // Angular Rate Coupling
        Q(4,4) = dt2 * var_yaw_;        // theta
        Q(7,7) = 1e-6;                  // theta_dot
    };

    void predict(double dt) override {
        double px = x(0), py   = x(1), pz   = x(2);
        double v  = x(3), th   = x(4), ph   = x(5);
        double a  = x(6); 

        if (v <= 0.0) {
            std::cout << "CA got negative velocity" << std::endl;
        }

        setQ(dt);

        // 1. Predict Kinematics
        double v_new = v + a * dt;
        
        // v can't be negative
        if (v_new < 0.0) {
            v_new = 0.0;
            a = (0.0 - v) / dt; // Recalculate 'a' so it perfectly stops the target at v=0
        }

        double dist_step = v * dt + 0.5 * a * dt * dt;
        
        x(0) = px + dist_step * cos(ph) * cos(th);
        x(1) = py + dist_step * cos(ph) * sin(th);
        x(2) = pz + dist_step * sin(ph);
        x(4) = th;
        x(5) = ph;
        
        // a ≠ 0
        x(3) = v_new;
        x(6) = a;

        // θ̇ = 0
        x(7) = 0.0;

        // No phi for now
        x(8) = 0.0;

        // 2. Jacobian Matrix (F)
        double d_dist_dv = dt;
        double d_dist_da = 0.5 * dt * dt;
        double d_v_da    = dt;
        double d_a_da    = 1.0; // Acceleration is constant, new_a = old_a
        
        Eigen::MatrixXd F = getF(th, ph, d_dist_dv, d_dist_da, d_v_da, d_a_da, dist_step, dt);

        // 3. Update Covariance
        P = F * P * F.transpose() + Q;
    }
};

/*
 * a = 0, θ̇ ≠ 0
 */
class CTModel : public BaseEKFModel {
private:
    double var_v_;
    double var_yaw_;
    Mode mode_;
public:
    CTModel(int state_dim, int meas_dim, double sigma_a, double sigma_yaw_rate, Mode mode) 
    : BaseEKFModel(state_dim, meas_dim),
      var_v_ (sigma_a * sigma_a),
      var_yaw_ (sigma_yaw_rate * sigma_yaw_rate),
      mode_ (mode) {}

    void setQ(double dt) {
        Q.setZero();
        double dt2 = dt * dt;
        double dt3 = dt2 * dt;

        // Position noise 
        Q(0,0) = 0.5 * dt3 * var_v_;    // x
        Q(1,1) = 0.5 * dt3 * var_v_;    // y
        Q(2,2) = 0.5 * dt3 * var_v_;    // z
        
        // Velocity noise
        Q(3,3) = dt2 * var_v_;          // v
        
        // Acceleration noise is strictly ZERO for a pure CV model
        Q(6,6) = 1e-6;

        // Angular Rate Coupling
        Q(4,4) = dt2 * var_yaw_;         // theta
        Q(7,7) = var_yaw_;               // theta_dot
    };

    void predict(double dt) override {
        double px = x(0), py   = x(1), pz   = x(2);
        double v  = x(3), th   = x(4), ph   = x(5); 
        double th_d = x(7);

        if (v <= 0.0) {
            std::cout << "CT got negative velocity" << std::endl;
        }

        setQ(dt);

        // 1. Predict Kinematics
        double dist_step = v * dt;
        
        x(0) = px + dist_step * cos(ph) * cos(th);
        x(1) = py + dist_step * cos(ph) * sin(th);
        x(2) = pz + dist_step * sin(ph);
        x(4) = normalize_angle(th + th_d * dt);
        x(5) = ph;
        
        // a = 0
        x(3) = v;
        x(6) = 0.0;

        // θ̇ ≠ 0
        x(7) = th_d;

        // No phi for now
        x(8) = 0.0;

        // 2. Jacobian Matrix (F)
        double d_dist_dv = dt;
        double d_dist_da = 0.5 * dt * dt;
        double d_v_da    = dt;
        double d_a_da    = 1.0; // Acceleration is constant, new_a = old_a
        
        Eigen::MatrixXd F = getF(th, ph, d_dist_dv, d_dist_da, d_v_da, d_a_da, dist_step, dt);

        F(3, 6) = 0.0;  // dv/da     = 0 
        F(6, 6) = 0.0;  // da/da     = 0
  
        // 3. Update Covariance
        P = F * P * F.transpose() + Q;
    }
};
