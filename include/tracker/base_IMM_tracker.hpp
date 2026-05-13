#pragma once

#include <Eigen/Dense>

class IMMTrackerModel {
public:
    int state_dim_;
    int meas_dim_;

    Eigen::VectorXd x; 
    Eigen::MatrixXd P;
    Eigen::MatrixXd Q;
    Eigen::MatrixXd R;
    Eigen::MatrixXd H;
    double likelihood = 1e-30;

    IMMTrackerModel(int state_dim, int meas_dim) 
        : state_dim_(state_dim), meas_dim_(meas_dim) {
        x = Eigen::VectorXd::Zero(state_dim_);
        P = Eigen::MatrixXd::Identity(state_dim_, state_dim_);
        Q = Eigen::MatrixXd::Identity(state_dim_, state_dim_);
        R = Eigen::MatrixXd::Identity(meas_dim_, meas_dim_);
        H = Eigen::MatrixXd::Zero(meas_dim_, state_dim_);
        
        // Standard 3D Position Measurement Matrix mapping to [x, y, z]
        H(0, 0) = 1.0; 
        H(1, 1) = 1.0; 
        H(2, 2) = 1.0;
    }

    virtual ~IMMTrackerModel() = default;
    virtual void predict(double dt) = 0;
    virtual void update(const Eigen::VectorXd& z) = 0;

    void setP(const Eigen::MatrixXd& P_in) { this->P = P_in; }
    void setR(const Eigen::MatrixXd& R_in) { this->R = R_in; }
};