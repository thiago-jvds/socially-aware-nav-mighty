#ifndef IMM_OBSTACLE_TRACKER_NODE_HPP_
#define IMM_OBSTACLE_TRACKER_NODE_HPP_

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <dynus_interfaces/msg/dyn_traj.hpp>
#include <dynus_interfaces/msg/state.hpp>
#include <mighty/mighty_type.hpp>

#include "base_KF_tracker.hpp"

constexpr int NUM_MODES = 2;

struct IMMParams {
    double prob_transition_stay = 0.90;
    double p_init_pos_var = 0.5;
    double p_init_vel_var = 1.0;
    double p_init_acc_var = 5.0;
    double r_meas_pos_var = 0.1;
    double model_noise_cv = 0.8;
    double model_noise_ca = 0.5;
    double model_noise_sta = 0.05;
    bool imm_adapt_tpm = true;
    double imm_adapt_tpm_gain = 0.9;
};

struct IMMTrack {
    int id = 0;
    double time_last_updated = 0.0;
    Eigen::Vector3d bbox = Eigen::Vector3d::Zero();
    std_msgs::msg::ColorRGBA color;

    // Combined IMM state.
    Eigen::VectorXd x;
    Eigen::MatrixXd P;
    Eigen::MatrixXd R;

    // Mode-specific filter models.
    std::vector<std::shared_ptr<IMMTrackerModel>> models;

    // IMM probability bookkeeping.
    Eigen::VectorXd mode_probs;
    Eigen::VectorXd prev_mode_probs;
    Eigen::VectorXd likelihoods;
    Eigen::VectorXd c_bar;
    Eigen::MatrixXd trans_prob_mat_;

    bool is_first_meas = true;
    bool tpm_adaptation_enabled_ = true;
    double tpm_adaptation_gain_ = 0.9;

    IMMTrack() = default;

    IMMTrack(
        int state_dim,
        int meas_dim,
        double time,
        const Eigen::Vector3d& centroid,
        const Eigen::Vector3d& bbox_in,
        int id_in,
        const IMMParams& imm_params)
    {
        id = id_in;
        time_last_updated = time;
        bbox = bbox_in;
        is_first_meas = true;

        x = Eigen::VectorXd::Zero(state_dim);
        x.head(3) = centroid;

        P = Eigen::MatrixXd::Identity(state_dim, state_dim);
        P.block<3, 3>(0, 0) *= imm_params.p_init_pos_var;
        P.block<3, 3>(3, 3) *= imm_params.p_init_vel_var;
        P.block<3, 3>(6, 6) *= imm_params.p_init_acc_var;

        R = Eigen::MatrixXd::Identity(meas_dim, meas_dim);
        R(0, 0) = imm_params.r_meas_pos_var;
        R(1, 1) = imm_params.r_meas_pos_var;
        R(2, 2) = imm_params.r_meas_pos_var;

        models.push_back(std::make_shared<KFTrackers::CVModel>(state_dim, meas_dim, imm_params.model_noise_cv));
        models.push_back(std::make_shared<KFTrackers::CAModel>(state_dim, meas_dim, imm_params.model_noise_ca));
        // models.push_back(std::make_shared<KFTrackers::StationaryModel>(state_dim, meas_dim, imm_params.model_noise_sta));

        for (auto& model : models) {
            model->x = x;
            model->setP(P);
            model->setR(R);
        }

        mode_probs = Eigen::VectorXd::Constant(NUM_MODES, 1.0 / static_cast<double>(NUM_MODES));
        prev_mode_probs = mode_probs;
        likelihoods = Eigen::VectorXd::Zero(NUM_MODES);
        c_bar = Eigen::VectorXd::Zero(NUM_MODES);

        setColor();
        initializeTPM(NUM_MODES, imm_params.prob_transition_stay);

        tpm_adaptation_enabled_ = imm_params.imm_adapt_tpm;
        tpm_adaptation_gain_ = imm_params.imm_adapt_tpm_gain;
    }

    void setColor()
    {
        color.r = static_cast<float>(rand()) / RAND_MAX;
        color.g = static_cast<float>(rand()) / RAND_MAX;
        color.b = static_cast<float>(rand()) / RAND_MAX;
        color.a = 1.0f;
    }

    void initializeTPM(int num_modes, double prob_transition_stay)
    {
        this->trans_prob_mat_ = Eigen::MatrixXd::Zero(num_modes, num_modes);

        // this->trans_prob_mat_ << 
        //     0.80, 0.15, 0.05,  // From CV: 15% chance to enter acceleration/deceleration
        //     0.20, 0.60, 0.20,  // From CA: 20% to reach a full stop, 20% to return to steady CV
        //     0.10, 0.20, 0.70;  // From STOP: 20% chance to start moving again (via CA)
        this->trans_prob_mat_ << 0.90, 0.10,
                                 0.20, 0.80;
    }

    void adaptTPM()
    {
        if (!tpm_adaptation_enabled_) {
            prev_mode_probs = mode_probs;
            return;
        }

        const int num_modes = static_cast<int>(models.size());
        const double n = std::clamp(tpm_adaptation_gain_, 0.0, 1.0);

        for (int m1 = 0; m1 < num_modes; ++m1) {
            double norm = 0.0;
            const double delta_mu_m1 = std::exp(mode_probs[m1] - prev_mode_probs[m1]);

            for (int m2 = 0; m2 < num_modes; ++m2) {
                const double delta_mu_m2 = std::exp(mode_probs[m2] - prev_mode_probs[m2]);

                const double correction_factor =
                    (m1 == m2) ? std::pow(delta_mu_m2, n) : std::pow(delta_mu_m2 / delta_mu_m1, n);

                trans_prob_mat_(m1, m2) = correction_factor * trans_prob_mat_(m1, m2);
                norm += trans_prob_mat_(m1, m2);
            }

            if (norm > 1e-9) {
                for (int m2 = 0; m2 < num_modes; ++m2) {
                    trans_prob_mat_(m1, m2) /= norm;
                }
            }
        }

        prev_mode_probs = mode_probs;
    }
};

struct Measurement {
    IMMTrack assigned_track;
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    builtin_interfaces::msg::Time source_stamp;
    bool has_match = false;
};

class IMMObstacleTrackerNode : public rclcpp::Node
{
public:
    IMMObstacleTrackerNode();

private:
    void stateCallback(const dynus_interfaces::msg::State::SharedPtr msg); 
    void detectionsCallback(const vision_msgs::msg::Detection3DArray::SharedPtr msg);

    void interaction(
        IMMTrack& track,
        std::vector<Eigen::VectorXd>& x_mixed,
        std::vector<Eigen::MatrixXd>& P_mixed);

    void predict(
        IMMTrack& track,
        const std::vector<Eigen::VectorXd>& x_mixed,
        const std::vector<Eigen::MatrixXd>& P_mixed,
        double dt);

    void update(
        IMMTrack& track,
        const Eigen::VectorXd& z,
        double time_updated,
        const Eigen::Vector3d& bbox);

    void combine(IMMTrack& track);

    void deleteOldTracks(double reference_time_sec);

    std::vector<int> associateMeasurements(
        const std::vector<Measurement>& new_detections,
        const std::vector<IMMTrack>& tracks,
        double tolerance);

    void publishPredictions(
        const std::vector<Measurement>& measurements,
        double current_time_sec);

    rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr sub_detections_;
    rclcpp::Subscription<dynus_interfaces::msg::State>::SharedPtr sub_state_;

    rclcpp::Publisher<dynus_interfaces::msg::DynTraj>::SharedPtr predicted_traj_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr tracker_bbox_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr tracker_prediction_pub_;

    std::vector<IMMTrack> tracks_;
    int track_id_ = 0;

    int state_dim_ = 9;
    int meas_dim_ = 3;

    std::string frame_id_ = "map";

    double assignment_tolerance_ = 2.0;
    double prediction_horizon_ = 3.0;
    double prediction_dt_ = 0.1;
    double time_to_delete_old_obstacles_ = 3.0;
    double box_visualization_duration_ = 3.0;
    double velocity_threshold_ = 0.0;
    double acceleration_threshold_ = 0.1;
    bool tracker_debug_ = false;
    double adaptive_kf_dt_ = 0.1;

    bool report_min_dist_to_ego_ = true;
    double min_so_far = std::numeric_limits<double>::max();
    bool state_initialized_ = false;
    dynus_interfaces::msg::State current_state_;

    IMMParams imm_params_;
};

#endif // IMM_OBSTACLE_TRACKER_NODE_HPP_
