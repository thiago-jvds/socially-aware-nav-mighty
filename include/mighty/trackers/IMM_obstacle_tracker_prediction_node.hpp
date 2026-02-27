#ifndef IMM_OBSTACLE_TRACKER_PREDICTION_NODE_HPP_
#define IMM_OBSTACLE_TRACKER_PREDICTION_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <dynus_interfaces/msg/dyn_traj.hpp>
#include <mighty/mighty_type.hpp>
#include <mighty/utils.hpp>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include "base_EKF_tracker.hpp"
#include <math.h>

const int NUM_MODES = 3;

// --- IMM Struct ---
// Holds the history/prediction of an object
struct IMMTrack {
    int id;
    double time_last_updated;
    Eigen::Vector3d bbox;
    std_msgs::msg::ColorRGBA color;

    // The Overall Combined State
    Eigen::VectorXd x;    // [x, y, z, v, theta, phi, a, theta_dot, phi_dot]  
    Eigen::MatrixXd P; 
    Eigen::MatrixXd R;     

    // The Independent Models
    std::vector<std::shared_ptr<BaseEKFModel>> models;
    
    // IMM Math Vectors
    Eigen::VectorXd mode_probs;
    Eigen::VectorXd prev_mode_probs;
    Eigen::VectorXd likelihoods;
    Eigen::VectorXd c_bar;

    // TPM
    Eigen::MatrixXd trans_prob_mat_;

    bool is_first_meas;

    IMMTrack() {}
    IMMTrack(int state_dim, int meas_dim, double time, const Eigen::Vector3d& centroid, const Eigen::Vector3d& bbox_in,
        int id_in, double sigma_a_CA, double sigma_yaw_CA, double prob_transition_stay) {
        id = id_in++;
        time_last_updated = time;
        bbox = bbox_in;
        is_first_meas = true;

        // Setup Main State
        x = Eigen::VectorXd::Zero(state_dim);
        x.head(3) = centroid;

        // Set P
        P = Eigen::MatrixXd::Identity(state_dim, state_dim);
        P.block<3,3>(0, 0) *= 0.5;
        P(3, 3) = 25.0; // v
        P(4, 4) = M_PI; // th
        P(6, 6) = 5.0;  // a
        
        // Set R
        R = Eigen::MatrixXd::Identity(3, 3);
        R(0,0) = 0.1; // X noise
        R(1,1) = 0.1; // Y noise
        R(2,2) = 0.1; // Z noise

        // Instantiate Models
        int num_modes = NUM_MODES; 
        
        // Push your specific derived models
        models.push_back(std::make_shared<CVModel>(state_dim, meas_dim, 0.01, sigma_yaw_CA, MODE_FWD));
        models.push_back(std::make_shared<CAModel>(state_dim, meas_dim, sigma_a_CA, sigma_yaw_CA, MODE_FWD));
        models.push_back(std::make_shared<CTModel>(state_dim, meas_dim, 0.5, sigma_yaw_CA, MODE_FWD));

        // Sync initial state to all models
        for (auto& model : models) {
            model->x = x;
            model->setP(P);
            model->setR(R);
        }

        // Setup Probabilities
        mode_probs = Eigen::VectorXd::Zero(num_modes);
        mode_probs.fill(1.0 / num_modes); 

        prev_mode_probs = Eigen::VectorXd::Zero(num_modes);
        prev_mode_probs.fill(1.0 / num_modes); 
        
        likelihoods = Eigen::VectorXd::Zero(num_modes);
        c_bar = Eigen::VectorXd::Zero(num_modes);

        setColor();

        initializeTPM(num_modes, prob_transition_stay);
    }

    void setColor() {
        this->color.r = static_cast<float>(rand()) / RAND_MAX;
        this->color.g = static_cast<float>(rand()) / RAND_MAX;
        this->color.b = static_cast<float>(rand()) / RAND_MAX;
        this->color.a = 1.0;
    }

    void initializeTPM(int num_modes, double prob_transition_stay) {
        this->trans_prob_mat_ = Eigen::MatrixXd::Zero(num_modes, num_modes);

        double p_stay = prob_transition_stay; 
        double p_switch = (1.0 - p_stay) / (num_modes - 1); 

        for (int i = 0; i < num_modes; ++i) {
            for (int j = 0; j < num_modes; ++j) {
                this->trans_prob_mat_(i, j) = (i == j) ? p_stay : p_switch;
            }
        }
    }

    /*
     * Gao, Jianshu, et al. "Moving-Target Tracking in Airport Airside Operations Using AIMM-STUKF." Sensors 26.1 (2025): 166
     */
    void adaptTPM() {
        if (this->is_first_meas || this->mode_probs.isApprox(this->prev_mode_probs, 1e-6)) {
            this->prev_mode_probs = this->mode_probs;
            return;
        }
        
        int num_modes = static_cast<int>(models.size());
        double n = 0.6;
        double th = 0.9;

        for (int m1 = 0; m1 < num_modes; m1++) {
            double norm = 0.0;
            double delta_mu_m1 = this->mode_probs[m1] - this->prev_mode_probs[m1];

            for (int m2 = 0; m2 < num_modes; m2++) {
                double delta_mu_m2 = this->mode_probs[m2] - this->prev_mode_probs[m2];
                
                double correction_factor = (m1 == m2) ? std::pow(delta_mu_m2, n) : std::pow(delta_mu_m2 / delta_mu_m1, n);
                
                this->trans_prob_mat_(m1, m2) = correction_factor * this->trans_prob_mat_(m1, m2);
                norm += this->trans_prob_mat_(m1, m2);
            }
            
            // Normalize the row
            for (int m2 = 0; m2 < num_modes; m2++) {
                this->trans_prob_mat_(m1, m2) /= norm;
            }
            
            // Apply lower-bound threshold to the diagonal element
            if (this->trans_prob_mat_(m1, m1) < th) {
                double old_diag = this->trans_prob_mat_(m1, m1);
                this->trans_prob_mat_(m1, m1) = th;
                
                // Scale the off-diagonal elements so the row still sums to 1.0
                double off_diag_sum = 1.0 - old_diag;
                if (off_diag_sum > 0.0) {
                    double scale = (1.0 - th) / off_diag_sum;
                    for (int m2 = 0; m2 < num_modes; m2++) {
                        if (m1 != m2) {
                            this->trans_prob_mat_(m1, m2) *= scale;
                        }
                    }
                }
            }
        }
        this->prev_mode_probs = this->mode_probs;
    }
};

// --- Measurement Wrapper ---
// Holds the incoming data for this frame.
struct Measurement {
    IMMTrack assigned_track;
    Eigen::Vector3d centroid;
    bool has_match;
    Measurement() {}
};

// --- Node Class ---
class IMMObstacleTrackerPredictionNode : public rclcpp::Node
{
public:
    IMMObstacleTrackerPredictionNode();

private:
    // Callbacks
    void detectionsCallback(const vision_msgs::msg::Detection3DArray::SharedPtr msg);

    // 1. Interaction (Mixing Step)
    void interaction(IMMTrack& track, 
                         std::vector<Eigen::VectorXd>& x_mixed, 
                         std::vector<Eigen::MatrixXd>& P_mixed);

    // 2. Prediction (Mode-Specific Physics)
    void predict(IMMTrack& track, 
                     const std::vector<Eigen::VectorXd>& x_mixed, 
                     const std::vector<Eigen::MatrixXd>& P_mixed, 
                     double dt);
                     
    // 3. Update & 4. Probability Update (Measurement Integration)
    void update(IMMTrack& track, const Eigen::VectorXd& z, 
                    double alpha, double time_updated, 
                    const Eigen::Vector3d &bbox);

    // 5. Combination (Output Calculation)
    void combine(IMMTrack& track);

    // --- Associate and Lifecycle Functions ---
    void deleteOldTracks();
    std::vector<int> associateMeasurements(
        const std::vector<Measurement>& new_detections,
        const std::vector<IMMTrack>& tracks,
        double tolerance);
    
    // --- DynTraj Msgs creation ---       
    Eigen::VectorXd polyfit(const std::vector<double>& t, const std::vector<double>& y, int degree);
    double calculateVariance(const std::vector<double>& t, const std::vector<double>& y, const Eigen::VectorXd& beta, int degree);
    std::vector<std::pair<double, Eigen::Vector3d>> generatePrediction(const IMMTrack& track, int best_mode);
    std::vector<std::pair<double, Eigen::Vector3d>> generateMergedPrediction(const IMMTrack& track); 
    void publishPredictions(const std::vector<Measurement> &measurements);    

    // ROS Interfaces
    rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr sub_detections_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_markers_;
    rclcpp::Publisher<dynus_interfaces::msg::DynTraj>::SharedPtr pub_predicted_traj_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pred_pos_pub_;
    rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr pred_vel_pub_;

    // State Variables
    std::vector<IMMTrack> tracks_;
    int track_id_ = 0;

    // IMM Variables
    Eigen::MatrixXd trans_prob_mat_;      // Transition Probability Matrix (4x4)

    double degree_for_pwp_ = 3;
    double degree_for_poly_ = 5;
    int state_dim_ = 9;
    int meas_dim_ = 3;      // x, y, z only

    std::string frame_id_ = "map";
    
    bool tracker_debug_ = true;
    
    // Parameters
    double cluster_tolerance_;
    double adaptive_kf_dt_;
    double adaptive_kf_alpha_;
    bool use_adaptive_kf_;
    double velocity_threshold_;
    double acceleration_threshold_; // Added for static filtering
    double prediction_horizon_;
    double prediction_dt_;
    double time_to_delete_old_obstacles_;
    bool use_life_time_for_box_visualization_;
    double box_visualization_duration_;
    double dynus_map_res_;
    int visual_level_;
    std::string tracking_frame_;
    
    // IMM specific params
    double prob_transition_stay_; // Prob of staying in current mode
    double prob_transition_switch_; // Prob of switching

    double sigma_a_CA_      = 1.5;
    double sigma_yaw_CA_    = 0.5;
    double fixed_yaw_rate_    = 0.6;
};

#endif // IMM_OBSTACLE_TRACKER_PREDICTION_NODE_HPP_