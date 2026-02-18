#ifndef CA_OBSTACLE_TRACKER_PREDICTION_NODE_HPP_
#define CA_OBSTACLE_TRACKER_PREDICTION_NODE_HPP_

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

// --- EKF State Struct ---
// Holds the history/prediction of an object
struct EKFState {
    Eigen::VectorXd x;      // State vector [x, y, z, x_dot, y_dot, z_dot, x_ddot, y_ddot, z_ddot]
    Eigen::MatrixXd P;      // Covariance matrix
    Eigen::MatrixXd Q;      // Process noise
    Eigen::MatrixXd R;      // Measurement noise
    double time_last_updated;
    Eigen::Vector3d bbox;   // Dimensions [length, width, height]
    int id;
    std_msgs::msg::ColorRGBA color;

    EKFState() {} // Constructor for Cluster struct
    EKFState(int state_dim, const Eigen::MatrixXd& Q_in, const Eigen::MatrixXd& R_in, 
             double time, const Eigen::Vector3d& bbox_in, int id_in) {

        x = Eigen::VectorXd::Zero(state_dim);
        P = Eigen::MatrixXd::Identity(state_dim, state_dim);
        Q = Q_in;
        R = R_in;
        time_last_updated = time;
        bbox = bbox_in;
        setColor();
    }

    void setColor() {
        this->color.r = static_cast<float>(rand()) / RAND_MAX;  // Random red
        this->color.g = static_cast<float>(rand()) / RAND_MAX;  // Random green
        this->color.b = static_cast<float>(rand()) / RAND_MAX;  // Random blue
        // If you want to set a specific color
        // this->color.r = 0.0 / 255.0;
        // this->color.g = 0.0 / 255.0;
        // this->color.b = 255.0 / 255.0;
        this->color.a = 1.0;  // Opacity
    }
};

// --- Measurement Wrapper ---
// Holds the incoming data for this frame.
struct Measurement {
    EKFState assigned_ekf_state;
    Eigen::Vector3d centroid;
    bool has_match;
    Measurement() {}
};

// --- Node Class ---
class CAObstacleTrackerPredictionNode : public rclcpp::Node
{
public:
    CAObstacleTrackerPredictionNode();

private:
    // Callbacks
    void detectionsCallback(const vision_msgs::msg::Detection3DArray::SharedPtr msg);

    // EKF Core Math
    void deleteOldEKFstates();
    std::vector<int> associate_measurements(
        const std::vector<Measurement>& new_detections,
        const std::vector<EKFState>& ekf_states,
        double tolerance);
    void ekf_predict(EKFState& ekf_state, double dt);
    void aekf_update(EKFState& ekf_state, const Eigen::VectorXd& z, 
                     double alpha, double time_updated, 
                     const Eigen::Vector3d& bbox, bool use_adaptive_kf);
    void calculateAverageQandR(Eigen::MatrixXd& Q_avg, Eigen::MatrixXd& R_avg);
    Eigen::VectorXd polyfit(const std::vector<double>& t, const std::vector<double>& y, int degree);
    double calculateVariance(const std::vector<double>& t, const std::vector<double>& y, const Eigen::VectorXd& beta, int degree);
    void publishPredictions(const std::vector<Measurement> &measurements);


    // ROS Interfaces
    rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr sub_detections_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_markers_;
    rclcpp::Publisher<dynus_interfaces::msg::DynTraj>::SharedPtr pub_predicted_traj_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pred_pos_pub_;
    rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr pred_vel_pub_;

    // State Variables
    std::vector<EKFState> ekf_states_;
    int ekf_state_id_ = 0;

    double degree_for_pwp_ = 3;
    double degree_for_poly_ = 5;

    std::string frame_id_ = "map";
    
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
};

#endif // CA_OBSTACLE_TRACKER_PREDICTION_NODE_HPP_