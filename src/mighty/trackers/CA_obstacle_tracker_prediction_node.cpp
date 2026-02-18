// =================================================================================
//   Constant Acceleration (CA) Obstacle Tracker & Prediction Node
//  
//   This node implements a high-fidelity multi-object tracking system using a 
//   parabolic Constant Acceleration (CA) motion model wrapped in an 
//   Adaptive Extended Kalman Filter (AEKF).
//  
//   Key Algorithms & Features:
//   * 1. Constant Acceleration Model:
//       - State space is 9 dimensions: [x, y, z, vx, vy, vz, ax, ay, az].
//       - Explicitly estimates acceleration, allowing for accurate tracking 
//         of objects that are speeding up, slowing down, or turning.
//       - Transition Matrix (F) integrates acceleration: p_new = p + v*dt + 0.5*a*dt^2.
//   * 2. EKF State Estimation:
//       - Maintains a 9x9 covariance matrix (P) to track uncertainty across 
//         all kinematic derivatives (pos, vel, acc).
//       - Implements an "Adaptive" update step (AEKF) that dynamically adjusts 
//         process noise (Q) and measurement noise (R), handling sudden maneuvers better.
//   * 3. Data Association:
//       - Matches incoming Detection3D measurements to existing EKF states.
//       - Uses a Euclidean distance cost function with a gating threshold 
//         (cluster_tolerance) to solve the assignment problem.
//   * 4. Parabolic Trajectory Prediction:
//       - Projects future states using the full kinematic equation (including acceleration).
//       - Generates curved trajectory predictions suitable for dynamic agents.
// 
//   Publishes: visualization_msgs::MarkerArray (3D Boxes, Velocity Text, & IDs)
//   Publishes: dynus_interfaces::msg::DynTraj (Polynomial coefficients for planning)
//
//   Subscribes: vision_msgs::msg::Detection3DArray (Raw perception bounding boxes)
// =================================================================================

#include "mighty/trackers/CA_obstacle_tracker_prediction_node.hpp"
#include <cmath>

using std::placeholders::_1;

CAObstacleTrackerPredictionNode::CAObstacleTrackerPredictionNode() 
: Node("CA_obstacle_tracker_prediction_node") 
{

    // 1. Parameters
    this->declare_parameter("visual_level", 1);
    this->declare_parameter("use_adaptive_kf", false);
    this->declare_parameter("adaptive_kf_alpha", 0.98);
    this->declare_parameter("adaptive_kf_dt", 0.1);
    this->declare_parameter("cluster_tolerance", 2.0);
    this->declare_parameter("prediction_horizon", 2.0);
    this->declare_parameter("prediction_dt", 0.1);
    this->declare_parameter("time_to_delete_old_obstacles", 10.0);
    this->declare_parameter("use_life_time_for_box_visualization", false);
    this->declare_parameter("box_visualization_duration", 3.0);
    this->declare_parameter("dynus_map_res", 0.5);
    this->declare_parameter("velocity_threshold", 0.0);
    this->declare_parameter("acceleration_threshold", 0.1);

    // Set parameters
    visual_level_ = this->get_parameter("visual_level").as_int();
    use_adaptive_kf_ = this->get_parameter("use_adaptive_kf").as_bool();
    adaptive_kf_alpha_ = this->get_parameter("adaptive_kf_alpha").as_double();
    adaptive_kf_dt_ = this->get_parameter("adaptive_kf_dt").as_double();
    cluster_tolerance_ = this->get_parameter("cluster_tolerance").as_double();
    prediction_horizon_ = this->get_parameter("prediction_horizon").as_double();
    prediction_dt_ = this->get_parameter("prediction_dt").as_double();
    time_to_delete_old_obstacles_ = this->get_parameter("time_to_delete_old_obstacles").as_double();
    use_life_time_for_box_visualization_ = this->get_parameter("use_life_time_for_box_visualization").as_bool();
    box_visualization_duration_ = this->get_parameter("box_visualization_duration").as_double();
    dynus_map_res_ = this->get_parameter("dynus_map_res").as_double();
    velocity_threshold_ = this->get_parameter("velocity_threshold").as_double();
    acceleration_threshold_ = this->get_parameter("acceleration_threshold").as_double();

    // 2. Pub/Sub
    sub_detections_ = this->create_subscription<vision_msgs::msg::Detection3DArray>(
        "detected_objects", 10, 
        std::bind(&CAObstacleTrackerPredictionNode::detectionsCallback, this, _1));

    pub_markers_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("tracked_obstacles", 10);
    pub_predicted_traj_ = this->create_publisher<dynus_interfaces::msg::DynTraj>("predicted_trajs", 10);

    pred_pos_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("predicted_positions", 10);
    pred_vel_pub_ = this->create_publisher<geometry_msgs::msg::TwistStamped>("predicted_velocities", 10);

    RCLCPP_INFO(this->get_logger(), " CA Tracker & Prediction Initialized.");
}

void CAObstacleTrackerPredictionNode::detectionsCallback(const vision_msgs::msg::Detection3DArray::SharedPtr msg)
{
    // --- Step 1: Parse Inputs ---
    std::vector<Measurement> current_measurements;
    current_measurements.reserve(msg->detections.size());

    for (const auto& det : msg->detections) {
        Measurement m;
        m.centroid = Eigen::Vector3d(det.bbox.center.position.x, det.bbox.center.position.y, det.bbox.center.position.z);
        m.has_match = false;
        
        // [OPTIMIZATION] Store the RAW bbox directly in the EKFState container
        // effectively using it as temporary storage until we calculate the filter.
        m.assigned_ekf_state.bbox = Eigen::Vector3d(det.bbox.size.x, det.bbox.size.y, det.bbox.size.z);
        
        current_measurements.push_back(m);
    }

    // --- Step 2: Lifecycle ---
    deleteOldEKFstates();
    // Reserve memory to prevent invalidation during push_back
    ekf_states_.reserve(ekf_states_.size() + current_measurements.size());

    // --- Step 3: Predict for current tracks --
    for (auto& state : ekf_states_) {
        ekf_predict(state, adaptive_kf_dt_); 
    }

    // --- Step 4: Association & Update ---
    std::vector<int> assignments = associate_measurements(current_measurements, ekf_states_, cluster_tolerance_);
    
    for (int i = 0; i < static_cast<int>(assignments.size()); i++){
        int match_idx = assignments[i];
        auto& meas = current_measurements[i];
        Eigen::Vector3d raw_bbox = meas.assigned_ekf_state.bbox;

        if (match_idx >= 0) {
            
            // Pass 'raw_bbox' to the update function
            aekf_update(ekf_states_[match_idx], meas.centroid, adaptive_kf_alpha_, 
                        this->now().seconds(), raw_bbox, use_adaptive_kf_);
            
            // Overwrite the placeholder state with the REAL filtered state
            meas.assigned_ekf_state = ekf_states_[match_idx];
            meas.has_match = true;
        
        } else {

            // NEW TRACK
            Eigen::MatrixXd Q_avg(9,9), R_avg(3,3);
            calculateAverageQandR(Q_avg, R_avg);
            
            // Use 'raw_bbox' to initialize the new state
            EKFState new_state(9, Q_avg, R_avg, this->now().seconds(), raw_bbox, ekf_state_id_++);
            new_state.x = Eigen::VectorXd::Zero(9); 
            new_state.x.head(3) = meas.centroid;

            new_state.P.block(0, 0, 3, 3) *= 1.0;   // initial position uncertainty
            new_state.P.block(3, 3, 3, 3) *= 10.0;  // initial velocity uncertainty
            new_state.P.block(6, 6, 3, 3) *= 10.0;  // initial acceleration uncertainty
            
            ekf_states_.push_back(new_state);
            
            // Store the new state in the measurement
            meas.assigned_ekf_state = new_state;
            meas.has_match = true;
        }
    }

    // --- Step 5: Publish ---
    publishPredictions(current_measurements);
}


// =================================================================================
// EKF MATH IMPLEMENTATION
// =================================================================================

void CAObstacleTrackerPredictionNode::deleteOldEKFstates() {
    double current_time = this->now().seconds();
    
    auto it = ekf_states_.begin();
    while (it != ekf_states_.end()) {
        if ((current_time - it->time_last_updated) > time_to_delete_old_obstacles_) {
            it = ekf_states_.erase(it);
        } else {
            ++it;
        }
    }
}

/*   
 * @brief // O(max(current_meas.size(), ekf_states_.size())^3)
 * (for max_N ~ 10 detections, 1000 ops, negligible)
 */
std::vector<int> CAObstacleTrackerPredictionNode::associate_measurements(
    const std::vector<Measurement>& new_detections,
    const std::vector<EKFState>& ekf_states,
    double tolerance
){

    int M = new_detections.size();
    int N = ekf_states.size();

    const double INF = std::numeric_limits<double>::max();

    // No matches found if nothing detected, or all new traks
    if (M == 0 || N == 0) return std::vector<int>(M, -1);

    // Build cost

    // added one extra dim for convenience
    int dim = std::max(M, N);
    std::vector<std::vector<double>> cost_matrix(dim + 1, std::vector<double>(dim + 1, 0.0));

    for (int m = 0; m < M; m++){
        for (int n = 0; n < N; n++){
            cost_matrix[m + 1][n + 1] = (ekf_states[n].x.head(3) - new_detections[m].centroid).norm();
        }
    }
        
    // Solve
    std::vector<double> potential_row(dim + 1, 0);
    std::vector<double> potential_col(dim + 1, 0);

    std::vector<int> matches(dim + 1, 0);

    // predecessor_col[j] stores which column led to column j in the augmenting path.
    std::vector<int> predecessor_col(dim + 1, 0);

    for (int i = 1; i <= dim; ++i) {
        matches[0] = i; // Dummy column 0 holds the current row we are trying to match
        int current_col = 0;
        
        // 'min_slack[j]' tracks the smallest cost to add column j to the current tree
        std::vector<double> min_slack(dim + 1, INF);
        std::vector<bool> col_visited(dim + 1, false);

        // Find an augmenting path
        while (matches[current_col] != 0) { // Continue until we find an unassigned column
            col_visited[current_col] = true;
            int current_row = matches[current_col];
            double delta = INF;
            int next_col = 0;

            // Scan all columns to update slacks and find the next best move
            for (int j = 1; j <= dim; ++j) {
                if (!col_visited[j]) {
                    double reduced_cost = cost_matrix[current_row][j] - potential_row[current_row] - potential_col[j];
                    
                    if (reduced_cost < min_slack[j]) {
                        min_slack[j] = reduced_cost;
                        predecessor_col[j] = current_col;
                    }
                    
                    if (min_slack[j] < delta) {
                        delta = min_slack[j];
                        next_col = j;
                    }
                }
            }

            // Update potentials (dual variables) to create a new zero-cost edge
            for (int j = 0; j <= dim; ++j) {
                if (col_visited[j]) {
                    potential_row[matches[j]] += delta;
                    potential_col[j] -= delta;
                } else {
                    min_slack[j] -= delta;
                }
            }
            current_col = next_col;
        }

        // Backtrack to update the matching based on the path found
        while (current_col != 0) {
            int prev_col = predecessor_col[current_col];
            matches[current_col] = matches[prev_col];
            current_col = prev_col;
        } 
    }

    // Extract Results and Apply Tolerance
    std::vector<int> assignments(M, -1);

    for (int j = 1; j <= dim; ++j) {
        int assigned_row = matches[j];
        int assigned_col = j;

        // Check if indices are within the real data range (ignoring padding)
        if (assigned_row > 0 && assigned_row <= M &&
            assigned_col > 0 && assigned_col <= N) {
            
            // Even if the algorithm matched them, we reject if distance is too high
            if (cost_matrix[assigned_row][assigned_col] <= tolerance) {
                assignments[assigned_row - 1] = assigned_col - 1;
            }
        }
    }

    return assignments;
}

void CAObstacleTrackerPredictionNode::ekf_predict(EKFState &ekf_state, 
    double dt)
{
    int state_size = 9;

    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(state_size, state_size);
    // Position update with velocity: v*dt term
    F(0, 3) = dt;
    F(1, 4) = dt;
    F(2, 5) = dt;

    // Position update with acceleration: 0.5*a*dt^2
    F(0, 6) = 0.5 * dt * dt;
    F(1, 7) = 0.5 * dt * dt;
    F(2, 8) = 0.5 * dt * dt;

    // Velocity update with acceleration: a*dt
    F(3, 6) = dt;
    F(4, 7) = dt;
    F(5, 8) = dt;

    // Predict state element by element 
    ekf_state.x = F * ekf_state.x;
    // Predict covariance
    ekf_state.P = F * ekf_state.P.selfadjointView<Eigen::Lower>() * F.transpose() + ekf_state.Q;
}

void CAObstacleTrackerPredictionNode::aekf_update(EKFState &ekf_state, const Eigen::VectorXd &z, 
                                                 double alpha, double time_updated, 
                                                 const Eigen::Vector3d &bbox, bool use_adaptive_kf)
{
    int state_size = 9;
    Eigen::MatrixXd H; // Measurement matrix (we only measure position [x, y, z])
    H = Eigen::MatrixXd::Zero(3, state_size);
    H(0, 0) = 1;
    H(1, 1) = 1;
    H(2, 2) = 1;

    Eigen::VectorXd d = z - H * ekf_state.x;                           // Measurement residual
    Eigen::MatrixXd S = H * ekf_state.P * H.transpose() + ekf_state.R; // Residual covariance
    Eigen::MatrixXd K = ekf_state.P * H.transpose() * S.inverse();     // Kalman gain

    // Update state
    ekf_state.x = ekf_state.x + K * d;

    // residual
    Eigen::VectorXd epsilon = z - H * ekf_state.x;

    // Adaptive Kalman Filter
    if (use_adaptive_kf)
    {
        ekf_state.R = alpha * ekf_state.R + (1 - alpha) * (epsilon * epsilon.transpose() + H * ekf_state.P * H.transpose());
        ekf_state.Q = alpha * ekf_state.Q + (1 - alpha) * (K * d * d.transpose() * K.transpose());
    }
    else
    {
        ekf_state.R = Eigen::MatrixXd::Identity(3, 3) * 0.001;
        ekf_state.Q = Eigen::MatrixXd::Identity(9, 9) * 0.01;

        // Dubins -> from branch dynamic_acl_DYNUS
        // ekf_state.Q(3,3) *= 10;   // theta
        // ekf_state.Q(4,4) *= 10;   // phi

        // small process noise for position
        ekf_state.Q.block(0, 0, 3, 3) *= 0.01;

        // //moderate process noise for velocity
        ekf_state.Q.block(3, 3, 3, 3) *= 0.1;

        // // higher process noise for acceleration
        ekf_state.Q.block(6, 6, 3, 3) *= 100.0;
    }

    // Update covariance
    ekf_state.P = (Eigen::MatrixXd::Identity(9, 9) - K * H) * ekf_state.P; // Update covariance

    // Update time
    ekf_state.time_last_updated = time_updated;

    // Update bounding box
    ekf_state.bbox = 0.5 * ekf_state.bbox + (1 - 0.5) * bbox;
}

void CAObstacleTrackerPredictionNode::calculateAverageQandR(Eigen::MatrixXd& Q_avg, Eigen::MatrixXd& R_avg) {
    Q_avg = Eigen::MatrixXd::Zero(9, 9);
    R_avg = Eigen::MatrixXd::Zero(3, 3);

    // TODO: we can implemente weighted average based on the time since the last update (old obstacles have less weight)
    if (!ekf_states_.empty())
    {
        for (const auto &ekf_state : ekf_states_)
        {
            Q_avg += ekf_state.Q;
            R_avg += ekf_state.R;
        }
        Q_avg /= ekf_states_.size();
        R_avg /= ekf_states_.size();
    }
    else
    {
        // If there are no EKF states, initialize Q and R to default values
        Q_avg = Eigen::MatrixXd::Identity(9, 9) * 0.01;
        R_avg = Eigen::MatrixXd::Identity(3, 3) * 0.1;
        
        // Higher process noise for velocity and acceleration since they're not directly measured
        // Q_avg.block(3, 3, 6, 6) *= 100.0; 
    }
}

// =================================================================================
// EKF PREDICTION IMPLEMENTATION
// =================================================================================

// Visualize the predictions over a time horizon, updating both position and velocity
void CAObstacleTrackerPredictionNode::publishPredictions(const std::vector<Measurement> &measurements)
{

    visualization_msgs::msg::MarkerArray markers;
    int id = 0;
    int num_steps = static_cast<int>(prediction_horizon_ / prediction_dt_); // Number of steps

    // Initialize knots and positions
    std::vector<double> t_values;
    std::vector<double> x_values;
    std::vector<double> y_values;
    std::vector<double> z_values;

    for (int i = 0; i < static_cast<int>(measurements.size()); ++i)
    {

        // Initialize position, velocity, and acceleration from EKF state
        Eigen::Vector3d current_position;
        Eigen::Vector3d initial_position;
        current_position = Eigen::Vector3d(measurements[i].assigned_ekf_state.x[0], measurements[i].assigned_ekf_state.x[1], /*z=*/0.0);
        initial_position = current_position;

        Eigen::Vector3d current_velocity;
        Eigen::Vector3d initial_velocity;
        current_velocity = Eigen::Vector3d(measurements[i].assigned_ekf_state.x[3], measurements[i].assigned_ekf_state.x[4], measurements[i].assigned_ekf_state.x[5]);
        initial_velocity = current_velocity;

        Eigen::Vector3d acceleration;
        acceleration = Eigen::Vector3d(measurements[i].assigned_ekf_state.x[6], measurements[i].assigned_ekf_state.x[7], measurements[i].assigned_ekf_state.x[8]);

        // RCLCPP_INFO(this->get_logger(), "Initial State: pos=[%.2f, %.2f, %.2f], vel=[%.2f, %.2f, %.2f], acc=[%.2f, %.2f, %.2f]", 
        //             current_position[0], current_position[1], current_position[2],
        //             current_velocity[0], current_velocity[1], current_velocity[2],
        //             acceleration[0], acceleration[1], acceleration[2]);

        // // Avoid high acceleration values
        // for (int j = 0; j < 3; ++j)
        // {
        //     if (acceleration[j] > 2.0)
        //         acceleration[j] = 2.0;
        //     if (acceleration[j] < -2.0)
        //         acceleration[j] = -2.0;
        // }

        // Store the initial position and time
        t_values.push_back(0.0);
        x_values.push_back(current_position[0]);
        y_values.push_back(current_position[1]);
        z_values.push_back(current_position[2]);

        // Ignore low velocitis (noise) 
        if (initial_velocity.norm() <= velocity_threshold_)
        {
            continue; 
        }

        for (int step = 0; step < num_steps; ++step)
        {

            for (int j = 0; j < 3; ++j)
            {
                // Avoid high velocity values
                if (current_velocity[j] > 5.0)
                    current_velocity[j] = 5.0;
                if (current_velocity[j] < -5.0)
                    current_velocity[j] = -5.0;
            
                // Avoid high acceleration values
                if (acceleration[j] > 8.0)
                    acceleration[j] = 8.0;
                if (acceleration[j] < -8.0)
                    acceleration[j] = -8.0;
            }

            // Calculate time for this step
            double t = step * prediction_dt_;

            // Update velocity: v(t) = v_0 + a * t
            Eigen::Vector3d future_velocity = initial_velocity + acceleration * t;

            // Predict future position: p(t) = p_0 + v_0 * t + 0.5 * a * t^2
            Eigen::Vector3d future_position;
            future_position[0] = initial_position[0] + initial_velocity[0] * t + 0.5 * acceleration[0] * t * t;
            future_position[1] = initial_position[1] + initial_velocity[1] * t + 0.5 * acceleration[1] * t * t;
            future_position[2] = initial_position[2] + initial_velocity[2] * t + 0.5 * acceleration[2] * t * t;
            
            // Store the time and position values
            t_values.push_back(t);
            x_values.push_back(future_position[0]);
            y_values.push_back(future_position[1]);
            z_values.push_back(future_position[2]);

            // Create a marker to visualize the predicted position at this time step
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = frame_id_;
            marker.id = id++;
            marker.type = visualization_msgs::msg::Marker::ARROW;
            marker.action = visualization_msgs::msg::Marker::ADD;

            // Set arrow start (current position) and end (future position)
            geometry_msgs::msg::Point start, end;
            start.x = current_position[0];
            start.y = current_position[1];
            start.z = current_position[2];
            end.x = future_position[0];
            end.y = future_position[1];
            end.z = future_position[2];

            // Set arrow start and end points
            marker.points.push_back(start);
            marker.points.push_back(end);

            // Set scale (arrow width and length)
            marker.scale.x = 0.1;
            marker.scale.y = 0.2;

            // Set color (you can gradually fade it based on time step)
            marker.color.r = measurements[i].assigned_ekf_state.color.r;
            marker.color.g = measurements[i].assigned_ekf_state.color.g;
            marker.color.b = measurements[i].assigned_ekf_state.color.b;
            marker.color.a = measurements[i].assigned_ekf_state.color.a;

            // Add this marker to the marker array
            markers.markers.push_back(marker);

            // Update the current position and velocity for the next step
            current_position = future_position;
            current_velocity = future_velocity;
        }

        // Check if x_values, y_values, z_values changed much (especially in the beginning, the predicted trajectories can be very short and hard to fit)
        // If they are not changing much, we can skip the polynomial fitting
        double x_diff = abs(x_values.front() - x_values.back());
        double y_diff = abs(y_values.front() - y_values.back());
        double z_diff = abs(z_values.front() - z_values.back());

        double cutoff_length_threshold = 0.1; // TODO: make this a parameter?

        // If the predicted trajectory is too short, skip the polynomial fitting
        if (x_diff < cutoff_length_threshold && y_diff < cutoff_length_threshold && z_diff < cutoff_length_threshold)
        {
            t_values.clear();
            x_values.clear();
            y_values.clear();
            z_values.clear();
            continue;
        }

        // Fit a polynomial to the predicted positions
        Eigen::VectorXd beta_x = polyfit(t_values, x_values, degree_for_pwp_);
        Eigen::VectorXd beta_y = polyfit(t_values, y_values, degree_for_pwp_);
        Eigen::VectorXd beta_z = polyfit(t_values, z_values, degree_for_pwp_);

        // Calculate variance of the residuals
        double variance_x = calculateVariance(t_values, x_values, beta_x, degree_for_pwp_);
        double variance_y = calculateVariance(t_values, y_values, beta_y, degree_for_pwp_);
        double variance_z = calculateVariance(t_values, z_values, beta_z, degree_for_pwp_);

        // Convert t_values, beta_x, beta_y, beta_z to PieceWisePol
        PieceWisePol pwp;
        double current_time = this->now().seconds();
        pwp.times.push_back(current_time);
        pwp.times.push_back(current_time + prediction_horizon_);
        pwp.coeff_x.push_back({beta_x(0), beta_x(1), beta_x(2), beta_x(3)});
        pwp.coeff_y.push_back({beta_y(0), beta_y(1), beta_y(2), beta_y(3)});
        pwp.coeff_z.push_back({beta_z(0), beta_z(1), beta_z(2), beta_z(3)});

        // Fit a quintic polynomial to the predicted positions
        Eigen::VectorXd beta_x_quintic = polyfit(t_values, x_values, degree_for_poly_);
        Eigen::VectorXd beta_y_quintic = polyfit(t_values, y_values, degree_for_poly_);
        Eigen::VectorXd beta_z_quintic = polyfit(t_values, z_values, degree_for_poly_);

        // Publish DynTraj message with the predicted trajectory
        dynus_interfaces::msg::DynTraj msg;
        msg.header.stamp = this->now();
        msg.header.frame_id = frame_id_;
        msg.id = measurements[i].assigned_ekf_state.id;
        msg.bbox.push_back(measurements[i].assigned_ekf_state.bbox.x());
        msg.bbox.push_back(measurements[i].assigned_ekf_state.bbox.y());
        msg.bbox.push_back(measurements[i].assigned_ekf_state.bbox.z());
        msg.pwp = mighty_utils::convertPwp2PwpMsg(pwp);
        msg.mode = "quintic";
        msg.ekf_cov_p.push_back(measurements[i].assigned_ekf_state.P(0, 0));
        msg.ekf_cov_p.push_back(measurements[i].assigned_ekf_state.P(1, 1));
        msg.ekf_cov_p.push_back(measurements[i].assigned_ekf_state.P(2, 2));
        msg.ekf_cov_q.push_back(measurements[i].assigned_ekf_state.Q(0, 0));
        msg.ekf_cov_q.push_back(measurements[i].assigned_ekf_state.Q(1, 1));
        msg.ekf_cov_q.push_back(measurements[i].assigned_ekf_state.Q(2, 2));
        msg.ekf_cov_r.push_back(measurements[i].assigned_ekf_state.R(0, 0));
        msg.ekf_cov_r.push_back(measurements[i].assigned_ekf_state.R(1, 1));
        msg.ekf_cov_r.push_back(measurements[i].assigned_ekf_state.R(2, 2));
        msg.poly_cov.push_back(variance_x);
        msg.poly_cov.push_back(variance_y);
        msg.poly_cov.push_back(variance_z);

        // coefficients for quintic polynomial
        msg.poly_coeffs_x.clear();
        msg.poly_coeffs_y.clear();
        msg.poly_coeffs_z.clear();
        for (int j = 0; j < degree_for_poly_ + 1; ++j)
        {
            msg.poly_coeffs_x.push_back(beta_x_quintic(j));
            msg.poly_coeffs_y.push_back(beta_y_quintic(j));
            msg.poly_coeffs_z.push_back(beta_z_quintic(j));
        }

        // Set the start and end times for the trajectory
        msg.poly_start_time = current_time;
        msg.poly_end_time = current_time + prediction_horizon_;

        msg.is_agent = false;
        pub_predicted_traj_->publish(msg);

        // Clear the vectors for the next EKF state
        t_values.clear();
        x_values.clear();
        y_values.clear();
        z_values.clear();

        if (initial_velocity.norm() > velocity_threshold_)
        {
            // Publish predicted position 
            geometry_msgs::msg::PoseStamped pred_pos_msg;
            pred_pos_msg.header.stamp = this->now();
            // pred_pos_msg.header.stamp = pc_timestamp_;
            pred_pos_msg.header.frame_id = frame_id_; 
            pred_pos_msg.pose.position.x = initial_position[0];
            pred_pos_msg.pose.position.y = initial_position[1];
            pred_pos_msg.pose.position.z = initial_position[2];
            pred_pos_pub_->publish(pred_pos_msg);

            geometry_msgs::msg::TwistStamped pred_vel_msg;
            pred_vel_msg.header.stamp = this->now();
            // pred_vel_msg.header.stamp = pc_timestamp_;
            pred_vel_msg.header.frame_id = frame_id_; 
            pred_vel_msg.twist.linear.x = initial_velocity[0];
            pred_vel_msg.twist.linear.y = initial_velocity[1];
            pred_vel_msg.twist.linear.z = initial_velocity[2];
            pred_vel_pub_->publish(pred_vel_msg);
        }

    }

    // Publish the marker array with predicted trajectories
    if (visual_level_ >= 0)
        pub_markers_->publish(markers);
}


Eigen::VectorXd CAObstacleTrackerPredictionNode::polyfit(const std::vector<double> &t, const std::vector<double> &y, int degree)
{

    // Number of data points
    int n = t.size();

    // Construct the Vandermonde matrix for polynomial fitting
    Eigen::MatrixXd X(n, degree + 1);
    Eigen::VectorXd Y(n);

    for (int i = 0; i < n; ++i)
    {
        Y(i) = y[i];
        for (int j = 0; j <= degree; ++j)
        {
            X(i, j) = std::pow(t[i], degree - j); // Note: the coefficients are stored as [a b c d] for a*t^3 + b*t^2 + c*t + d so we need to flip it instead of std::pow(t[i], j)
        }
    }

    // Solve the normal equations: X^T * X * beta = X^T * Y
    Eigen::VectorXd beta = (X.transpose() * X).ldlt().solve(X.transpose() * Y);

    return beta;
}

double CAObstacleTrackerPredictionNode::calculateVariance(const std::vector<double> &t, const std::vector<double> &y, const Eigen::VectorXd &beta, int degree)
{
    // Calculate residuals and estimate variance
    int n = t.size();
    double residual_sum = 0.0;

    for (int i = 0; i < n; ++i)
    {
        double fitted_value = 0.0;
        for (int j = 0; j <= degree; ++j)
        {
            fitted_value += beta(j) * std::pow(t[i], j);
        }
        double residual = y[i] - fitted_value;
        residual_sum += residual * residual;
    }

    return residual_sum / (n - degree - 1); // variance = sum(residuals^2) / (n - p - 1)
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<CAObstacleTrackerPredictionNode>(); 
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}