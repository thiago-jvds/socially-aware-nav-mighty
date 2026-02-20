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

#include "mighty/trackers/IMM_obstacle_tracker_prediction_node.hpp"
#include <cmath>

using std::placeholders::_1;

IMMObstacleTrackerPredictionNode::IMMObstacleTrackerPredictionNode() 
: Node("IMM_obstacle_tracker_prediction_node") 
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

    // IMM Tuning Parameters
    this->declare_parameter("prob_transition_stay", 0.95);   // High probability to stay in current mode
    this->declare_parameter("prob_transition_switch", 0.05); // Low probability to switch

    this->declare_parameter("use_2d_predictions", true);

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
    prob_transition_stay_ = this->get_parameter("prob_transition_stay").as_double();
    prob_transition_switch_ = this->get_parameter("prob_transition_switch").as_double();
    use_2d_predictions_ = this->get_parameter("use_2d_predictions").as_bool();


    // 2. Pub/Sub
    sub_detections_ = this->create_subscription<vision_msgs::msg::Detection3DArray>(
        "detected_objects", 10, 
        std::bind(&IMMObstacleTrackerPredictionNode::detectionsCallback, this, _1));

    pub_markers_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("tracked_obstacles", 10);
    pub_predicted_traj_ = this->create_publisher<dynus_interfaces::msg::DynTraj>("predicted_trajs", 10);

    pred_pos_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("predicted_positions", 10);
    pred_vel_pub_ = this->create_publisher<geometry_msgs::msg::TwistStamped>("predicted_velocities", 10);

    initialize_imm_matrices();

    RCLCPP_INFO(this->get_logger(), " IMM Tracker & Prediction Initialized.");
}

void IMMObstacleTrackerPredictionNode::initialize_imm_matrices()
{
    // --- A. Transition Probability Matrix (TPM) ---
    // Rows sum to 1. Entry (i, j) is Prob(Mode j | Mode i)
    
    trans_prob_mat_ = Eigen::MatrixXd::Zero(NUM_MODES, NUM_MODES);

    double p_stay = prob_transition_stay_; 
    double p_switch = (1.0 - p_stay) / (NUM_MODES - 1); 

    for (int i = 0; i < NUM_MODES; ++i) {
        for (int j = 0; j < NUM_MODES; ++j) {
            trans_prob_mat_(i, j) = (i == j) ? p_stay : p_switch;
        }
    }
}

void IMMObstacleTrackerPredictionNode::detectionsCallback(const vision_msgs::msg::Detection3DArray::SharedPtr msg)
{
    // --- Step 1: Parse Inputs ---
    std::vector<Measurement> current_measurements;
    current_measurements.reserve(msg->detections.size());

    for (const auto& det : msg->detections) {
        Measurement m;
        m.centroid = Eigen::Vector3d(det.bbox.center.position.x, det.bbox.center.position.y, det.bbox.center.position.z);
        m.has_match = false;
        
        // [OPTIMIZATION] Store the RAW bbox directly in the IMMTrack container
        // effectively using it as temporary storage until we calculate the filter.
        m.assigned_track.bbox = Eigen::Vector3d(det.bbox.size.x, det.bbox.size.y, det.bbox.size.z);
        
        current_measurements.push_back(m);
    }

    // --- Step 2: Lifecycle ---
    deleteOldTracks();
    // Reserve memory to prevent invalidation during push_back
    tracks_.reserve(tracks_.size() + current_measurements.size());

    std::vector<Eigen::VectorXd> x_mixed(NUM_MODES);
    std::vector<Eigen::MatrixXd> P_mixed(NUM_MODES);
    
    // --- Step 3: Predict for current tracks --
    for (auto& track : tracks_) {
        interaction(track, x_mixed, P_mixed);
        predict(track, x_mixed, P_mixed, adaptive_kf_dt_); 
        combine(track);
    }

    // --- Step 4: Association & Update ---
    std::vector<int> assignments = associateMeasurements(current_measurements, tracks_, cluster_tolerance_);
    
    for (int i = 0; i < static_cast<int>(assignments.size()); i++){
        int match_idx = assignments[i];
        auto& meas = current_measurements[i];
        Eigen::Vector3d raw_bbox = meas.assigned_track.bbox;
        Eigen::Vector3d& centroid = meas.centroid;


        if (match_idx >= 0) {
            IMMTrack& matched_track = tracks_[match_idx];
            
            update(matched_track, centroid, 
                adaptive_kf_alpha_, this->now().seconds(), raw_bbox);
            combine(matched_track);

            // Overwrite the placeholder state with the REAL filtered state
            meas.assigned_track = matched_track;
            meas.has_match = true;
            
        } else {
            
            IMMTrack new_track(
                state_dim_,meas_dim_, this->now().seconds(), 
                centroid, raw_bbox, track_id_++,
                sigma_a_CA_,
                sigma_yaw_CA_,
                sigma_a_CTRV_,
                sigma_yaw_CTRV_,
                th_threshold_);

            tracks_.push_back(new_track);            
            
            // Store the new track in the measurement
            meas.assigned_track = new_track;
            meas.has_match = true;
        }
    }

    // --- Step 5: Publish ---
    publishPredictions(current_measurements);
}


// =================================================================================
// ASSOCIATION & DELETION 
// =================================================================================

/*   
 * @brief O(max(current_meas.size(), tracks_.size())^3)
 * (for max_N ~ 10 detections, 1000 ops, negligible)
 */
std::vector<int> IMMObstacleTrackerPredictionNode::associateMeasurements(
    const std::vector<Measurement>& new_detections,
    const std::vector<IMMTrack>& tracks,
    double tolerance
){

    int M = new_detections.size();
    int N = tracks.size();

    const double INF = std::numeric_limits<double>::max();

    // No matches found if nothing detected, or all new traks
    if (M == 0 || N == 0) return std::vector<int>(M, -1);

    // Build cost

    // added one extra dim for convenience
    int dim = std::max(M, N);
    std::vector<std::vector<double>> cost_matrix(dim + 1, std::vector<double>(dim + 1, 0.0));

    for (int m = 0; m < M; m++){
        for (int n = 0; n < N; n++){
            cost_matrix[m + 1][n + 1] = (tracks[n].x.head(3) - new_detections[m].centroid).norm();
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

void IMMObstacleTrackerPredictionNode::deleteOldTracks() {
    double current_time = this->now().seconds();
    
    auto it = tracks_.begin();
    while (it != tracks_.end()) {
        if ((current_time - it->time_last_updated) > time_to_delete_old_obstacles_) {
            it = tracks_.erase(it);
        } else {
            ++it;
        }
    }
}


// =================================================================================
// IMM MATH IMPLEMENTATION
// =================================================================================

void IMMObstacleTrackerPredictionNode::interaction(
    IMMTrack& track, 
    std::vector<Eigen::VectorXd>& x_mixed, 
    std::vector<Eigen::MatrixXd>& P_mixed)
{
    // Ensure outputs are sized correctly
    if (x_mixed.size() != NUM_MODES) x_mixed.resize(NUM_MODES);
    if (P_mixed.size() != NUM_MODES) P_mixed.resize(NUM_MODES);
    
    // --- A. Compute Normalization Constants (c_bar) ---
    // c_bar[j] = Sum_i ( TransitionProb(i->j) * ModeProb(i) )
    track.c_bar.setZero();
    
    for (int j = 0; j < NUM_MODES; ++j) {
        for (int i = 0; i < NUM_MODES; ++i) {
            track.c_bar(j) += trans_prob_mat_(i, j) * track.mode_probs(i);
        }
    }
    
    // --- B. Compute Mixing Probabilities & Mixed Estimates ---
    for (int j = 0; j < NUM_MODES; ++j) {
        
        x_mixed[j] = Eigen::VectorXd::Zero(state_dim_);
        P_mixed[j] = Eigen::MatrixXd::Zero(state_dim_, state_dim_);
        
        double c_j = track.c_bar(j);
        if (c_j < 1e-6) c_j = 1e-6;
        
        // 2. Calculate Weighted Average for track (Mean)
        // x_0j = Sum_i ( x_i * w_ij )
        for (int i = 0; i < NUM_MODES; ++i) {
            // Mixing Weight: Prob(Came from i | Currently in j)
            // w_ij = (T_ij * mu_i) / c_bar[j]
            double w_ij = (trans_prob_mat_(i, j) * track.mode_probs(i)) / c_j;
            
            x_mixed[j] += w_ij * track.models[i]->x;
        }
        
        // 3. Calculate Weighted Covariance + Spread
        // P_0j = Sum_i ( w_ij * (P_i + (x_i - x_0j)(x_i - x_0j)^T) )
        for (int i = 0; i < NUM_MODES; ++i) {
            double w_ij = (trans_prob_mat_(i, j) * track.mode_probs(i)) / c_j;
            
            // Calculate difference (spread)
            Eigen::VectorXd diff = track.models[i]->x - x_mixed[j];
            
            diff(4) = normalize_angle(diff(4));    // theta
            diff(5) = normalize_angle(diff(5));    // phi
            
            P_mixed[j] += w_ij * (track.models[i]->P + diff * diff.transpose());
        }
    }
}

void IMMObstacleTrackerPredictionNode::predict(
    IMMTrack& track, 
    const std::vector<Eigen::VectorXd>& x_mixed, 
    const std::vector<Eigen::MatrixXd>& P_mixed, 
    double dt)
{
    for (int m = 0; m < NUM_MODES; ++m) {
        track.models[m]->x = x_mixed[m];
        track.models[m]->P = P_mixed[m];

        track.models[m]->predict(dt);
    }
}
    
void IMMObstacleTrackerPredictionNode::update(IMMTrack &track, 
    const Eigen::VectorXd &z, 
    double alpha, double time_updated, 
    const Eigen::Vector3d &bbox)
{
    
    double normalization_sum = 0.0;
    
    for (int m = 0; m < NUM_MODES; ++m) {
        // 1. Standard Kalman Update (Math handled inside the model)
        track.models[m]->update(z);
        
        // 2. Retrieve the likelihood calculated by the model
        double likelihood = track.models[m]->likelihood;
        track.likelihoods(m) = likelihood;
        
        // 3. Mode Probability Update
        double unnormalized_prob = likelihood * track.c_bar(m);
        track.mode_probs(m) = unnormalized_prob;
        normalization_sum += unnormalized_prob;
    }
    
    // Normalize probabilities to sum to 1
    if (normalization_sum > 1e-9) {
        track.mode_probs /= normalization_sum;
    } else {
        // Fallback if numerical issues occurred: keep previous probs or reset
        RCLCPP_WARN(this->get_logger(), "IMM Probability collapse. Resetting to uniform.");
        track.mode_probs.fill(1.0 / NUM_MODES);
    }
    
    // Update time
    track.time_last_updated = time_updated;
    
    // Update bounding box
    track.bbox = 0.5 * track.bbox + (1 - 0.5) * bbox;

    RCLCPP_INFO(this->get_logger(), 
        "Track [%d] Probabilities -> FWD: %.2f, LEFT: %.2f, RIGHT: %.2f",
        track.id, track.mode_probs(0), track.mode_probs(1), 
    track.mode_probs(2));
    
}
        
void IMMObstacleTrackerPredictionNode::combine(IMMTrack& track)
{
    // 1. Initialize Combined track
    track.x.setZero();
    
    // --- Combine track Vector (Weighted Average) ---
    // X_combined = Sum( mu_i * x_i )
    for (int m = 0; m < NUM_MODES; ++m) {
        track.x += track.mode_probs(m) * track.models[m]->x;
    }
    
    // 2. Initialize Combined Covariance
    track.P.setZero();
    
    // --- Combine Covariance ---
    // P_combined = Sum( mu_i * (P_i + (x_i - X_combined)*(x_i - X_combined)^T) )
    for (int m = 0; m < NUM_MODES; ++m) {
        Eigen::VectorXd diff = track.models[m]->x - track.x;
        
        diff(4) = normalize_angle(diff(4));      // theta
        diff(5)   = normalize_angle(diff(5));    // phi
        
        // P += prob * ( P_mode + spread_term )
        track.P += track.mode_probs(m) * (track.models[m]->P + diff * diff.transpose());
    }
}
            
std::vector<std::pair<double, Eigen::Vector3d>> IMMObstacleTrackerPredictionNode::generatePrediction(
    const IMMTrack& track, const int best_mode)
{
    std::vector<std::pair<double, Eigen::Vector3d>> trajectory;    

    // --- 1. Set Horizon based on Mode ---
    int num_steps = std::ceil(prediction_horizon_ / prediction_dt_);

    // --- 2. Initialize Simulation State ---
    double x    = track.models[best_mode]->x(0);
    double y    = track.models[best_mode]->x(1);
    double z    = track.models[best_mode]->x(2);
    double v    = track.models[best_mode]->x(3);
    double th   = track.models[best_mode]->x(4);
    double ph   = track.models[best_mode]->x(5);
    double a    = track.models[best_mode]->x(6);
    double th_d = track.models[best_mode]->x(7);
    double ph_d = track.models[best_mode]->x(8);

    if  (v > 5.0) v = 5.0;
    if (v < -5.0) v = -5.0;

    if  (a > 8.0) a = 8.0;
    if (a < -8.0) a = -8.0;

    RCLCPP_INFO(this->get_logger(), " [Track %d] v: %.3f, a: %.3f, th: %.3f", track.id, v, a);

    // --- 3. Step-by-Step Integration ---
    for (int step = 0; step < num_steps; ++step) {
        double t = step * prediction_dt_;
        trajectory.push_back({t, Eigen::Vector3d(x, y, z)});

        double dist = v * prediction_dt_ + 0.5 * a * std::pow(prediction_dt_, 2);

        x += dist * cos(ph) * cos(th);
        y += dist * cos(ph) * sin(th);
        z += dist * sin(ph);

        th += th_d * prediction_dt_;
        ph += ph_d * prediction_dt_;

        v += a * prediction_dt_;
    }

    return trajectory;
}

// =================================================================================
// EKF PREDICTION IMPLEMENTATION
// =================================================================================

void IMMObstacleTrackerPredictionNode::publishPredictions(const std::vector<Measurement> &measurements)
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
        const IMMTrack& track = measurements[i].assigned_track;

        int best_mode = 0;
        track.mode_probs.maxCoeff(&best_mode);

        if (tracker_debug_) {
            std::string mode_names[] = {"FWD", "LEFT", "RIGHT"};

            // --- 2. Determine Color Based on Mode ---
            std_msgs::msg::ColorRGBA mode_color;
            mode_color.a = 1.0;
            if (best_mode == MODE_FWD)          { mode_color.r = 0.0; mode_color.g = 1.0; mode_color.b = 0.0; } // Green
            else if (best_mode == MODE_LEFT)    { mode_color.r = 1.0; mode_color.g = 0.0; mode_color.b = 0.0; } // Red
            else if (best_mode == MODE_RIGHT)   { mode_color.r = 0.0; mode_color.g = 0.0; mode_color.b = 1.0; } // Blue

            // --- 3. Add Status Text Marker ---
            visualization_msgs::msg::Marker text_marker;
            text_marker.header.frame_id = frame_id_;
            text_marker.header.stamp = this->now();
            text_marker.id = id++;
            text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
            text_marker.action = visualization_msgs::msg::Marker::ADD;
            
            // Position text 1.5m above the object
            text_marker.pose.position.x = track.x(0);
            text_marker.pose.position.y = track.x(1);
            text_marker.pose.position.z = track.x(2) + 1.5;
            text_marker.scale.z = 0.4; // Text height
            text_marker.color.r = 1.0; text_marker.color.g = 1.0; text_marker.color.b = 1.0; text_marker.color.a = 1.0;

            // Display ID, Best Mode, and Probability
            std::stringstream ss;
            ss << "ID: " << track.id << "\n" 
            << mode_names[best_mode] << " (" << std::fixed << std::setprecision(2) << track.mode_probs(best_mode) << ")";
            text_marker.text = ss.str();
            markers.markers.push_back(text_marker);
        }

        if (fabs(track.x(3)) <= velocity_threshold_) continue;

        std::vector<std::pair<double, Eigen::Vector3d>> predicted_trajectory = generatePrediction(track, best_mode);

        for (int k = 0; k < predicted_trajectory.size() - 1; ++k)
        {
            // Calculate time for this step
            double t = predicted_trajectory[k].first;
            Eigen::Vector3d p_curr = predicted_trajectory[k].second;

            t_values.push_back(t);
            x_values.push_back(p_curr(0));
            y_values.push_back(p_curr(1));
            z_values.push_back(p_curr(2));

            if (k == predicted_trajectory.size() - 1) continue;

            Eigen::Vector3d p_next = predicted_trajectory[k+1].second;
            
            // Create a marker to visualize the predicted position at this time step
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = frame_id_;
            marker.id = id++;
            marker.type = visualization_msgs::msg::Marker::ARROW;
            marker.action = visualization_msgs::msg::Marker::ADD;

            // Set arrow start (current position) and end (future position)
            geometry_msgs::msg::Point start, end;
            start.x = p_curr(0);
            start.y = p_curr(1);
            start.z = p_curr(2);
            end.x = p_next(0);
            end.y = p_next(1);
            end.z = p_next(2);

            // Set arrow start and end points
            marker.points.push_back(start);
            marker.points.push_back(end);

            // Set scale (arrow width and length)
            marker.scale.x = 0.1;
            marker.scale.y = 0.2;

            // Set color (you can gradually fade it based on time step)
            marker.color.r = track.color.r;
            marker.color.g = track.color.g;
            marker.color.b = track.color.b;
            marker.color.a = track.color.a;

            // lifetime
            // marker.lifetime = rclcpp::Duration::from_seconds(0.1);

            // Add this marker to the marker array
            markers.markers.push_back(marker);
        }

        // Check if x_values, y_values, z_values changed much (especially in the beginning, the predicted trajectories can be very short and hard to fit)
        // If they are not changing much, we can skip the polynomial fitting
        double x_diff = abs(x_values.front() - x_values.back());
        double y_diff = abs(y_values.front() - y_values.back());
        double z_diff = abs(z_values.front() - z_values.back());

        double cutoff_length_threshold = 0.00; // TODO: make this a parameter?

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
        msg.id = track.id;
        msg.bbox.push_back(track.bbox.x());
        msg.bbox.push_back(track.bbox.y());
        msg.bbox.push_back(track.bbox.z());
        msg.pwp = mighty_utils::convertPwp2PwpMsg(pwp);
        msg.mode = "quintic";
        msg.ekf_cov_p.push_back(track.P(0, 0));
        msg.ekf_cov_p.push_back(track.P(1, 1));
        msg.ekf_cov_p.push_back(track.P(2, 2));

        msg.ekf_cov_q.push_back(track.models[best_mode]->Q(0, 0));
        msg.ekf_cov_q.push_back(track.models[best_mode]->Q(1, 1));
        msg.ekf_cov_q.push_back(track.models[best_mode]->Q(2, 2));

        msg.ekf_cov_r.push_back(track.R(0, 0));
        msg.ekf_cov_r.push_back(track.R(1, 1));
        msg.ekf_cov_r.push_back(track.R(2, 2));
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

        // if (initial_velocity.norm() > velocity_threshold_)
        // {
        //     // Publish predicted position 
        //     geometry_msgs::msg::PoseStamped pred_pos_msg;
        //     pred_pos_msg.header.stamp = this->now();
        //     // pred_pos_msg.header.stamp = pc_timestamp_;
        //     pred_pos_msg.header.frame_id = frame_id_; 
        //     pred_pos_msg.pose.position.x = initial_position[0];
        //     pred_pos_msg.pose.position.y = initial_position[1];
        //     pred_pos_msg.pose.position.z = initial_position[2];
        //     pred_pos_pub_->publish(pred_pos_msg);

        //     geometry_msgs::msg::TwistStamped pred_vel_msg;
        //     pred_vel_msg.header.stamp = this->now();
        //     // pred_vel_msg.header.stamp = pc_timestamp_;
        //     pred_vel_msg.header.frame_id = frame_id_; 
        //     pred_vel_msg.twist.linear.x = initial_velocity[0];
        //     pred_vel_msg.twist.linear.y = initial_velocity[1];
        //     pred_vel_msg.twist.linear.z = initial_velocity[2];
        //     pred_vel_pub_->publish(pred_vel_msg);
        // }

    }

    // Publish the marker array with predicted trajectories
    if (visual_level_ >= 0)
        pub_markers_->publish(markers);
}

Eigen::VectorXd IMMObstacleTrackerPredictionNode::polyfit(const std::vector<double> &t, const std::vector<double> &y, int degree)
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

double IMMObstacleTrackerPredictionNode::calculateVariance(const std::vector<double> &t, const std::vector<double> &y, const Eigen::VectorXd &beta, int degree)
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
  auto node = std::make_shared<IMMObstacleTrackerPredictionNode>(); 
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}