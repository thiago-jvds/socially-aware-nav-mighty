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
    this->declare_parameter("tracker_debug", false);
    
    // Yield mode parameters
    this->declare_parameter("d_08", 3.0);   // distance at which to reduce 80% of speed          
    this->declare_parameter("max_considered_distance", 10.0); // beyond this distance, no reduction is applied

    this->declare_parameter("d_immediate_max", 5.0);    // distance threshold to enter yield mode
    this->declare_parameter("d_personal_max", 10.0);    // distance threshold for personal space

    // IMM Tuning Parameters
    this->declare_parameter("prob_transition_stay", 0.85);   // High probability to stay in current mode

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
    tracker_debug_ = this->get_parameter("tracker_debug").as_bool();

    double d_08 = this->get_parameter("d_08").as_double();
    max_considered_distance_ = this->get_parameter("max_considered_distance").as_double();

    double d_08_sq = d_08 * d_08; 
    d_08_4_ = d_08_sq * d_08_sq; // precompute for efficiency

    d_immediate_max_ = this->get_parameter("d_immediate_max").as_double();
    d_personal_max_ = this->get_parameter("d_personal_max").as_double();


    // 2. Pub/Sub
    sub_detections_ = this->create_subscription<vision_msgs::msg::Detection3DArray>(
        "detected_objects", 10, 
        std::bind(&IMMObstacleTrackerPredictionNode::detectionsCallback, this, _1));
    
    sub_trajectory_ = this->create_subscription<dynus_interfaces::msg::Trajectory>(
        "trajectory", 10,
        std::bind(&IMMObstacleTrackerPredictionNode::trajectoryCallback, this, std::placeholders::_1));

    sub_state_ = this->create_subscription<dynus_interfaces::msg::State>(
        "state", 10,
        std::bind(&IMMObstacleTrackerPredictionNode::stateCallback, this, std::placeholders::_1));

    pub_markers_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("tracked_obstacles", 10);
    pub_predicted_traj_ = this->create_publisher<dynus_interfaces::msg::DynTraj>("predicted_trajs", 10);

    pred_pos_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("predicted_positions", 10);
    pred_vel_pub_ = this->create_publisher<geometry_msgs::msg::TwistStamped>("predicted_velocities", 10);
    pub_yield_mode_ = this->create_publisher<dynus_interfaces::msg::YieldMode>("yield_mode", 10);

    pub_unc_sphere_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("uncertainty_spheres", 10);


    RCLCPP_INFO(this->get_logger(), " IMM Tracker & Prediction Initialized.");
}

void IMMObstacleTrackerPredictionNode::trajectoryCallback(const dynus_interfaces::msg::Trajectory::SharedPtr msg)
{
    trajectory_ = *msg;
    trajectory_initialized_ = true;
}

void IMMObstacleTrackerPredictionNode::stateCallback(const dynus_interfaces::msg::State::SharedPtr msg)
{
    current_state_ = *msg;
    state_initialized_ = true;
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

            // cold start
            if (matched_track.is_first_meas) { 
                matched_track.is_first_meas = false;

                if (!linear_kf_) 
                {

                    double dx = centroid(0) - matched_track.models[0]->x(0); 
                    double dy = centroid(1) - matched_track.models[0]->x(1);

                    double initial_heading = std::atan2(dy, dx);
                    
                    double dt = this->now().seconds() - matched_track.time_last_updated;
                    double initial_velocity = std::sqrt(dx*dx + dy*dy) / dt;

                    for (int m = 0; m < NUM_MODES; ++m) {
                        matched_track.models[m]->x(3) = initial_velocity;
                        matched_track.models[m]->x(4) = initial_heading;
                    }
                } else 
                {
                    double dx = centroid(0) - matched_track.models[0]->x(0); 
                    double dy = centroid(1) - matched_track.models[0]->x(1);
                    double dz = centroid(2) - matched_track.models[0]->x(2);
                    
                    // Time delta between first and second measurement
                    double dt = this->now().seconds() - matched_track.time_last_updated;
                    
                    // Pure linear velocity in each axis (no trig needed!)
                    double vx_init = dx / dt;
                    double vy_init = dy / dt;
                    double vz_init = dz / dt;

                    for (int m = 0; m < NUM_MODES; ++m) {
                        // Update the state vector for all IMM models
                        // Reminder: State is [x, y, z, vx, vy, vz, ax, ay, az]
                        matched_track.models[m]->x(3) = vx_init;
                        matched_track.models[m]->x(4) = vy_init;
                        matched_track.models[m]->x(5) = vz_init;
                    }
                }
            }
            
            update(matched_track, centroid, 
                adaptive_kf_alpha_, this->now().seconds(), raw_bbox);
            combine(matched_track);

            // Overwrite the placeholder state with the REAL filtered state
            meas.assigned_track = matched_track;
            meas.has_match = true;
            
        } else {
            RCLCPP_DEBUG(this->get_logger(), "No match for detection at (%.2f, %.2f). Creating new track.", 
                centroid(0), centroid(1));
            IMMTrack new_track(
                state_dim_,meas_dim_, this->now().seconds(), 
                centroid, raw_bbox, track_id_++,
                sigma_a_CA_,
                sigma_yaw_CA_,
                prob_transition_stay_);

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
        if ((current_time - it->time_last_updated) > 1.0) {
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
            track.c_bar(j) += track.trans_prob_mat_(i, j) * track.mode_probs(i);
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
            double w_ij = (track.trans_prob_mat_(i, j) * track.mode_probs(i)) / c_j;
            
            x_mixed[j] += w_ij * track.models[i]->x;
        }
        
        // 3. Calculate Weighted Covariance + Spread
        // P_0j = Sum_i ( w_ij * (P_i + (x_i - x_0j)(x_i - x_0j)^T) )
        for (int i = 0; i < NUM_MODES; ++i) {
            double w_ij = (track.trans_prob_mat_(i, j) * track.mode_probs(i)) / c_j;
            
            // Calculate difference (spread)
            Eigen::VectorXd diff = track.models[i]->x - x_mixed[j];

            if (!linear_kf_)
            {
                diff(4) = normalize_angle(diff(4));    // theta
                diff(5) = normalize_angle(diff(5));    // phi
            } 
            
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

        // RCLCPP_INFO(this->get_logger(), "Track [%d] Mode %d Likelihood: %.6f", track.id, m, likelihood);
        
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

    // Adapt TPM
    track.adaptTPM();

    // RCLCPP_INFO(this->get_logger(), 
    //     "Track [%d] Probabilities -> CV: %.2f, CA: %.2f, STOP: %.2f",
    //     track.id, track.mode_probs(0), track.mode_probs(1), track.mode_probs(2));
    
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
        
        if (!linear_kf_)
        {
            diff(4) = normalize_angle(diff(4));      // theta
            diff(5)  = normalize_angle(diff(5));    // phi
        }
        
        // P += prob * ( P_mode + spread_term )
        track.P += track.mode_probs(m) * (track.models[m]->P + diff * diff.transpose());
    }
}
            
std::vector<std::pair<double, Eigen::Vector4d>> IMMObstacleTrackerPredictionNode::generatePrediction(
    const IMMTrack& track)
{
    std::vector<std::pair<double, Eigen::Vector4d>> trajectory;    
    std::vector<Eigen::Matrix2d> covariances; 

    // --- 1. Set Horizon based on Mode ---
    int num_steps = std::ceil(prediction_horizon_ / prediction_dt_);

    // Store previous and current measurement for each mode
    std::array<double, NUM_MODES> x, y, z, v, th, ph, a, th_d, ph_d;
    std::array<Eigen::Matrix2d, NUM_MODES> P_xy; 
    std::array<Eigen::Matrix2d, NUM_MODES> Q_xy;

    Eigen::VectorXd current_probs = track.mode_probs;

    if (tracker_debug_) {
        RCLCPP_INFO(this->get_logger(), " [Track %d] v: %.3f, a: %.3f, th: %.3f, th_d: %.3f", track.id, v, a, th, th_d);
    }

    // initialize
    for (int m = 0; m < NUM_MODES; m++) {
        x[m]    = track.models[m]->x(0);
        y[m]    = track.models[m]->x(1);
        z[m]    = track.models[m]->x(2);
        th[m]   = track.models[m]->x(4);
        ph[m]   = track.models[m]->x(5);
        th_d[m] = track.models[m]->x(7);
        ph_d[m] = track.models[m]->x(8);

        double v_ = track.models[m]->x(3);
        double a_ = track.models[m]->x(6);
        
        v[m] = std::clamp(v_, -5.0, 5.0);
        a[m] = std::clamp(a_, -3.0, 3.0);

        P_xy[m] = track.models[m]->P.block<2, 2>(0, 0);
        Q_xy[m] = track.models[m]->Q.block<2, 2>(0, 0);
    }

    // --- 2. Step-by-Step Integration ---
    for (int step = 0; step < num_steps; ++step) {
        double t = step * prediction_dt_;
        double x_mixed, y_mixed, z_mixed, cos_mixed, sin_mixed;
        x_mixed = 0.0;
        y_mixed = 0.0;
        z_mixed = 0.0;
        cos_mixed = 0.0;
        sin_mixed = 0.0;

        Eigen::Matrix2d P_mixed = Eigen::Matrix2d::Zero();

        for (int m = 0; m < NUM_MODES; m++) {
            // mix
            x_mixed += current_probs(m) * x[m];
            y_mixed += current_probs(m) * y[m];
            z_mixed += current_probs(m) * z[m];
            cos_mixed += current_probs(m) * cos(th[m]);
            sin_mixed += current_probs(m) * sin(th[m]);
        }

        double th_mixed = std::atan2(sin_mixed, cos_mixed);
        for (int m = 0; m < NUM_MODES; m++) {
            P_xy[m] += Q_xy[m] * prediction_dt_;

            Eigen::Vector2d diff = Eigen::Vector2d(x[m], y[m]) - Eigen::Vector2d(x_mixed, y_mixed);

            P_mixed += current_probs(m) * (P_xy[m] + diff * diff.transpose());

            // integrate
            double dist = v[m] * prediction_dt_ + 0.5 * a[m] * std::pow(prediction_dt_, 2);
            
            x[m] += dist * cos(ph[m]) * cos(th[m]);
            y[m] += dist * cos(ph[m]) * sin(th[m]);
            z[m] += dist * sin(ph[m]);
            
            th[m] += th_d[m] * prediction_dt_;
            ph[m] += ph_d[m] * prediction_dt_;
            
            v[m] += a[m] * prediction_dt_;

            v[m] = std::clamp(v[m], -5.0, 5.0);
            a[m] = std::clamp(a[m], -3.0, 3.0);

        }
        trajectory.push_back({t, Eigen::Vector4d(x_mixed, y_mixed, z_mixed, th_mixed)});
        covariances.push_back(P_mixed);

        current_probs = track.trans_prob_mat_.transpose() * current_probs;
    }

    publishUncertaintyMarkers(track.id, trajectory, covariances);
    return trajectory;
}


std::vector<std::pair<double, Eigen::Vector4d>> IMMObstacleTrackerPredictionNode::generatePredictionLinearKF(
    const IMMTrack& track)
{
    std::vector<std::pair<double, Eigen::Vector4d>> trajectory;    
    std::vector<Eigen::Matrix2d> covariances; 

    // --- 1. Set Horizon based on Mode ---
    int num_steps = std::ceil(prediction_horizon_ / prediction_dt_);

    // Store previous and current measurement for each mode
    std::array<double, NUM_MODES> x, y, z, vx, vy, vz, ax, ay, az;
    std::array<Eigen::Matrix2d, NUM_MODES> P_xy; 
    std::array<Eigen::Matrix2d, NUM_MODES> Q_xy;

    Eigen::VectorXd current_probs = track.mode_probs;

    // initialize
    for (int m = 0; m < NUM_MODES; m++) {
        x[m]    = track.models[m]->x(0);
        y[m]    = track.models[m]->x(1);
        z[m]    = track.models[m]->x(2);
        vx[m]   = track.models[m]->x(3);
        vy[m]   = track.models[m]->x(4);
        vz[m]   = track.models[m]->x(5);
        ax[m]   = track.models[m]->x(6);
        ay[m]   = track.models[m]->x(7);
        az[m]   = track.models[m]->x(8);

        // clamp v and a
        for (int i = 3; i < 6; i++) {
            // v
            track.models[m]->x(i) = std::clamp(track.models[m]->x(i), -5.0, 5.0);
            
            // a
            track.models[m]->x(i + 3) = std::clamp(track.models[m]->x(i + 3), -3.0, 3.0);
        }

        P_xy[m] = track.models[m]->P.block<2, 2>(0, 0);
        Q_xy[m] = track.models[m]->Q.block<2, 2>(0, 0);
    }

    // --- 2. Step-by-Step Integration ---
    for (int step = 0; step < num_steps; ++step) {
        double t = step * prediction_dt_;
        double x_mixed, y_mixed, z_mixed, cos_mixed, sin_mixed;
        x_mixed = 0.0;
        y_mixed = 0.0;
        z_mixed = 0.0;

        Eigen::Matrix2d P_mixed = Eigen::Matrix2d::Zero();

        for (int m = 0; m < NUM_MODES; m++) {
            // mix
            x_mixed += current_probs(m) * x[m];
            y_mixed += current_probs(m) * y[m];
            z_mixed += current_probs(m) * z[m];
        }

        for (int m = 0; m < NUM_MODES; m++) {
            P_xy[m] += Q_xy[m] * prediction_dt_;

            Eigen::Vector2d diff = Eigen::Vector2d(x[m], y[m]) - Eigen::Vector2d(x_mixed, y_mixed);

            P_mixed += current_probs(m) * (P_xy[m] + diff * diff.transpose());

            // integrate            
            x[m] = x[m] + vx[m] * prediction_dt_ + 0.5 * ax[m] * std::pow(prediction_dt_, 2);
            y[m] = y[m] + vy[m] * prediction_dt_ + 0.5 * ay[m] * std::pow(prediction_dt_, 2);
            z[m] = z[m] + vz[m] * prediction_dt_ + 0.5 * az[m] * std::pow(prediction_dt_, 2);
            
            vx[m] = vx[m] + ax[m] * prediction_dt_;
            vy[m] = vy[m] + ay[m] * prediction_dt_;
            vz[m] = vz[m] + az[m] * prediction_dt_;

            vx[m] = std::clamp(vx[m], -5.0, 5.0);
            vy[m] = std::clamp(vy[m], -5.0, 5.0);
            vz[m] = std::clamp(vz[m], -5.0, 5.0);
        }
        trajectory.push_back({t, Eigen::Vector4d(x_mixed, y_mixed, z_mixed, -1.0)}); // -1.0 as placeholder for heading since linear KF doesn't model it
        covariances.push_back(P_mixed);

        current_probs = track.trans_prob_mat_.transpose() * current_probs;
    }

    publishUncertaintyMarkers(track.id, trajectory, covariances);
    return trajectory;
}

void IMMObstacleTrackerPredictionNode::publishUncertaintyMarkers(
    int track_id,
    const std::vector<std::pair<double, Eigen::Vector4d>>& trajectory,
    const std::vector<Eigen::Matrix2d>& covariances)
{
    visualization_msgs::msg::MarkerArray marker_array;    
    int num_steps = trajectory.size();

    for (size_t step = 0; step < num_steps; ++step) {
        // Extract data
        const auto& state = trajectory[step].second;
        double x_mixed = state(0);
        double y_mixed = state(1);
        double z_mixed = state(2);
        Eigen::Matrix2d P_mixed = covariances[step];

        // 1. Solve for eigenvalues and eigenvectors
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigensolver(P_mixed);
        if (eigensolver.info() != Eigen::Success) continue;

        double lambda1 = eigensolver.eigenvalues()(0); // Minor axis variance
        double lambda2 = eigensolver.eigenvalues()(1); // Major axis variance
        Eigen::Vector2d e2 = eigensolver.eigenvectors().col(1); // Major axis vector

        double angle = std::atan2(e2(1), e2(0));

        // 2. Construct the Marker
        visualization_msgs::msg::Marker ellipse_marker;
        ellipse_marker.header.frame_id = frame_id_;
        ellipse_marker.ns = "imm_uncertainty_track_" + std::to_string(track_id);
        ellipse_marker.id = step;
        ellipse_marker.type = visualization_msgs::msg::Marker::CYLINDER;
        ellipse_marker.action = visualization_msgs::msg::Marker::ADD;

        ellipse_marker.pose.position.x = x_mixed;
        ellipse_marker.pose.position.y = y_mixed;
        ellipse_marker.pose.position.z = z_mixed; 

        tf2::Quaternion q;
        q.setRPY(0, 0, angle);
        ellipse_marker.pose.orientation.x = q.x();
        ellipse_marker.pose.orientation.y = q.y();
        ellipse_marker.pose.orientation.z = q.z();
        ellipse_marker.pose.orientation.w = q.w();

        // 3. Scale based on Chi-Square distribution (5.991 for 95% confidence in 2D)
        const double chi2_95 = 5.991; 
        ellipse_marker.scale.x = 2.0 * std::sqrt(lambda2 * chi2_95); // Major axis length
        ellipse_marker.scale.y = 2.0 * std::sqrt(lambda1 * chi2_95); // Minor axis length
        ellipse_marker.scale.z = 0.05; // Flat disc

        // 4. Color and fade out over the horizon
        ellipse_marker.color.r = 0.0;
        ellipse_marker.color.g = 0.5;
        ellipse_marker.color.b = 1.0;
        ellipse_marker.color.a = 0.5 * (1.0 - (static_cast<double>(step) / num_steps)); 

        ellipse_marker.lifetime = rclcpp::Duration::from_seconds(0.1); 

        marker_array.markers.push_back(ellipse_marker);
    }

    // Assuming you have a publisher named marker_pub_ initialized in your node
    pub_unc_sphere_->publish(marker_array);
}

// =================================================================================
// EKF PREDICTION IMPLEMENTATION
// =================================================================================

bool IMMObstacleTrackerPredictionNode::checkTrajectoryCollision(
    const std::vector<std::pair<double, Eigen::Vector4d>>& ped_traj, // Now includes theta in index 3
    const dynus_interfaces::msg::Trajectory& robot_traj,
    double cv_mode_prob,     // Pass the IMM's confidence in the straight-walking model
    double base_lat_dist,    // E.g., 0.4 meters (shoulder width)
    double base_lon_dist)
{
    if (ped_traj.empty() || robot_traj.goals.empty()) return false;

    double robot_dt = robot_traj.dt;
    double ped_dt = prediction_dt_;

    // Dynamically size the lateral threshold based on IMM confidence.
    // If CV prob is 1.0 (walking straight), lateral threshold is tight (base_lat_dist).
    // If CV prob is 0.0 (turning/chaotic), lateral expands to match longitudinal (a circle).
    double dynamic_lat_dist = (cv_mode_prob * base_lat_dist) + ((1.0 - cv_mode_prob) * base_lon_dist);

    for (size_t i = 0; i < robot_traj.goals.size(); ++i) {
        double current_t = i * robot_dt;
        size_t ped_idx = static_cast<size_t>(current_t / ped_dt);

        if (ped_idx >= ped_traj.size() - 1) break;

        // Interpolate position AND heading
        double t0 = ped_traj[ped_idx].first;
        double t1 = ped_traj[ped_idx + 1].first;
        double ratio = (current_t - t0) / (t1 - t0);
        
        Eigen::Vector4d p0 = ped_traj[ped_idx].second;
        Eigen::Vector4d p1 = ped_traj[ped_idx + 1].second;
        Eigen::Vector4d ped_state = p0 + ratio * (p1 - p0); 
        
        // Robot position
        double rx = robot_traj.goals[i].p.x;
        double ry = robot_traj.goals[i].p.y;

        // 1. Find the relative position vector
        double dx = rx - ped_state(0);
        double dy = ry - ped_state(1);
        double theta = ped_state(3); // Pedestrian's mixed heading

        // 2. Rotate the relative position into the pedestrian's local frame
        // This splits the distance into Along-Track (longitudinal) and Cross-Track (lateral)
        double dx_local =  dx * std::cos(theta) + dy * std::sin(theta);
        double dy_local = -dx * std::sin(theta) + dy * std::cos(theta);

        // 3. Elliptical Collision Check
        // Equation of an ellipse: (x/a)^2 + (y/b)^2 <= 1
        double lon_check = std::pow(dx_local / base_lon_dist, 2);
        double lat_check = std::pow(dy_local / dynamic_lat_dist, 2);

        if ((lon_check + lat_check) <= 1.0) {
            return true; // Imminent collision detected!
        }
    }

    return false;
}

bool IMMObstacleTrackerPredictionNode::checkTrajectoryCollisionLinearKF(
    const std::vector<std::pair<double, Eigen::Vector4d>>& ped_traj, 
    const dynus_interfaces::msg::Trajectory& robot_traj,
    double cv_mode_prob,
    double stop_mode_prob)     // Pass the IMM's confidence in the straight-walking model
{

    // Edge case: empty trajectories
    if (ped_traj.empty() || robot_traj.goals.empty()) {
        return false;
    }

    if (stop_mode_prob > 0.5) 
    {
        return false;
    }

    double robot_dt = robot_traj.dt; // Should be 0.01
    double ped_dt = prediction_dt_;;

    // Evaluate collision at every step of the robot's plan
    for (size_t i = 0; i < robot_traj.goals.size(); ++i) {
        double current_t = i * robot_dt;

        // Find the corresponding indices in the pedestrian trajectory
        size_t ped_idx = static_cast<size_t>(current_t / ped_dt);

        // If the robot's plan extends beyond our pedestrian prediction horizon, 
        // we stop checking (or you could assume the pedestrian stays at their last position)
        if (ped_idx >= ped_traj.size() - 1) {
            break; 
        }

        // --- Linear Interpolation of Pedestrian Position ---
        double t0 = ped_traj[ped_idx].first;
        double t1 = ped_traj[ped_idx + 1].first;
        
        Eigen::Vector3d p0 = ped_traj[ped_idx].second.head<3>(); // Extract (x, y, z) from the trajectory point
        Eigen::Vector3d p1 = ped_traj[ped_idx + 1].second.head<3>();

        // How far along are we between t0 and t1? (Values 0.0 to 1.0)
        double interpolation_ratio = (current_t - t0) / (t1 - t0);
        
        Eigen::Vector3d ped_pos_at_t = p0 + interpolation_ratio * (p1 - p0);

        // --- Fetch Robot Position ---
        Eigen::Vector3d robot_pos_at_t(
            robot_traj.goals[i].p.x, 
            robot_traj.goals[i].p.y, 
            robot_traj.goals[i].p.z
        );

        // --- Distance Calculation ---
        double distance;
        distance = (ped_pos_at_t.head<2>() - robot_pos_at_t.head<2>()).norm();

        // --- Collision Check ---
        const double collision_threshold = 1.0;
        if (distance <= collision_threshold) {
            return true; // Imminent collision detected!
        }
    }

    return false; // Path is clear
}

void IMMObstacleTrackerPredictionNode::publishPredictions(const std::vector<Measurement> &measurements)
{
    
    visualization_msgs::msg::MarkerArray markers;
    int id = 0;
    int num_steps = static_cast<int>(prediction_horizon_ / prediction_dt_); // Number of steps
    double d_min = D_INF;
    
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

        std::vector<std::pair<double, Eigen::Vector4d>> predicted_trajectory;
        if (linear_kf_) {
            predicted_trajectory = generatePredictionLinearKF(track);
        } else {
             predicted_trajectory = generatePrediction(track);
        }
        double d_EV_to_ped = D_INF;

        if (use_yield_mode_ && trajectory_initialized_ && !trajectory_.goals.empty() && state_initialized_) {
            double base_lon_dist = 1.0; // [m]
            double base_lat_dist = 0.4; // [m]

            bool collision;
            
            if (!linear_kf_) {
                collision = checkTrajectoryCollision(
                    predicted_trajectory,
                    trajectory_,
                    track.mode_probs(0), // CV mode probability as a proxy for confidence in straight walking
                    base_lat_dist,      // base lateral distance (e.g., shoulder width)
                    base_lon_dist);    // base longitudinal distance (e.g., average step length)

            } else 
            {
                collision = checkTrajectoryCollisionLinearKF(
                    predicted_trajectory,
                    trajectory_,
                    track.mode_probs(0),    // CV mode probability as a proxy for confidence in straight walking
                    track.mode_probs(2));    // STOP mode prob
            }

            d_EV_to_ped = std::sqrt(
                std::pow(track.x(0) - current_state_.pos.x, 2) + 
                std::pow(track.x(1) - current_state_.pos.y, 2)
            );


            if (collision) {
                RCLCPP_WARN(this->get_logger(), "Collision in Immediate space detected for track [%d] with current robot plan!", track.id);
                
                if (d_EV_to_ped < d_min) {
                    d_min = d_EV_to_ped;
                }
            }
        }
        int count = 0;
        for (const auto& pt : predicted_trajectory) {
            t_values.push_back(pt.first);
            x_values.push_back(pt.second(0));
            y_values.push_back(pt.second(1));
            z_values.push_back(pt.second(2));
            count++;
            if (count == (int)(prediction_horizon_ / prediction_dt_)) break;
        }
        double end = 0.1 * (double)count;

        // --- Helper Lambda to generate markers for ANY trajectory ---
        auto add_trajectory_markers = [&](const std::vector<std::pair<double, Eigen::Vector4d>>& traj, 
                                          float r, float g, float b, float a, float scale_x,
                                          const std::string& ns) 
        {
            for (size_t k = 0; k < traj.size() - 1; ++k)
            {
                Eigen::Vector3d p_curr = traj[k].second.head<3>(); // Extract (x, y, z) from the trajectory point
                Eigen::Vector3d p_next = traj[k+1].second.head<3>();
                
                visualization_msgs::msg::Marker marker;
                marker.header.frame_id = frame_id_;
                marker.ns = ns;
                marker.id = id++; // Crucial: Unique ID for every single arrow across BOTH trajectories
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

                marker.points.push_back(start);
                marker.points.push_back(end);

                // Set scale (shaft width, head width, head length)
                marker.scale.x = scale_x; // Varying thickness helps differentiate them
                marker.scale.y = scale_x * 2.0;
                marker.scale.z = scale_x * 2.0;

                // Set custom color passed into the lambda
                marker.color.r = r;
                marker.color.g = g;
                marker.color.b = b;
                marker.color.a = a;

                // Add this marker to the global marker array
                markers.markers.push_back(marker);
            }
        };

        // add_trajectory_markers(predicted_trajectory, 
        //                        1.0, 0.0, 0.0, 1.0, 0.1, "best_mode_traj");

        add_trajectory_markers(predicted_trajectory, 
                               1.0, 1.0, 0.0, 1.0, 0.1, "merged_mode_traj");

        // Check if x_values, y_values, z_values changed much (especially in the beginning, the predicted trajectories can be very short and hard to fit)
        // If they are not changing much, we can skip the polynomial fitting
        double x_diff = abs(x_values.front() - x_values.back());
        double y_diff = abs(y_values.front() - y_values.back());
        double z_diff = abs(z_values.front() - z_values.back());

        double cutoff_length_threshold = 0.0; // TODO: make this a parameter?

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
        pwp.times.push_back(current_time + end); // Assuming uniform time intervals in the predicted trajectory
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
        msg.pos.x = track.x(0);
        msg.pos.y = track.x(1);
        msg.pos.z = track.x(2);
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
        msg.poly_end_time = current_time + end;

        msg.is_agent = false;
        
        // d_immediate_max_ <= d_EV_to_ped <= d_personal_max_
        if (d_EV_to_ped != D_INF && d_EV_to_ped >= d_immediate_max_ && d_EV_to_ped <= d_personal_max_)
        {
        }
        pub_predicted_traj_->publish(msg);

        // Clear the vectors for the next EKF state
        t_values.clear();
        x_values.clear();
        y_values.clear();
        z_values.clear();

    }

    if (use_yield_mode_) {
        if (d_min < D_INF) {
            RCLCPP_WARN(this->get_logger(), "Minimum distance to robot in yield mode: %.2f meters", d_min);
        }
        dynus_interfaces::msg::YieldMode yield_msg;
        // yield_msg.alpha = yield_mode_reduction(d_min);
        yield_msg.alpha = 1.0;

        pub_yield_mode_->publish(yield_msg);
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