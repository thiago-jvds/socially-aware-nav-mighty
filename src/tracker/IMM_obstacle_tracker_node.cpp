#include "tracker/IMM_obstacle_tracker_node.hpp"
#include <cmath>

namespace {

dynus_interfaces::msg::PWPTraj convertPwp2PwpMsg(const PieceWisePol& pwp)
{
    dynus_interfaces::msg::PWPTraj pwp_msg;
    pwp_msg.times.reserve(pwp.times.size());
    pwp_msg.coeff_x.reserve(pwp.coeff_x.size());
    pwp_msg.coeff_y.reserve(pwp.coeff_y.size());
    pwp_msg.coeff_z.reserve(pwp.coeff_z.size());

    for (double t : pwp.times) {
        pwp_msg.times.push_back(t);
    }

    for (const auto& coeff : pwp.coeff_x) {
        dynus_interfaces::msg::CoeffPoly3 msg_coeff;
        msg_coeff.a = coeff(0);
        msg_coeff.b = coeff(1);
        msg_coeff.c = coeff(2);
        msg_coeff.d = coeff(3);
        pwp_msg.coeff_x.push_back(msg_coeff);
    }

    for (const auto& coeff : pwp.coeff_y) {
        dynus_interfaces::msg::CoeffPoly3 msg_coeff;
        msg_coeff.a = coeff(0);
        msg_coeff.b = coeff(1);
        msg_coeff.c = coeff(2);
        msg_coeff.d = coeff(3);
        pwp_msg.coeff_y.push_back(msg_coeff);
    }

    for (const auto& coeff : pwp.coeff_z) {
        dynus_interfaces::msg::CoeffPoly3 msg_coeff;
        msg_coeff.a = coeff(0);
        msg_coeff.b = coeff(1);
        msg_coeff.c = coeff(2);
        msg_coeff.d = coeff(3);
        pwp_msg.coeff_z.push_back(msg_coeff);
    }

    return pwp_msg;
}

}  // namespace

using std::placeholders::_1;

IMMObstacleTrackerNode::IMMObstacleTrackerNode() 
: Node("IMM_obstacle_tracker_node") 
{

    // 1. Parameters
    this->declare_parameter("assignment_tolerance", 1.0);
    this->declare_parameter("prediction_horizon", 3.0);
    this->declare_parameter("prediction_dt", 0.1);
    this->declare_parameter("time_to_delete_old_obstacles", 3.0);
    this->declare_parameter("box_visualization_duration", 3.0);
    this->declare_parameter("velocity_threshold", 0.0);
    this->declare_parameter("acceleration_threshold", 0.1);
    this->declare_parameter("tracker_debug", false);
    this->declare_parameter("frame_id", std::string("map"));

    this->declare_parameter("prob_transition_stay", 0.90);
    this->declare_parameter("imm_adapt_tpm", true);
    this->declare_parameter("imm_adapt_tpm_gain", 0.9);
    this->declare_parameter("imm_model_noise_cv", 0.8);
    this->declare_parameter("imm_model_noise_ca", 0.5);
    this->declare_parameter("imm_model_noise_sta", 0.05);
    this->declare_parameter("imm_p_init_pos_var", 0.5);
    this->declare_parameter("imm_p_init_vel_var", 1.0);
    this->declare_parameter("imm_p_init_acc_var", 5.0);
    this->declare_parameter("imm_r_meas_pos_var", 0.1);
    
    
    // Set parameters
    assignment_tolerance_ = this->get_parameter("assignment_tolerance").as_double();
    prediction_horizon_ = this->get_parameter("prediction_horizon").as_double();
    prediction_dt_ = this->get_parameter("prediction_dt").as_double();
    time_to_delete_old_obstacles_ = this->get_parameter("time_to_delete_old_obstacles").as_double();
    box_visualization_duration_ = this->get_parameter("box_visualization_duration").as_double();
    velocity_threshold_ = this->get_parameter("velocity_threshold").as_double();
    acceleration_threshold_ = this->get_parameter("acceleration_threshold").as_double();
    tracker_debug_ = this->get_parameter("tracker_debug").as_bool();
    frame_id_ = this->get_parameter("frame_id").as_string();

    // IMM specific params
    imm_params_.prob_transition_stay = this->get_parameter("prob_transition_stay").as_double();
    imm_params_.imm_adapt_tpm = this->get_parameter("imm_adapt_tpm").as_bool();
    imm_params_.imm_adapt_tpm_gain = this->get_parameter("imm_adapt_tpm_gain").as_double();
    imm_params_.model_noise_cv = this->get_parameter("imm_model_noise_cv").as_double();
    imm_params_.model_noise_ca = this->get_parameter("imm_model_noise_ca").as_double();
    imm_params_.model_noise_sta = this->get_parameter("imm_model_noise_sta").as_double();
    imm_params_.p_init_pos_var = this->get_parameter("imm_p_init_pos_var").as_double();
    imm_params_.p_init_vel_var = this->get_parameter("imm_p_init_vel_var").as_double();
    imm_params_.p_init_acc_var = this->get_parameter("imm_p_init_acc_var").as_double();
    imm_params_.r_meas_pos_var = this->get_parameter("imm_r_meas_pos_var").as_double();

    // 2. Pub/Sub
    sub_detections_ = this->create_subscription<vision_msgs::msg::Detection3DArray>(
        "detected_objects", 10, 
        std::bind(&IMMObstacleTrackerNode::detectionsCallback, this, _1));

    sub_state_ = this->create_subscription<dynus_interfaces::msg::State>(
        "state", 10, 
        std::bind(&IMMObstacleTrackerNode::stateCallback, this, _1));

    predicted_traj_pub_ = this->create_publisher<dynus_interfaces::msg::DynTraj>("predicted_trajs", 10);
    tracker_bbox_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("cluster_bounding_boxes", 10);
    tracker_prediction_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("tracked_obstacles", 10);

    RCLCPP_INFO(this->get_logger(), "IMM Obstacle Tracker Node Initialized.");
}

void IMMObstacleTrackerNode::stateCallback(const dynus_interfaces::msg::State::SharedPtr msg)
{
    current_state_ = *msg;
    state_initialized_ = true;
}


void IMMObstacleTrackerNode::detectionsCallback(const vision_msgs::msg::Detection3DArray::SharedPtr msg)
{
    // --- Step 1: Parse Inputs ---
    std::vector<Measurement> current_measurements;
    current_measurements.reserve(msg->detections.size());

    for (const auto& det : msg->detections) {
        Measurement m;
        m.centroid = Eigen::Vector3d(det.bbox.center.position.x, det.bbox.center.position.y, det.bbox.center.position.z);
        m.source_stamp = det.header.stamp;
        m.has_match = false;
        
        m.assigned_track.bbox = Eigen::Vector3d(det.bbox.size.x, det.bbox.size.y, det.bbox.size.z);
        
        current_measurements.push_back(m);
    }

    // Obtain msg time in seconds, with fallback to now() if not provided
    builtin_interfaces::msg::Time source_stamp = msg->header.stamp;
    if (source_stamp.sec == 0 && source_stamp.nanosec == 0) {
        source_stamp = this->now();
    }
    const double current_time_sec =
        static_cast<double>(source_stamp.sec) + static_cast<double>(source_stamp.nanosec) * 1e-9;

    // --- Step 2: Lifecycle ---
    deleteOldTracks(current_time_sec);
    // Reserve memory to prevent invalidation during push_back
    tracks_.reserve(tracks_.size() + current_measurements.size());

    std::vector<Eigen::VectorXd> x_mixed(NUM_MODES);
    std::vector<Eigen::MatrixXd> P_mixed(NUM_MODES);
    
    // --- Step 3: Predict for current tracks --
    for (auto& track : tracks_) {
        interaction(track, x_mixed, P_mixed);
        double actual_dt = current_time_sec - track.time_last_updated;
        if (actual_dt < 1e-6)
            actual_dt = prediction_dt_;
        predict(track, x_mixed, P_mixed, actual_dt); 
        combine(track);
    }

    // --- Step 4: Association & Update ---
    std::vector<int> assignments = associateMeasurements(current_measurements, tracks_, assignment_tolerance_);

    for (int i = 0; i < static_cast<int>(assignments.size()); i++){
        int match_idx = assignments[i];
        auto& meas = current_measurements[i];
        Eigen::Vector3d raw_bbox = meas.assigned_track.bbox;
        Eigen::Vector3d& centroid = meas.centroid;

        double meas_time_sec = static_cast<double>(meas.source_stamp.sec) 
                                            + static_cast<double>(meas.source_stamp.nanosec) * 1e-9;

        if (match_idx >= 0) {
            IMMTrack& matched_track = tracks_[match_idx];

            // cold start
            if (matched_track.is_first_meas) { 
                matched_track.is_first_meas = false;

                double dx = centroid(0) - matched_track.models[0]->x(0); 
                double dy = centroid(1) - matched_track.models[0]->x(1);
                double dz = centroid(2) - matched_track.models[0]->x(2);
                
                // Time delta between first and second measurement
                double dt = meas_time_sec - matched_track.time_last_updated;
                if (dt < 1e-6) {
                    dt = prediction_dt_;
                }
                
                double vx_init = dx / dt;
                double vy_init = dy / dt;
                double vz_init = dz / dt;

                for (int m = 0; m < NUM_MODES; ++m) 
                {
                    matched_track.models[m]->x(3) = vx_init;
                    matched_track.models[m]->x(4) = vy_init;
                    matched_track.models[m]->x(5) = vz_init;
                }
                
            }
            
            update(matched_track, centroid, meas_time_sec, raw_bbox);
            combine(matched_track);

            meas.assigned_track = matched_track;
            meas.has_match = true;
            
        } else {
            RCLCPP_DEBUG(this->get_logger(), "No match for detection at (%.2f, %.2f). Creating new track.", 
                centroid(0), centroid(1));
            IMMTrack new_track(
                state_dim_,meas_dim_, meas_time_sec, 
                centroid, raw_bbox, track_id_++,
                imm_params_
            );

            tracks_.push_back(new_track);            
            
            // Store the new track in the measurement
            meas.assigned_track = new_track;
            meas.has_match = true;
        }
    }

    // --- Step 5: Publish ---
    publishPredictions(current_measurements, current_time_sec);
}


// =================================================================================
// ASSOCIATION & DELETION 
// =================================================================================

/*   
 * @brief O(max(current_meas.size(), tracks_.size())^3)
 * (for max_N ~ 10 detections, 1000 ops, negligible)
 */
std::vector<int> IMMObstacleTrackerNode::associateMeasurements(
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

void IMMObstacleTrackerNode::deleteOldTracks(double reference_time_sec) {
    
    auto it = tracks_.begin();
    while (it != tracks_.end()) {
        if ((reference_time_sec - it->time_last_updated) > time_to_delete_old_obstacles_) {
            it = tracks_.erase(it);
        } else {
            ++it;
        }
    }
}


// =================================================================================
// IMM MATH IMPLEMENTATION
// =================================================================================

void IMMObstacleTrackerNode::interaction(
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
            
            P_mixed[j] += w_ij * (track.models[i]->P + diff * diff.transpose());
        }
    }
}

void IMMObstacleTrackerNode::predict(
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
    
void IMMObstacleTrackerNode::update(IMMTrack &track, 
    const Eigen::VectorXd &z, 
    double time_updated, 
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
        
void IMMObstacleTrackerNode::combine(IMMTrack& track)
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
        
        // P += prob * ( P_mode + spread_term )
        track.P += track.mode_probs(m) * (track.models[m]->P + diff * diff.transpose());
    }
}
            

// =================================================================================
// PREDICTION IMPLEMENTATION
// =================================================================================

void IMMObstacleTrackerNode::publishPredictions(
    const std::vector<Measurement>& measurements,
    double current_time_sec)
{
    visualization_msgs::msg::MarkerArray prediction_markers;
    visualization_msgs::msg::MarkerArray bbox_markers;
    int marker_id = 0;
    double min_dist = std::numeric_limits<double>::max();

    for (int i = 0; i < static_cast<int>(measurements.size()); ++i)
    {
        const IMMTrack& track = measurements[i].assigned_track;

        if (report_min_dist_to_ego_ && state_initialized_) {
            Eigen::Vector2d d_vec = Eigen::Vector2d(measurements[i].centroid.x(), measurements[i].centroid.y()) - 
                                Eigen::Vector2d(current_state_.pos.x, current_state_.pos.y);
            double dist_to_ego = d_vec.norm();
            if (dist_to_ego < min_dist) {
                min_dist = dist_to_ego;
            }
        }

        Eigen::Vector3d cur_pos = track.x.head<3>();
        Eigen::Vector3d vel(track.x[3], track.x[4], track.x[5]);
        Eigen::Vector3d acc = Eigen::Vector3d::Zero();

        // Use IMM mixed-model prediction whenever all three modes are available.
        // Mode order is fixed at track creation: [CV, CA, Stationary].
        if (track.models.size() >= 3 && track.mode_probs.size() >= 3)
        {
            const double mu_cv = track.mode_probs(0);
            const double mu_ca = track.mode_probs(1);
            const double mu_sta = 0.0;

            const Eigen::VectorXd& x_cv = track.models[0]->x;
            const Eigen::Vector3d p_cv = x_cv.head<3>();
            const Eigen::Vector3d v_cv = x_cv.segment<3>(3);
            const Eigen::Vector3d a_cv = x_cv.segment<3>(6);


            const Eigen::VectorXd& x_ca = track.models[1]->x;
            const Eigen::Vector3d p_ca = x_ca.head<3>();
            const Eigen::Vector3d v_ca = x_ca.segment<3>(3);
            const Eigen::Vector3d a_ca = x_ca.segment<3>(6);

            // const Eigen::VectorXd& x_sta = track.models[2]->x;
            // const Eigen::Vector3d p_sta = x_sta.head<3>();
            // const Eigen::Vector3d v_sta = x_sta.segment<3>(3);
            // const Eigen::Vector3d a_sta = x_sta.segment<3>(6);

            cur_pos = mu_cv * p_cv + mu_ca * p_ca; // + mu_sta * p_sta;
            vel = mu_cv * v_cv + mu_ca * v_ca; // + mu_sta * v_sta;
            acc = mu_ca * a_ca + mu_cv * a_cv; // + mu_sta * a_sta; 
        }

        // double speed = vel.norm();

        // // Clamp velocity to max obstacle velocity
        // if (speed > max_obstacle_velocity_)
        // {
        //     const double scale = max_obstacle_velocity_ / speed;
        //     vel *= scale;
        // }

        // double acc_mag = acc.norm();

        // // Clamp acceleration to max obstacle acceleration
        // if (acc_mag > max_obstacle_acceleration_)
        // {
        //     const double scale = max_obstacle_acceleration_ / acc_mag;
        //     acc *= scale;
        // }

        // IMM mixed prediction: P(t) = p0 + v0*t + 0.5*a0*t^2
        PieceWisePol pwp;
        pwp.times.push_back(current_time_sec);
        pwp.times.push_back(current_time_sec + prediction_horizon_);
        pwp.coeff_x.push_back({0.0, 0.5 * acc.x(), vel.x(), cur_pos.x()});
        pwp.coeff_y.push_back({0.0, 0.5 * acc.y(), vel.y(), cur_pos.y()});
        pwp.coeff_z.push_back({0.0, 0.5 * acc.z(), vel.z(), cur_pos.z()});

        // DynTraj message
        dynus_interfaces::msg::DynTraj msg;
        msg.header.frame_id = frame_id_;
        msg.id = track.id;
        msg.pos.x = cur_pos.x();
        msg.pos.y = cur_pos.y();
        msg.pos.z = cur_pos.z();
        msg.mode = "pwp";
        msg.bbox = {track.bbox.x(), track.bbox.y(), track.bbox.z()};
        msg.pwp = convertPwp2PwpMsg(pwp);
        msg.ekf_cov_p = {track.P(0, 0), track.P(1, 1), track.P(2, 2)};
        msg.ekf_cov_q = {track.models[0]->Q(0, 0), track.models[0]->Q(1, 1), track.models[0]->Q(2, 2)};
        msg.ekf_cov_r = {track.R(0, 0), track.R(1, 1), track.R(2, 2)};
        msg.poly_cov = {0.0, 0.0, 0.0};
        msg.poly_coeffs_x = {vel.x(), cur_pos.x()};
        msg.poly_coeffs_y = {vel.y(), cur_pos.y()};
        msg.poly_coeffs_z = {vel.z(), cur_pos.z()};
        msg.poly_start_time = current_time_sec;
        msg.poly_end_time = current_time_sec + prediction_horizon_;
        msg.is_agent = false;
        msg.mu = {track.mode_probs(0), track.mode_probs(1), 0.0};  // cv, ca, st
        msg.use_mu = true;
        msg.header.stamp = this->now();
        predicted_traj_pub_->publish(msg);

        double dt_vis = prediction_dt_ * 0.25;
        int num_vis_samples = static_cast<int>(prediction_horizon_ / dt_vis) + 1;

        visualization_msgs::msg::Marker line_marker;
        line_marker.header.frame_id = frame_id_;
        line_marker.header.stamp = measurements[i].source_stamp;
        line_marker.ns = "predictions";
        line_marker.id = marker_id++;
        line_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        line_marker.action = visualization_msgs::msg::Marker::ADD;
        line_marker.lifetime = rclcpp::Duration::from_seconds(0.2);
        line_marker.scale.x = 0.05;
        line_marker.color = track.color;
        line_marker.color.a = 1.0f;
        line_marker.points.reserve(num_vis_samples);

        for (int step = 0; step < num_vis_samples; ++step)
        {
            double t = step * dt_vis;
            geometry_msgs::msg::Point pt;
            pt.x = cur_pos.x() + vel.x() * t + 0.5 * acc.x() * t * t;
            pt.y = cur_pos.y() + vel.y() * t + 0.5 * acc.y() * t * t;
            pt.z = cur_pos.z() + vel.z() * t + 0.5 * acc.z() * t * t;
            line_marker.points.push_back(pt);
        }
        prediction_markers.markers.push_back(line_marker);

        const auto& measurement = measurements[i];
        const double cx = measurement.centroid[0];
        const double cy = measurement.centroid[1];
        const double cz = measurement.centroid[2];
        const double hx = std::max(measurement.assigned_track.bbox[0], 0.05) * 0.5;
        const double hy = std::max(measurement.assigned_track.bbox[1], 0.05) * 0.5;
        const double hz = std::max(measurement.assigned_track.bbox[2], 0.05) * 0.5;

        // --- Wireframe box (LINE_LIST: 12 edges = 24 points) ---
        visualization_msgs::msg::Marker wire;
        wire.header.frame_id = frame_id_;
        wire.header.stamp = measurements[i].source_stamp;
        wire.ns = "wireframe";
        wire.id = track.id;
        wire.type = visualization_msgs::msg::Marker::LINE_LIST;
        wire.action = visualization_msgs::msg::Marker::ADD;
        wire.lifetime = rclcpp::Duration::from_seconds(0.2);
        wire.scale.x = 0.06;    // edge thickness
        wire.color = track.color;
        wire.color.a = 1.0f;
        wire.pose.orientation.w = 1.0;
        wire.points.reserve(24);

        auto addEdge = [&](double x1, double y1, double z1,
                        double x2, double y2, double z2) {
            geometry_msgs::msg::Point p1, p2;
            p1.x = x1; p1.y = y1; p1.z = z1;
            p2.x = x2; p2.y = y2; p2.z = z2;
            wire.points.push_back(p1);
            wire.points.push_back(p2);
        };

        // Bottom face
        addEdge(cx-hx, cy-hy, cz-hz, cx+hx, cy-hy, cz-hz);
        addEdge(cx+hx, cy-hy, cz-hz, cx+hx, cy+hy, cz-hz);
        addEdge(cx+hx, cy+hy, cz-hz, cx-hx, cy+hy, cz-hz);
        addEdge(cx-hx, cy+hy, cz-hz, cx-hx, cy-hy, cz-hz);
        // Top face
        addEdge(cx-hx, cy-hy, cz+hz, cx+hx, cy-hy, cz+hz);
        addEdge(cx+hx, cy-hy, cz+hz, cx+hx, cy+hy, cz+hz);
        addEdge(cx+hx, cy+hy, cz+hz, cx-hx, cy+hy, cz+hz);
        addEdge(cx-hx, cy+hy, cz+hz, cx-hx, cy-hy, cz+hz);
        // Vertical edges
        addEdge(cx-hx, cy-hy, cz-hz, cx-hx, cy-hy, cz+hz);
        addEdge(cx+hx, cy-hy, cz-hz, cx+hx, cy-hy, cz+hz);
        addEdge(cx+hx, cy+hy, cz-hz, cx+hx, cy+hy, cz+hz);
        addEdge(cx-hx, cy+hy, cz-hz, cx-hx, cy+hy, cz+hz);

        bbox_markers.markers.push_back(wire);

        // --- Centroid sphere ---
        visualization_msgs::msg::Marker sphere;
        sphere.header.frame_id = frame_id_;
        sphere.header.stamp = measurements[i].source_stamp;
        sphere.ns = "centroid";
        sphere.id = track.id;
        sphere.type = visualization_msgs::msg::Marker::SPHERE;
        sphere.action = visualization_msgs::msg::Marker::ADD;
        sphere.lifetime = rclcpp::Duration::from_seconds(0.2);
        sphere.pose.position.x = cx;
        sphere.pose.position.y = cy;
        sphere.pose.position.z = cz;
        sphere.pose.orientation.w = 1.0;
        sphere.scale.x = 0.2;
        sphere.scale.y = 0.2;
        sphere.scale.z = 0.2;
        sphere.color = track.color;
        sphere.color.a = 1.0f;

        bbox_markers.markers.push_back(sphere);

        // --- Dominant IMM mode text (CV/CA) above cube ---
        visualization_msgs::msg::Marker mode_text;
        mode_text.header.frame_id = frame_id_;
        mode_text.header.stamp = measurements[i].source_stamp;
        mode_text.ns = "mode_text";
        mode_text.id = track.id;
        mode_text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
        mode_text.action = visualization_msgs::msg::Marker::ADD;
        mode_text.lifetime = rclcpp::Duration::from_seconds(0.2);
        mode_text.pose.position.x = cx;
        mode_text.pose.position.y = cy;
        mode_text.pose.position.z = cz + hz + 0.25;
        mode_text.pose.orientation.w = 1.0;
        mode_text.scale.z = 0.28;
        mode_text.color.r = 0.0f;
        mode_text.color.g = 0.0f;
        mode_text.color.b = 0.0f;
        mode_text.color.a = 1.0f;

        if (track.mode_probs.size() >= 2)
        {
            const bool is_cv = track.mode_probs(0) >= track.mode_probs(1);
            const double p = is_cv ? track.mode_probs(0) : track.mode_probs(1);
            std::string prob = std::to_string(p);
            const std::size_t dot_pos = prob.find('.');
            if (dot_pos != std::string::npos && dot_pos + 3 < prob.size())
            {
                prob = prob.substr(0, dot_pos + 3);
            }
            mode_text.text = (is_cv ? "CV " : "CA ") + prob;
        }
        else
        {
            mode_text.text = "CV";
        }

        bbox_markers.markers.push_back(mode_text);

    }

    tracker_prediction_pub_->publish(prediction_markers);
    tracker_bbox_pub_->publish(bbox_markers);

    if (report_min_dist_to_ego_ && state_initialized_) {
        min_so_far = std::min(min_so_far, min_dist);
        RCLCPP_INFO(this->get_logger(), "Closest Obstacle Distance to Ego: %.2f meters", min_so_far);
    }
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<IMMObstacleTrackerNode>(); 
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}