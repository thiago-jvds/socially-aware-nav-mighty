#include <mighty/trajectory_tracker.hpp>
#include <fstream>

TrajectoryTracker::TrajectoryTracker()
    : Node("trajectory_tracker"), traj_start_time_(0, 0, RCL_ROS_TIME) {
  // Declare parameters
  this->declare_parameter("control_rate", 50.0);
  this->declare_parameter("max_velocity", 0.5);
  this->declare_parameter("max_angular_velocity", 1.5);
  this->declare_parameter("stopping_radius", 0.3);
  this->declare_parameter("Kp_along", 1.0);
  this->declare_parameter("Kp_cross", 2.0);
  this->declare_parameter("Kp_yaw", 2.0);
  this->declare_parameter("w_smoothing", 0.3);
  this->declare_parameter("use_hardware", false);
  this->declare_parameter("map_frame_id", "map");

  // Get parameters
  control_rate_ = this->get_parameter("control_rate").as_double();
  max_velocity_ = this->get_parameter("max_velocity").as_double();
  max_angular_velocity_ = this->get_parameter("max_angular_velocity").as_double();
  stopping_radius_ = this->get_parameter("stopping_radius").as_double();
  Kp_along_ = this->get_parameter("Kp_along").as_double();
  Kp_cross_ = this->get_parameter("Kp_cross").as_double();
  Kp_yaw_ = this->get_parameter("Kp_yaw").as_double();
  w_smoothing_ = this->get_parameter("w_smoothing").as_double();
  use_hardware_ = this->get_parameter("use_hardware").as_bool();
  map_frame_id_ = this->get_parameter("map_frame_id").as_string();

  // Publishers
  std::string cmd_vel_topic = use_hardware_ ? "cmd_vel_auto" : "cmd_vel";
  pub_cmd_vel_ = this->create_publisher<geometry_msgs::msg::Twist>(cmd_vel_topic, 10);
  pub_ref_marker_ = this->create_publisher<visualization_msgs::msg::Marker>("tracker_ref_marker", 10);

  // Subscribers — only trajectory and state, no goal topic needed
  sub_trajectory_ = this->create_subscription<dynus_interfaces::msg::Trajectory>(
      "trajectory", 10, std::bind(&TrajectoryTracker::trajectoryCallback, this, std::placeholders::_1));
  sub_state_ = this->create_subscription<dynus_interfaces::msg::State>(
      "state", 10, std::bind(&TrajectoryTracker::stateCallback, this, std::placeholders::_1));

  // Control timer
  timer_ = this->create_wall_timer(
      std::chrono::milliseconds(static_cast<int>(1000.0 / control_rate_)),
      std::bind(&TrajectoryTracker::controlCallback, this));

  RCLCPP_INFO(this->get_logger(), "Trajectory tracker started (time-based FF+FB, rate=%.0f Hz)", control_rate_);
}

// ---------------------------------------------------------------------------
// Callbacks
// ---------------------------------------------------------------------------

void TrajectoryTracker::trajectoryCallback(const dynus_interfaces::msg::Trajectory::SharedPtr msg) {
  if (msg->goals.size() < 2 || msg->dt <= 0.0) return;

  // On every new trajectory, find closest point to robot and set time offset.
  // We anchor to the closest waypoint so interpolation starts where the robot
  // actually is. To avoid the speed ramp-up problem (closest=0 → t_elapsed≈0
  // → tiny ref_speed → robot can't overcome friction → closest stays 0),
  // we skip at least a few indices ahead so the tracker reads cruise-speed
  // portions of the trajectory.
  if (state_initialized_) {
    double rx = current_state_.pos.x;
    double ry = current_state_.pos.y;
    double min_d2 = std::numeric_limits<double>::max();
    size_t closest = 0;
    for (size_t i = 0; i < msg->goals.size(); ++i) {
      double dx = msg->goals[i].p.x - rx;
      double dy = msg->goals[i].p.y - ry;
      double d2 = dx * dx + dy * dy;
      if (d2 < min_d2) { min_d2 = d2; closest = i; }
    }
    // Skip past ramp-up: start at least a few points ahead of closest
    size_t min_skip = std::min(static_cast<size_t>(10), msg->goals.size() - 1);
    size_t start_idx = std::max(closest, min_skip);
    double offset = start_idx * msg->dt;
    traj_start_time_ = this->now() - rclcpp::Duration::from_seconds(offset);
  } else if (!trajectory_initialized_) {
    traj_start_time_ = this->now();
  }

  current_traj_id_ = msg->trajectory_id;
  trajectory_ = *msg;
  trajectory_initialized_ = true;
}

void TrajectoryTracker::stateCallback(const dynus_interfaces::msg::State::SharedPtr msg) {
  current_state_ = *msg;
  state_initialized_ = true;
}

// ---------------------------------------------------------------------------
// Interpolation
// ---------------------------------------------------------------------------

TrajectoryTracker::RefPoint TrajectoryTracker::interpolateTrajectory(double t_elapsed) {
  const auto& goals = trajectory_.goals;
  double dt = trajectory_.dt;
  size_t n = goals.size();

  // Compute fractional index
  double frac_idx = t_elapsed / dt;

  // Clamp to trajectory bounds
  if (frac_idx <= 0.0) {
    const auto& g = goals.front();
    double yaw = std::atan2(g.v.y, g.v.x);
    return {g.p.x, g.p.y, g.v.x, g.v.y, yaw, g.dyaw};
  }
  if (frac_idx >= static_cast<double>(n - 1)) {
    const auto& g = goals.back();
    double yaw = std::atan2(g.v.y, g.v.x);
    // At end of trajectory, zero velocity
    return {g.p.x, g.p.y, 0.0, 0.0, yaw, 0.0};
  }

  // Linear interpolation between goals[i] and goals[i+1]
  size_t i = static_cast<size_t>(frac_idx);
  double t = frac_idx - static_cast<double>(i);
  const auto& g0 = goals[i];
  const auto& g1 = goals[i + 1];

  RefPoint ref;
  ref.x  = (1.0 - t) * g0.p.x + t * g1.p.x;
  ref.y  = (1.0 - t) * g0.p.y + t * g1.p.y;
  ref.vx = (1.0 - t) * g0.v.x + t * g1.v.x;
  ref.vy = (1.0 - t) * g0.v.y + t * g1.v.y;

  // Yaw from interpolated velocity direction
  double speed = std::sqrt(ref.vx * ref.vx + ref.vy * ref.vy);
  if (speed > 0.05) {
    ref.yaw = std::atan2(ref.vy, ref.vx);
  } else {
    // Low speed: interpolate yaw from the goals' velocity directions
    double yaw0 = std::atan2(g0.v.y, g0.v.x);
    double yaw1 = std::atan2(g1.v.y, g1.v.x);
    ref.yaw = yaw0 + t * wrapPi(yaw1 - yaw0);
  }

  ref.dyaw = (1.0 - t) * g0.dyaw + t * g1.dyaw;
  return ref;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

double TrajectoryTracker::wrapPi(double angle) {
  while (angle > M_PI) angle -= 2.0 * M_PI;
  while (angle < -M_PI) angle += 2.0 * M_PI;
  return angle;
}

double TrajectoryTracker::yawFromQuat(const geometry_msgs::msg::Quaternion& q) {
  return std::atan2(2.0 * (q.w * q.z + q.x * q.y),
                    1.0 - 2.0 * (q.y * q.y + q.z * q.z));
}

// ---------------------------------------------------------------------------
// Control loop
// ---------------------------------------------------------------------------

void TrajectoryTracker::controlCallback() {
  if (!state_initialized_ || !trajectory_initialized_ || trajectory_.goals.size() < 2) {
    return;
  }

  // --- Current robot state ---
  double rx = current_state_.pos.x;
  double ry = current_state_.pos.y;
  double r_yaw = yawFromQuat(current_state_.quat);

  // --- Time-based interpolation of reference ---
  double t_elapsed = (this->now() - traj_start_time_).seconds();
  RefPoint ref = interpolateTrajectory(t_elapsed);

  double ref_speed = std::sqrt(ref.vx * ref.vx + ref.vy * ref.vy);

  // When reference speed is too low, atan2-derived yaw is noise.
  // Use bearing to the next trajectory point if far enough away,
  // otherwise hold current yaw to avoid spinning in place.
  if (ref_speed < 0.05) {
    double dx = ref.x - rx;
    double dy = ref.y - ry;
    double dist = std::sqrt(dx * dx + dy * dy);
    if (dist > 0.1) {
      ref.yaw = std::atan2(dy, dx);
    } else {
      ref.yaw = r_yaw;
    }
  }

  // --- Check if at end of trajectory ---
  const auto& last = trajectory_.goals.back();
  double dx_goal = last.p.x - rx;
  double dy_goal = last.p.y - ry;
  double dist_to_goal = std::sqrt(dx_goal * dx_goal + dy_goal * dy_goal);

  double traj_duration = (trajectory_.goals.size() - 1) * trajectory_.dt;
  bool past_end = t_elapsed >= traj_duration;

  if (dist_to_goal < stopping_radius_ && (past_end || ref_speed < 0.05)) {
    geometry_msgs::msg::Twist twist;
    pub_cmd_vel_->publish(twist);
    prev_v_cmd_ = 0.0;
    prev_w_cmd_ = 0.0;
    return;
  }

  // --- Position error in body frame ---
  double ex_world = ref.x - rx;
  double ey_world = ref.y - ry;
  double cos_yaw = std::cos(r_yaw);
  double sin_yaw = std::sin(r_yaw);
  double e_along = cos_yaw * ex_world + sin_yaw * ey_world;
  double e_cross = -sin_yaw * ex_world + cos_yaw * ey_world;

  // --- Yaw error ---
  double yaw_error = wrapPi(ref.yaw - r_yaw);

  // --- Feedforward + Feedback ---

  // Linear velocity: use planned speed directly as feedforward
  double v_ff = ref_speed;
  double v_fb = Kp_along_ * e_along;
  double v_cmd = v_ff + v_fb;
  v_cmd = std::clamp(v_cmd, 0.0, max_velocity_);

  // Angular velocity
  double w_ff = ref.dyaw;
  double w_fb_yaw = Kp_yaw_ * yaw_error;
  double w_fb_cross = Kp_cross_ * e_cross;
  double speed_scale = std::min(1.0, std::abs(v_cmd) / 0.3);
  double w_cmd = w_ff + w_fb_yaw + w_fb_cross * speed_scale;
  w_cmd = std::clamp(w_cmd, -max_angular_velocity_, max_angular_velocity_);

  // --- Turn-in-place for large heading errors ---
  double abs_yaw_err = std::abs(yaw_error);
  if (abs_yaw_err > M_PI / 3.0) {
    v_cmd = 0.0;
    w_cmd = Kp_yaw_ * yaw_error;
    w_cmd = std::clamp(w_cmd, -max_angular_velocity_, max_angular_velocity_);
  } else if (abs_yaw_err > M_PI / 6.0) {
    double factor = 1.0 - (abs_yaw_err - M_PI / 6.0) / (M_PI / 3.0 - M_PI / 6.0);
    v_cmd *= std::max(0.1, factor);
  }

  // --- Smooth angular velocity ---
  w_cmd = w_smoothing_ * prev_w_cmd_ + (1.0 - w_smoothing_) * w_cmd;
  prev_v_cmd_ = v_cmd;
  prev_w_cmd_ = w_cmd;

  // --- Debug log to file ---
  {
    static std::ofstream log_file("/tmp/tracker_debug.txt", std::ios::app);
    static int log_count = 0;
    if (log_count % 10 == 0) {  // log every 10th cycle (~5 Hz at 50 Hz control)
      log_file << "t=" << t_elapsed
               << " pos=(" << rx << "," << ry << ")"
               << " r_yaw=" << r_yaw
               << " quat=(" << current_state_.quat.x << "," << current_state_.quat.y
               << "," << current_state_.quat.z << "," << current_state_.quat.w << ")"
               << " ref=(" << ref.x << "," << ref.y << ")"
               << " ref_yaw=" << ref.yaw << " ref_speed=" << ref_speed
               << " yaw_err=" << yaw_error
               << " e_along=" << e_along << " e_cross=" << e_cross
               << " v_cmd=" << v_cmd << " w_cmd=" << w_cmd
               << " dist_goal=" << dist_to_goal << " past_end=" << past_end
               << " traj_sz=" << trajectory_.goals.size()
               << " traj_id=" << current_traj_id_
               << "\n";
      log_file.flush();
    }
    log_count++;
  }

  // --- Publish cmd_vel ---
  geometry_msgs::msg::Twist twist;
  twist.linear.x = v_cmd;
  twist.angular.z = w_cmd;
  pub_cmd_vel_->publish(twist);

  // --- Visualize reference point ---
  visualization_msgs::msg::Marker marker;
  marker.header.stamp = this->now();
  marker.header.frame_id = map_frame_id_;
  marker.ns = "tracker_ref";
  marker.id = 0;
  marker.type = visualization_msgs::msg::Marker::SPHERE;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.pose.position.x = ref.x;
  marker.pose.position.y = ref.y;
  marker.pose.position.z = 0.3;
  marker.pose.orientation.w = 1.0;
  marker.scale.x = 0.2;
  marker.scale.y = 0.2;
  marker.scale.z = 0.2;
  marker.color.r = 0.0;
  marker.color.g = 1.0;
  marker.color.b = 0.0;
  marker.color.a = 0.9;
  marker.lifetime = rclcpp::Duration::from_seconds(0.2);
  pub_ref_marker_->publish(marker);
}

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<TrajectoryTracker>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
