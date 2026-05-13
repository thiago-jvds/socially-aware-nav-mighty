#include <mighty/ff_fb_controller.hpp>
#include <algorithm>

FFFB::FFFB()
    : Node("ff_fb"), state_initialized_(false), path_initialized_(false) {
  // Declare parameters

  this->declare_parameter("max_velocity", 1.5);
  this->declare_parameter("max_angular_velocity", 3.0);  // Max turn rate (rad/s)
  this->declare_parameter("control_rate", 2.0);             // Control loop rate (Hz)

  // Get parameters
  max_velocity_ = this->get_parameter("max_velocity").as_double();
  max_angular_velocity_ = this->get_parameter("max_angular_velocity").as_double();
  control_rate_ = this->get_parameter("control_rate").as_double();

  // Publishers and Subscribers
  std::string cmd_vel_string = "cmd_vel";
  pub_cmd_vel_ = this->create_publisher<geometry_msgs::msg::Twist>(cmd_vel_string, 10);

  sub_path_ = this->create_subscription<dynus_interfaces::msg::VwCommandList>(
      "spatial_temporal_cmd", 10, std::bind(&FFFB::pathCallback, this, std::placeholders::_1));

  sub_state_ = this->create_subscription<dynus_interfaces::msg::State>(
      "state", 10, std::bind(&FFFB::stateCallback, this, std::placeholders::_1));

  // Control timer
  timer_ =
      this->create_wall_timer(std::chrono::milliseconds(static_cast<int>(1000.0 / control_rate_)),
                              std::bind(&FFFB::controlCallback, this));
}

void FFFB::pathCallback(const dynus_interfaces::msg::VwCommandList::SharedPtr msg) {
  current_idx = 1;
  path_ = *msg;
  path_initialized_ = true;
}

void FFFB::stateCallback(const dynus_interfaces::msg::State::SharedPtr msg) {
  current_state_ = *msg;
  state_initialized_ = true;
}

double FFFB::wrapPi(double angle) {
  while (angle > M_PI) angle -= 2.0 * M_PI;
  while (angle < -M_PI) angle += 2.0 * M_PI;
  return angle;
}


void FFFB::controlCallback() {
  if (!state_initialized_ || !path_initialized_ || path_.commands.empty()) {
    return;
  }

  if (current_idx >= path_.commands.size()) {
    geometry_msgs::msg::Twist twist;
    twist.linear.x = 0.0;
    twist.angular.z = 0.0;
    pub_cmd_vel_->publish(twist);
    last_v_cmd_ = 0.0;
    last_w_cmd_ = 0.0;
    return;
  }

  double v_ref = path_.commands[current_idx].v.x;
  double w_ref = path_.commands[current_idx].dyaw;
  double x_target = path_.commands[current_idx].p.x;
  double y_target = path_.commands[current_idx].p.y;
  double yaw_target = path_.commands[current_idx].yaw;

  // ============================================================
  // CONTROL LAW
  // ============================================================

  // Get current yaw from quaternion
  double current_yaw = std::atan2(2.0 * (current_state_.quat.w * current_state_.quat.z +
                                         current_state_.quat.x * current_state_.quat.y),
                                  1.0 - 2.0 * (current_state_.quat.y * current_state_.quat.y +
                                               current_state_.quat.z * current_state_.quat.z));

  double current_v = current_state_.vel.x * std::cos(current_yaw) + current_state_.vel.y * std::sin(current_yaw);

  // Compute heading error to lookahead point
  double dx = x_target - current_state_.pos.x;
  double dy = y_target - current_state_.pos.y;
  double dist_to_target = std::hypot(dx, dy);
  // If we are close enough to the current waypoint, move to the next one
  double lookahead_distance = 0.2; // 20 cm
  if (dist_to_target < lookahead_distance) {
      current_idx++;
      if (current_idx >= path_.commands.size()) {
          // Reached the absolute end of the path
          geometry_msgs::msg::Twist twist;
          twist.linear.x = 0.0;
          twist.angular.z = 0.0;
          pub_cmd_vel_->publish(twist);
          return;
      }
      // Update targets to the new index
      x_target = path_.commands[current_idx].p.x;
      y_target = path_.commands[current_idx].p.y;
      yaw_target = path_.commands[current_idx].yaw;
      v_ref = path_.commands[current_idx].v.x;
      w_ref = path_.commands[current_idx].dyaw;
      
      // Recompute dx and dy for the new target
      dx = x_target - current_state_.pos.x;
      dy = y_target - current_state_.pos.y;
  }

  double heading_to_lookahead = std::atan2(dy, dx);
  double alpha = heading_to_lookahead - current_yaw;

  v_ref = std::min(v_ref, max_velocity_);
  w_ref = std::min(w_ref, max_angular_velocity_);

  double rho = hypot(dx, dy);
  double beta = yaw_target - current_yaw;

  // Apply control law (proportional with reduced gains to avoid oscillation)
  double K_rho_ = 1.0;    // Gain for distance to target (reduced from 2.5)
  double v_command = (v_ref - current_v) + K_rho_ * rho;

  v_command = std::clamp(v_command, (double)0.0, max_velocity_);

  // Angular velocity command (use signed alpha, not abs_alpha)
  double K_alpha_ = 0.3;  // Gain for heading error (reduced from 4.2)
  double K_beta = 0.0;   // Gain for final orientation error (reduced from 1.8)
  RCLCPP_INFO(this->get_logger(), "alpha: %.3f, beta: %.3f", alpha, beta);
  double w_command = w_ref + K_alpha_ * alpha + K_beta * beta;

  // Clamp angular velocity
  w_command = std::clamp(w_command, -max_angular_velocity_, max_angular_velocity_);

  // Publish cmd_vel
  geometry_msgs::msg::Twist twist;
  twist.linear.x = v_command;
  twist.angular.z = w_command;

  last_v_cmd_ = v_command;
  last_w_cmd_ = w_command;
  pub_cmd_vel_->publish(twist);

  // current_idx = current_idx + 1;
}

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<FFFB>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
