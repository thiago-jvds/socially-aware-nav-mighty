#pragma once

#include <cmath>
#include <vector>

#include <dynus_interfaces/msg/state.hpp>
#include <dynus_interfaces/msg/vw_command_list.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>

#include <rclcpp/rclcpp.hpp>


class FFFB : public rclcpp::Node {
 public:
  /** @brief Construct the node, declare parameters, and set up publishers/subscribers. */
  FFFB();

 private:
  void pathCallback(const dynus_interfaces::msg::VwCommandList::SharedPtr msg);
  void stateCallback(const dynus_interfaces::msg::State::SharedPtr msg);
  void controlCallback();

  double wrapPi(double angle);

  // Publishers and Subscribers
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr pub_cmd_vel_;
  rclcpp::Subscription<dynus_interfaces::msg::VwCommandList>::SharedPtr sub_path_;
  rclcpp::Subscription<dynus_interfaces::msg::State>::SharedPtr sub_state_;
  rclcpp::TimerBase::SharedPtr timer_;

  // State
  dynus_interfaces::msg::State current_state_;
  dynus_interfaces::msg::VwCommandList path_;
  bool state_initialized_;
  bool path_initialized_;
  int current_idx = 0;

  // Parameters
  double max_velocity_;                 // Maximum commanded velocity (m/s)
  double max_angular_velocity_;         // Maximum angular velocity (rad/s)
  double control_rate_;                // Control loop rate (Hz)
  double last_v_cmd_ = 0.0;
  double last_w_cmd_ = 0.0;
};
