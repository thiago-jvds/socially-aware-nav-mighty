#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <gazebo_msgs/msg/model_states.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <builtin_interfaces/msg/time.hpp>
#include <map>
#include <mutex>
#include <vector>
#include <string>

struct PoseWithStamp {
    geometry_msgs::msg::Pose pose;
    builtin_interfaces::msg::Time stamp;
};

class FakeSimPerceptionNode : public rclcpp::Node
{
public:
    FakeSimPerceptionNode() : Node("fake_sim_perception_node")
    {
        this->declare_parameter("num_humans", 10);
        this->declare_parameter("frame_id", "map");
        this->declare_parameter("truth_topic_prefix", "/human_");
        this->declare_parameter("truth_topic_suffix", "/ground_truth");
        this->declare_parameter("truth_source", "odom");
        this->declare_parameter("gazebo_model_states_topic", "/gazebo/model_states");
        this->declare_parameter("human_name_prefix", "human_");
        this->declare_parameter("detection_rate_hz", 10.0);
        
        num_humans_ = this->get_parameter("num_humans").as_int();
        frame_id_ = this->get_parameter("frame_id").as_string();
        truth_topic_prefix_ = this->get_parameter("truth_topic_prefix").as_string();
        truth_topic_suffix_ = this->get_parameter("truth_topic_suffix").as_string();
        truth_source_ = this->get_parameter("truth_source").as_string();
        gazebo_model_states_topic_ = this->get_parameter("gazebo_model_states_topic").as_string();
        human_name_prefix_ = this->get_parameter("human_name_prefix").as_string();
        detection_rate_hz_ = this->get_parameter("detection_rate_hz").as_double();

        // Publisher: Sends detections to tracker
        det_pub_ = this->create_publisher<vision_msgs::msg::Detection3DArray>(
            "detected_objects", 10);

        const bool use_gazebo = (truth_source_ == "gazebo_model_states" || truth_source_ == "auto");
        const bool use_odom = (truth_source_ == "odom" || truth_source_ == "auto");

        if (use_gazebo) {
            model_states_sub_ = this->create_subscription<gazebo_msgs::msg::ModelStates>(
                gazebo_model_states_topic_, 10,
                std::bind(&FakeSimPerceptionNode::modelStatesCallback, this, std::placeholders::_1));
            RCLCPP_INFO(this->get_logger(), "Enabled Gazebo model states source: %s", gazebo_model_states_topic_.c_str());
        }

        if (use_odom) {
            // Subscribe to each human's truth odometry topic.
            for (int i = 0; i < num_humans_; ++i) {
                const std::string topic = truth_topic_prefix_ + std::to_string(i) + truth_topic_suffix_;
                auto sub = this->create_subscription<nav_msgs::msg::Odometry>(
                    topic, 10,
                    [this, i](const nav_msgs::msg::Odometry::SharedPtr msg) {
                        this->odomCallback(msg, i);
                    });
                subs_.push_back(sub);
            }
            RCLCPP_INFO(this->get_logger(), "Enabled odometry truth source with prefix: %s", truth_topic_prefix_.c_str());
        }

        if (!use_gazebo && !use_odom) {
            RCLCPP_WARN(this->get_logger(),
                        "Unknown truth_source='%s'. Falling back to odom.",
                        truth_source_.c_str());
            for (int i = 0; i < num_humans_; ++i) {
                const std::string topic = truth_topic_prefix_ + std::to_string(i) + truth_topic_suffix_;
                auto sub = this->create_subscription<nav_msgs::msg::Odometry>(
                    topic, 10,
                    [this, i](const nav_msgs::msg::Odometry::SharedPtr msg) {
                        this->odomCallback(msg, i);
                    });
                subs_.push_back(sub);
            }
        }

        // Publish aggregated detections at configured rate.
        const double clamped_rate = detection_rate_hz_ > 0.0 ? detection_rate_hz_ : 10.0;
        timer_ = this->create_wall_timer(
            std::chrono::duration<double>(1.0 / clamped_rate),
            std::bind(&FakeSimPerceptionNode::publishDetections, this));
            
        RCLCPP_INFO(this->get_logger(), "Simulated Perception Bridge Started.");
    }

private:
    int parseHumanId(const std::string& name) const {
        const std::size_t pos = name.rfind(human_name_prefix_);
        if (pos == std::string::npos) {
            return -1;
        }
        const std::string suffix = name.substr(pos + human_name_prefix_.size());
        if (suffix.empty()) {
            return -1;
        }
        for (char c : suffix) {
            if (c < '0' || c > '9') {
                return -1;
            }
        }
        try {
            return std::stoi(suffix);
        } catch (...) {
            return -1;
        }
    }

    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg, int id) {
        std::lock_guard<std::mutex> lock(poses_mutex_);
        latest_poses_[id] = {msg->pose.pose, msg->header.stamp};
    }

    void modelStatesCallback(const gazebo_msgs::msg::ModelStates::SharedPtr msg) {
        const auto stamp = this->now();

        std::lock_guard<std::mutex> lock(poses_mutex_);
        for (size_t i = 0; i < msg->name.size() && i < msg->pose.size(); ++i) {
            const int id = parseHumanId(msg->name[i]);
            if (id < 0) {
                continue;
            }
            if (num_humans_ > 0 && id >= num_humans_) {
                continue;
            }
            latest_poses_[id] = {msg->pose[i], stamp};
        }
    }

    void publishDetections() {
        vision_msgs::msg::Detection3DArray output_msg;

        output_msg.header.frame_id = frame_id_;
        
        // Define box dimensions (Must match URDF)
        double box_x = 0.5;
        double box_y = 0.5;
        double box_z = 1.7;

        std::map<int, PoseWithStamp> poses_copy;
        {
            std::lock_guard<std::mutex> lock(poses_mutex_);
            poses_copy = latest_poses_;
        }

        if (poses_copy.empty()) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(), *this->get_clock(), 3000,
                "No truth poses received yet. Check truth_source, topic names, and human_name_prefix.");
        }

        for (const auto& [id, pose_stamp] : poses_copy) {
            // --- 1. Create Detection Message ---
            vision_msgs::msg::Detection3D detection;
            detection.header.stamp = pose_stamp.stamp;
            detection.header.frame_id = frame_id_;
            detection.bbox.center = pose_stamp.pose;
            detection.bbox.size.x = box_x;
            detection.bbox.size.y = box_y;
            detection.bbox.size.z = box_z;
            output_msg.detections.push_back(detection);
        }

        // Use the most recent detection stamp for the array header, fallback to now() if empty
        if (!output_msg.detections.empty()) {
            output_msg.header.stamp = output_msg.detections.back().header.stamp;
        } else {
            output_msg.header.stamp = this->now();
        }

        det_pub_->publish(output_msg);
    }

    int num_humans_;
    std::string frame_id_;
    std::string truth_topic_prefix_;
    std::string truth_topic_suffix_;
    std::string truth_source_;
    std::string gazebo_model_states_topic_;
    std::string human_name_prefix_;
    double detection_rate_hz_{10.0};
    std::map<int, PoseWithStamp> latest_poses_;
    std::mutex poses_mutex_;
    std::vector<rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr> subs_;
    rclcpp::Subscription<gazebo_msgs::msg::ModelStates>::SharedPtr model_states_sub_;
    rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr det_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<FakeSimPerceptionNode>());
  rclcpp::shutdown();
  return 0;
}