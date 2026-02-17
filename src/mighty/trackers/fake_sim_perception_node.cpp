#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp> 
#include <map>
#include <string>

class FakeSimPerceptionNode : public rclcpp::Node
{
public:
    FakeSimPerceptionNode() : Node("fake_sim_perception_node")
    {
        this->declare_parameter("num_humans", 10);
        this->declare_parameter("frame_id", "map");
        
        num_humans_ = this->get_parameter("num_humans").as_int();
        frame_id_ = this->get_parameter("frame_id").as_string();

        // Publisher: Sends detections to tracker
        det_pub_ = this->create_publisher<vision_msgs::msg::Detection3DArray>(
            "detected_objects", 10);

        marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "detected_objects_markers", 10);

        // Subscribers: Listen to every human's perfect position
        for (int i = 0; i < num_humans_; ++i) {
            std::string topic = "/human_" + std::to_string(i) + "/ground_truth";
            auto sub = this->create_subscription<nav_msgs::msg::Odometry>(
                topic, 10,
                [this, i](const nav_msgs::msg::Odometry::SharedPtr msg) {
                    this->odomCallback(msg, i);
                });
            subs_.push_back(sub);
        }

        // Publish aggregated detections at 10Hz (match LiDAR rate)
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&FakeSimPerceptionNode::publishDetections, this));
            
        RCLCPP_INFO(this->get_logger(), "Simulated Perception Bridge Started.");
    }

private:
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg, int id) {
        latest_poses_[id] = msg->pose.pose;
    }

    void publishDetections() {
        vision_msgs::msg::Detection3DArray output_msg;
        visualization_msgs::msg::MarkerArray marker_array;

        output_msg.header.stamp = this->now();
        output_msg.header.frame_id = frame_id_;
        
        // Define box dimensions (Must match URDF)
        double box_x = 0.5;
        double box_y = 0.5;
        double box_z = 1.7;

        int marker_id = 0;

        for (auto const& [id, pose] : latest_poses_) {
            
            // --- 1. Create Detection Message ---
            vision_msgs::msg::Detection3D detection;
            detection.header = output_msg.header;
            detection.bbox.center = pose;
            detection.bbox.size.x = box_x;
            detection.bbox.size.y = box_y;
            detection.bbox.size.z = box_z;
            output_msg.detections.push_back(detection);

            // --- 2. Create Visualization Marker ---
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = frame_id_;
            marker.header.stamp = this->now();
            marker.ns = "simulated_perception";
            marker.id = marker_id++;
            marker.type = visualization_msgs::msg::Marker::CUBE;
            marker.action = visualization_msgs::msg::Marker::ADD;

            marker.pose = pose;
            marker.scale.x = box_x;
            marker.scale.y = box_y;
            marker.scale.z = box_z;

            // Set Color (Green with transparency)
            marker.color.r = 0.0f;
            marker.color.g = 1.0f;
            marker.color.b = 0.0f;
            marker.color.a = 0.5f; // 50% transparent

            marker.lifetime = rclcpp::Duration::from_seconds(0.1); // Short lifetime so they disappear if not updated

            marker_array.markers.push_back(marker);
        }

        det_pub_->publish(output_msg);
        marker_pub_->publish(marker_array);
    }

    int num_humans_;
    std::string frame_id_;
    std::map<int, geometry_msgs::msg::Pose> latest_poses_;
    std::vector<rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr> subs_;
    rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr det_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_; 
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<FakeSimPerceptionNode>());
  rclcpp::shutdown();
  return 0;
}