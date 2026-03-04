#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp> 
#include <map>
#include <string>
#include <vector>

class BagPerceptionNode : public rclcpp::Node
{
public:
    BagPerceptionNode() : Node("bag_perception_node")
    {
        this->declare_parameter("frame_id", "map");
        frame_id_ = this->get_parameter("frame_id").as_string();

        // Specific topics to subscribe to
        this->declare_parameter<std::vector<std::string>>(
            "target_topics", 
            {"/Lucas6/world", "/HELMET3/world"}
        );

        std::vector<std::string> target_topics = this->get_parameter("target_topics").as_string_array();

        // Publisher: Sends detections to tracker
        det_pub_ = this->create_publisher<vision_msgs::msg::Detection3DArray>(
            "detected_objects", 10);

        marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "detected_objects_markers", 10);

        // Create subscriptions for the specific topics
        for (const auto& topic : target_topics) {
            auto sub = this->create_subscription<geometry_msgs::msg::PoseStamped>(
                topic, 10,
                [this, topic](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
                    // Use the topic string as the key in our map
                    this->odomCallback(msg, topic);
                });
            subs_.push_back(sub);
            RCLCPP_INFO(this->get_logger(), "Subscribed to: %s", topic.c_str());
        }

        // Publish aggregated detections at 10Hz
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&BagPerceptionNode::publishDetections, this));
            
        RCLCPP_INFO(this->get_logger(), "Simulated Perception Bridge Started.");
    }

private:
    // Changed id to string to match the topic names
    void odomCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg, const std::string& id) {
        latest_poses_[id] = msg->pose;
    }

    void publishDetections() {
        vision_msgs::msg::Detection3DArray output_msg;
        visualization_msgs::msg::MarkerArray marker_array;

        output_msg.header.stamp = this->now();
        output_msg.header.frame_id = frame_id_;
        
        double box_x = 0.5;
        double box_y = 0.5;
        double box_z = 1.7;

        int marker_id = 0;

        for (auto const& [id, pose] : latest_poses_) {
            // --- 1. Create Detection Message ---
            vision_msgs::msg::Detection3D detection;
            detection.header = output_msg.header;
            detection.bbox.center = pose;
            detection.bbox.center.position.z = 0.0;
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

            marker.color.r = 0.0f;
            marker.color.g = 1.0f;
            marker.color.b = 0.0f;
            marker.color.a = 0.5f; 

            marker.lifetime = rclcpp::Duration::from_seconds(0.1);
            marker_array.markers.push_back(marker);
        }

        det_pub_->publish(output_msg);
        marker_pub_->publish(marker_array);
    }

    std::string frame_id_;
    // Map key changed to string to accommodate named topics
    std::map<std::string, geometry_msgs::msg::Pose> latest_poses_;
    std::vector<rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr> subs_;
    rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr det_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_; 
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<BagPerceptionNode>());
  rclcpp::shutdown();
  return 0;
}