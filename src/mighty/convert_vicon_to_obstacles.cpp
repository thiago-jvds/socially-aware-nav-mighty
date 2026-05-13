#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp> 
#include <map>
#include <string>
#include <vector>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

class ConvertViconToObstacleNode : public rclcpp::Node
{
public:
    ConvertViconToObstacleNode() : Node("convert_vicon_to_obstacle_node")
    {

        this->declare_parameter("frame_id", "ST01/map");
        frame_id_ = this->get_parameter("frame_id").as_string();

        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        // Publisher: Sends detections to tracker
        det_pub_ = this->create_publisher<vision_msgs::msg::Detection3DArray>(
            "detected_objects", 10);

        // Specific topics to subscribe to
        this->declare_parameter<std::vector<std::string>>(
            "target_obstacles", 
            {"/Lucas6", "/HELMET3"}
        );
        std::vector<std::string> target_topics = this->get_parameter("target_obstacles").as_string_array();

        // Create subscriptions for the specific topics
        for (const auto& topic : target_topics) {
            auto topic_world = topic + "/world";
            auto sub = this->create_subscription<geometry_msgs::msg::PoseStamped>(
                topic_world, 10,
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
            std::bind(&ConvertViconToObstacleNode::publishDetections, this));
            
        RCLCPP_INFO(this->get_logger(), "Convert Vicon to Obstacle Node Started.");
    }

private:
    // Changed id to string to match the topic names
    void odomCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg, const std::string& id) {
        latest_poses_[id] = *msg;
    }

    void publishDetections() {
        vision_msgs::msg::Detection3DArray output_msg;

        output_msg.header.stamp = this->now();
        output_msg.header.frame_id = frame_id_;
        
        double box_x = 0.5;
        double box_y = 0.5;
        double box_z = 1.7;

        int marker_id = 0;

        for (auto const& [id, pose_stamped_msg] : latest_poses_) {
            geometry_msgs::msg::PoseStamped transformed_pose_msg;
            
            try {
                geometry_msgs::msg::PoseStamped pose_to_transform = pose_stamped_msg;
                // Override the timestamp to 0. This guarantees we get the latest available 
                // transform and avoids slight extrapolation errors between topic rates and TF rates.
                pose_to_transform.header.stamp.sec = 0;
                pose_to_transform.header.stamp.nanosec = 0;

                // 4. Transform the pose into the target frame_id_ ("map")
                transformed_pose_msg = tf_buffer_->transform(pose_to_transform, frame_id_);
            } catch (const tf2::TransformException & ex) {
                RCLCPP_WARN_THROTTLE(
                    this->get_logger(), *this->get_clock(), 1000, 
                    "Could not transform %s to %s: %s", 
                    pose_stamped_msg.header.frame_id.c_str(), frame_id_.c_str(), ex.what()
                );
                continue; // Skip this object if TF isn't ready
            }

            vision_msgs::msg::Detection3D detection;
            detection.header = output_msg.header;
            detection.bbox.center = transformed_pose_msg.pose; // Use transformed pose
            detection.bbox.center.position.z = 0.0;
            detection.bbox.size.x = box_x;
            detection.bbox.size.y = box_y;
            detection.bbox.size.z = box_z;
            output_msg.detections.push_back(detection);
        }

        det_pub_->publish(output_msg);
    }

    std::string frame_id_;
    // Map key changed to string to accommodate named topics
    std::map<std::string, geometry_msgs::msg::PoseStamped> latest_poses_;
    std::vector<rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr> subs_;
    rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr det_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ConvertViconToObstacleNode>());
  rclcpp::shutdown();
  return 0;
}