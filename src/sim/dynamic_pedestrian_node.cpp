#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <gazebo_msgs/msg/model_state.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <nlohmann/json.hpp>
#include <sim/exprtk.hpp>

#include <string>
#include <vector>
#include <stdexcept>
#include <cmath>

using json = nlohmann::json;

struct PedestrianSpec
{
    std::string model_name;
    std::string traj_x;
    std::string traj_y;
    std::string traj_z;

    double t_var{0.0};

    // exprtk plumbing
    exprtk::symbol_table<double> symbol_table;
    exprtk::expression<double> expr_x;
    exprtk::expression<double> expr_y;
    exprtk::expression<double> expr_z;
    bool compiled{false};

    // State memory for basic orientation calculation
    double last_x{0.0};
    double last_y{0.0};

    void compile()
    {
        symbol_table.clear();
        symbol_table.add_variable("t", t_var);
        symbol_table.add_constants();

        expr_x.register_symbol_table(symbol_table);
        expr_y.register_symbol_table(symbol_table);
        expr_z.register_symbol_table(symbol_table);

        exprtk::parser<double> parser;

        auto compile_one = [&](const std::string &label, const std::string &src, exprtk::expression<double> &expr)
        {
            if (!parser.compile(src, expr))
            {
                std::ostringstream oss;
                oss << "Failed to compile " << label << "='" << src << "'";
                throw std::runtime_error(oss.str());
            }
        };

        compile_one("traj_x", traj_x, expr_x);
        compile_one("traj_y", traj_y, expr_y);
        compile_one("traj_z", traj_z, expr_z);

        compiled = true;
    }

    inline void evaluate(double t_now, double &x, double &y, double &z)
    {
        if (!compiled)
        {
            x = y = z = std::numeric_limits<double>::quiet_NaN();
            return;
        }
        t_var = t_now;
        x = expr_x.value();
        y = expr_y.value();
        z = expr_z.value();
    }
};

class DynamicPedestrianNode : public rclcpp::Node
{
public:
    DynamicPedestrianNode() : Node("dynamic_pedestrian_node")
    {
        // Parameters
        publish_rate_hz_ = declare_parameter<double>("publish_rate_hz", 50.0);
        std::string pedestrians_json_str = declare_parameter<std::string>("pedestrians_json", "[]");
        global_frame_ = declare_parameter<std::string>("global_frame", "map");
        base_link_name_ = declare_parameter<std::string>("base_link_name", "base_link");

        // Parse JSON
        try
        {
            json J = json::parse(pedestrians_json_str);
            pedestrians_.reserve(J.size());
            for (auto &item : J)
            {
                PedestrianSpec p;
                p.model_name = item.value("name", "");
                p.traj_x = item.value("traj_x", "0.0");
                p.traj_y = item.value("traj_y", "0.0");
                p.traj_z = item.value("traj_z", "0.0");
                
                // Initialize last positions to avoid jumpy orientation at t=0
                p.last_x = item.value("x0", 0.0);
                p.last_y = item.value("y0", 0.0);
                
                // Create individual Odometry publishers for the perception node
                std::string odom_topic = "/" + p.model_name + "/ground_truth";
                odom_pubs_.push_back(create_publisher<nav_msgs::msg::Odometry>(odom_topic, 10));

                pedestrians_.push_back(std::move(p));
            }
            for (auto &p : pedestrians_) {
                p.compile();
            }
            RCLCPP_INFO(get_logger(), "Loaded & compiled %zu pedestrians from JSON.", pedestrians_.size());
        }
        catch (const std::exception &e)
        {
            RCLCPP_ERROR(get_logger(), "Failed to parse/compile pedestrians_json: %s", e.what());
        }

        // Setup Main Publishers
        model_state_pub_ = create_publisher<gazebo_msgs::msg::ModelState>("/gazebo/set_model_state", 50);
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        // Timer
        double period = (publish_rate_hz_ > 0.0) ? 1.0 / publish_rate_hz_ : 0.02;
        timer_ = create_wall_timer(
            std::chrono::duration<double>(period),
            std::bind(&DynamicPedestrianNode::timerCB, this));
    }

private:
    void timerCB()
    {
        auto stamp = now();
        // Exact same time calculation as the forest node
        double t_now = static_cast<double>(stamp.nanoseconds()) * 1e-9;

        int idx = 0;
        for (auto &p : pedestrians_)
        {
            // 1. Evaluate the math equations (identical to o.evaluate in the forest node)
            p.t_var = t_now;
            double x = p.expr_x.value();
            double y = p.expr_y.value();
            double z = p.expr_z.value();

            // 2. Publish Odometry for the fake_sim_perception_node (Replaces DynTraj)
            nav_msgs::msg::Odometry odom;
            odom.header.stamp = stamp;
            odom.header.frame_id = global_frame_;
            odom.child_frame_id = p.model_name + "/" + base_link_name_;
            odom.pose.pose.position.x = x;
            odom.pose.pose.position.y = y;
            odom.pose.pose.position.z = z;
            odom.pose.pose.orientation.w = 1.0; 
            
            if (idx < odom_pubs_.size() && odom_pubs_[idx]) {
                odom_pubs_[idx]->publish(odom);
            }

            // 3. Publish TF (Identical to the forest node)
            if (publish_tf_)
            {
                geometry_msgs::msg::TransformStamped tf;
                tf.header.stamp = stamp;
                tf.header.frame_id = global_frame_;
                tf.child_frame_id = p.model_name + "/" + base_link_name_;
                tf.transform.translation.x = x;
                tf.transform.translation.y = y;
                tf.transform.translation.z = z;
                tf.transform.rotation.w = 1.0;
                tf_broadcaster_->sendTransform(tf);
            }

            // 4. Move Gazebo Models (Identical to the forest node)
            if (move_models_)
            {
                gazebo_msgs::msg::ModelState ms;
                ms.model_name = p.model_name;
                ms.pose.position.x = x;
                ms.pose.position.y = y;
                ms.pose.position.z = z;
                ms.pose.orientation.w = 1.0;
                model_state_pub_->publish(ms);
            }

            p.last_x = x;
            p.last_y = y;

            ++idx;
        }
    }

    double publish_rate_hz_{50.0};
    std::string global_frame_;
    std::string base_link_name_;
    bool move_models_{false};
    bool publish_tf_{true};
    
    std::vector<PedestrianSpec> pedestrians_;

    rclcpp::Publisher<gazebo_msgs::msg::ModelState>::SharedPtr model_state_pub_;
    std::vector<rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr> odom_pubs_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DynamicPedestrianNode>());
    rclcpp::shutdown();
    return 0;
}