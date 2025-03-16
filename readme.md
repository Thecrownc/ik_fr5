
# Build
~~~python
sudo apt install ros-$ROS_DISTRO-joint-state-publisher-gui ros-$ROS_DISTRO-xacro

mkdir -p sim_ws/src && cd sim_ws/src
git clone https://github.com/linClubs/robot_description.git

cd ..
colcon build --packages-select robot_description
~~~

# Run

~~~python
# rviz
source install/setup.bash
ros2 launch robot_description qingloong_view.launch.py

# ik
ros2 run robot_description qingloong_ik.py
~~~

