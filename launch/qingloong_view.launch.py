import os
import launch
from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
	# 1. 启动robot_state_publisher,该节点以参数的方式加载urdf文件内容
	pkg_path = get_package_share_directory("robot_description")
	urdf_path=os.path.join(pkg_path, "urdf/qingloong/fairino5_v6.urdf")
	# xacro后带空格， ParameterValue作用是在终端解析urdf_path文件
	urdf_value = ParameterValue(Command(["xacro ", urdf_path]))
	robot_state_pub = Node(
		package="robot_state_publisher",
		executable="robot_state_publisher",
		parameters=[{'robot_description': urdf_value}]
	)

	# 2 gui界面的joint-state
	joint_state_publisher_node = Node(
		package="joint_state_publisher_gui",
		executable="joint_state_publisher_gui",
	)

	# 3. 启robot_state_pub动rviz2节点
	rviz2_path = os.path.join(pkg_path, "rviz/qingloong.rviz") 
	# 第一次没有rviz配置文件就注释掉arguments=['-d', rviz2_path],
	rviz2 = Node(
		package="rviz2",
		executable="rviz2",
		arguments=['-d', rviz2_path],
		parameters=[{'use_sim_time': True}] # 启用仿真时间
	)

	return LaunchDescription([joint_state_publisher_node, robot_state_pub, rviz2])