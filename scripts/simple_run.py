#!/usr/bin/env python3
# coding=utf-8
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import numpy as np
import os
import sys
import time

# 为简化版本，我们不使用pinocchio和casadi
# 而是直接使用一个预定义的轨迹

class SimpleIKPublisher(Node):
    def __init__(self):
        super().__init__('fairino_ik_publisher')
        
        # 创建关节状态发布器
        self.joint_publisher = self.create_publisher(JointState, 'joint_states', 10)
        
        # 创建定时器，每0.1秒调用一次回调函数
        self.timer = self.create_timer(0.1, self.timer_callback)
        
        # 关节名称列表 - 适配fairino5机器人的关节名称
        self.joint_names = ['j1', 'j2', 'j3', 'j4', 'j5', 'j6']
        
        # 轨迹参数
        self.trajectory_time = 0.0
        
        # 基本关节角度和振幅
        self.base_joint_angles = np.array([0.0, -1.57, 0.0, 0.0, -1.57, 0.0])
        self.joint_amplitudes = np.array([0.5, 0.3, 0.4, 0.5, 0.2, 0.3])
        
        self.get_logger().info('简化版Fairino5 IK发布器已初始化')
    
    def timer_callback(self):
        # 更新轨迹时间
        self.trajectory_time += 0.1
        
        # 创建简单的正弦波轨迹 - 每个关节都有不同的频率
        frequencies = np.array([0.5, 0.3, 0.7, 0.6, 0.4, 0.9])
        joint_positions = self.base_joint_angles + self.joint_amplitudes * np.sin(frequencies * self.trajectory_time)
        
        # 创建并发布关节状态消息
        joint_state_msg = JointState()
        joint_state_msg.header.stamp = self.get_clock().now().to_msg()
        joint_state_msg.name = self.joint_names
        joint_state_msg.position = joint_positions.tolist()
        
        self.joint_publisher.publish(joint_state_msg)
        self.get_logger().info(f'已发布关节状态: {joint_positions}')

def main(args=None):
    rclpy.init(args=args)
    
    node = SimpleIKPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('用户中断')
    finally:
        # 清理
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()