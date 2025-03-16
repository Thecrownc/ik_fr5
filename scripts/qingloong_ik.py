#!/usr/bin/env python3
# coding=utf-8
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin
import casadi
import time
import os
import sys
np.set_printoptions(precision=5, suppress=True, linewidth=200)

class Arm_IK:
    def __init__(self, urdf_dir):
        # 加载机器人模型
        print(f"加载机器人URDF: {urdf_dir}")
        self.robot = pin.RobotWrapper.BuildFromURDF(urdf_dir)
        
        # Fairino5机器人有6个关节，所有关节都用于IK
        print(f"机器人模型加载完成. 自由度: {self.robot.model.nq}")
        
        # 添加末端执行器框架（如果需要）- 从最后一个关节偏移一小段距离
        ee_name = 'ee_frame'
        last_joint_id = self.robot.model.getJointId('j6')
        
        # 在最后一个关节的末端添加末端执行器框架
        self.robot.model.addFrame(
            pin.Frame(ee_name, last_joint_id, 
                     pin.SE3(np.eye(3), np.array([0, 0, 0.1]).reshape(3,1)),
                     pin.FrameType.OP_FRAME)
        )
        
        # 获取并存储框架ID
        self.ee_frame_id = self.robot.model.getFrameId(ee_name)
        print(f"末端执行器框架 '{ee_name}' ID为: {self.ee_frame_id}")
        
        # 设置初始配置为零
        self.q_init = np.zeros(self.robot.model.nq)
        
        # 使用CasADi创建优化问题
        # 创建CasADi模型
        self.cmodel = cpin.Model(self.robot.model)
        self.cdata = self.cmodel.createData()
        
        # 定义优化的符号变量
        self.q_sym = casadi.SX.sym("q", self.robot.model.nq)
        self.target_pose_sym = casadi.SX.sym("pose", 4, 4)
        
        # 用符号配置更新框架
        cpin.forwardKinematics(self.cmodel, self.cdata, self.q_sym)
        cpin.updateFramePlacements(self.cmodel, self.cdata)
        
        # 使用SE3 log映射定义误差函数
        current_pose = self.cdata.oMf[self.ee_frame_id]
        desired_pose = cpin.SE3(self.target_pose_sym)
        
        position_error = current_pose.translation - desired_pose.translation
        # 对于方向，我们使用相对旋转的对数
        rel_rot = current_pose.rotation.T @ desired_pose.rotation
        orientation_error = cpin.log3(rel_rot)
        
        # 组合位置和方向误差
        error = casadi.vertcat(position_error, orientation_error)
        
        # 定义目标函数（平方误差）
        objective = casadi.dot(error, error)
        
        # 设置NLP求解器
        self.nlp = {'x': self.q_sym, 'f': objective, 'p': self.target_pose_sym}
        self.solver = casadi.nlpsol('solver', 'ipopt', self.nlp, {
            'ipopt.print_level': 0,
            'ipopt.sb': 'yes',
            'print_time': 0
        })
        
        print("IK求解器初始化成功")
    
    def solve_ik(self, target_pose, q_init=None):
        """
        为目标末端执行器姿态求解逆运动学
        
        参数:
        -----------
        target_pose : numpy.ndarray (4x4)
            目标末端执行器姿态的齐次变换矩阵
        q_init : numpy.ndarray, 可选
            关节角度的初始猜测，默认为零配置
            
        返回:
        --------
        q_sol : numpy.ndarray
            解出的关节角度
        success : bool
            优化是否收敛
        """
        # 使用提供的初始猜测或默认值
        if q_init is None:
            q_init = self.q_init
        
        # 定义关节角度的约束
        lbx = self.robot.model.lowerPositionLimit
        ubx = self.robot.model.upperPositionLimit
        
        # 确保约束有效（没有NaN或无穷）
        for i in range(len(lbx)):
            if np.isnan(lbx[i]) or np.isinf(lbx[i]):
                lbx[i] = -10.0
            if np.isnan(ubx[i]) or np.isinf(ubx[i]):
                ubx[i] = 10.0
        
        # 求解优化问题
        try:
            sol = self.solver(x0=q_init, p=target_pose, lbx=lbx, ubx=ubx)
            q_sol = np.array(sol['x']).flatten()
            
            # 存储解作为下一次的初始猜测
            self.q_init = q_sol.copy()
            
            # 检查解是否有效
            residual = sol['f']
            success = residual < 1e-4
            
            return q_sol, success
            
        except Exception as e:
            print(f"IK求解器错误: {e}")
            return q_init, False
    
    def check_solution(self, q, target_pose):
        """
        通过计算前向运动学并计算误差来检查解
        
        参数:
        -----------
        q : numpy.ndarray
            要检查的关节角度
        target_pose : numpy.ndarray (4x4)
            目标末端执行器姿态
            
        返回:
        --------
        position_error : float
            实际位置和目标位置之间的距离
        orientation_error : float
            实际方向和目标方向之间的角度差异
        """
        # 创建用于运动学计算的数据结构
        data = self.robot.model.createData()
        
        # 计算前向运动学
        pin.forwardKinematics(self.robot.model, data, q)
        pin.updateFramePlacements(self.robot.model, data)
        
        # 获取结果末端执行器姿态
        actual_pose = data.oMf[self.ee_frame_id]
        
        # 提取目标位置和方向
        target_position = target_pose[:3, 3]
        target_rotation = target_pose[:3, :3]
        
        # 计算位置误差（欧几里德距离）
        position_error = np.linalg.norm(actual_pose.translation - target_position)
        
        # 计算方向误差（旋转之间的角度）
        rel_rot = actual_pose.rotation.T @ target_rotation
        orientation_error = np.linalg.norm(pin.log3(rel_rot))
        
        return position_error, orientation_error, actual_pose

class FairinoIKPublisher(Node):
    def __init__(self, urdf_path):
        super().__init__('fairino_ik_publisher')
        
        # 创建IK求解器
        self.ik_solver = Arm_IK(urdf_path)
        
        # 创建关节状态发布器
        self.joint_publisher = self.create_publisher(JointState, 'joint_states', 10)
        
        # 创建定时器，每0.1秒调用一次回调函数
        self.timer = self.create_timer(0.1, self.timer_callback)
        
        # 关节名称列表
        self.joint_names = ['j1', 'j2', 'j3', 'j4', 'j5', 'j6']
        
        # 轨迹参数
        self.trajectory_time = 0.0
        
        # 初始目标位姿
        self.target_position = np.array([0.6, 0.0, 0.5])
        self.target_rotation = np.array([
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0]
        ])
        
        # 创建齐次变换矩阵
        self.target_pose = np.eye(4)
        self.target_pose[:3, :3] = self.target_rotation
        self.target_pose[:3, 3] = self.target_position
        
        self.get_logger().info('Fairino5 IK发布器已初始化')
    
    def timer_callback(self):
        # 更新轨迹时间
        self.trajectory_time += 0.1
        
        # 更新目标位置（例如，沿z轴做正弦运动）
        z = 0.5 + 0.2 * np.sin(self.trajectory_time)
        self.target_pose[2, 3] = z
        
        # 求解IK
        q_sol, success = self.ik_solver.solve_ik(self.target_pose)
        
        if success:
            # 检查解的质量
            pos_error, ori_error, _ = self.ik_solver.check_solution(q_sol, self.target_pose)
            self.get_logger().info(f'位置误差: {pos_error:.6f} m, 方向误差: {ori_error:.6f} rad')
            
            # 创建并发布关节状态消息
            joint_state_msg = JointState()
            joint_state_msg.header.stamp = self.get_clock().now().to_msg()
            joint_state_msg.name = self.joint_names
            joint_state_msg.position = q_sol.tolist()
            
            self.joint_publisher.publish(joint_state_msg)
            self.get_logger().info(f'已发布关节状态: {q_sol}')
        else:
            self.get_logger().warn('无法找到IK解')

def main(args=None):
    rclpy.init(args=args)
    
    # URDF文件路径 - 更新为正确的路径
    urdf_path = "/home/jhw/sim_ws/src/robot_description/urdf/qingloong/fairino5_v6.urdf"
    
    node = FairinoIKPublisher(urdf_path)
    
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