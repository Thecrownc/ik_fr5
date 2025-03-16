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

class Arm_IK:
    def __init__(self, urdf_dir):
        # 加载机器人模型
        print(f"加载机器人URDF: {urdf_dir}")
        self.robot = pin.RobotWrapper.BuildFromURDF(urdf_dir)
        
        # 打印关节限制，用于调试
        print("关节限制:")
        for i in range(self.robot.model.nq):
            joint_name = self.robot.model.names[i+1] if i < len(self.robot.model.names)-1 else f"Joint {i}"
            print(f"{joint_name}: 下限 = {self.robot.model.lowerPositionLimit[i]}, 上限 = {self.robot.model.upperPositionLimit[i]}")
        
        # 添加末端执行器框架
        ee_name = 'ee_frame'
        last_joint_id = self.robot.model.getJointId('j6')
        
        # 在最后一个关节的末端添加末端执行器框架 (通常工具的偏移量)
        self.robot.model.addFrame(
            pin.Frame(ee_name, last_joint_id, 
                     pin.SE3(np.eye(3), np.array([0, 0, 0.05]).reshape(3,1)),  # 末端偏移量
                     pin.FrameType.OP_FRAME)
        )
        
        # 获取并存储框架ID
        self.ee_frame_id = self.robot.model.getFrameId(ee_name)
        print(f"末端执行器框架 '{ee_name}' ID为: {self.ee_frame_id}")
        
        # 设置用户指定的初始配置
        # self.q_init = np.array([0.0, -0.795, 1.024, -1.554, -1.734, 0.0])
        self.q_init = np.array([0.0, -2.248, 2.247, -1.554, -1.734, 0.0])
        print(f"使用用户指定的初始配置: {self.q_init}")
        
        # 计算该配置下的末端执行器位置，用于轨迹参数设置
        data = self.robot.model.createData()
        pin.forwardKinematics(self.robot.model, data, self.q_init)
        pin.updateFramePlacements(self.robot.model, data)
        init_pose = data.oMf[self.ee_frame_id]
        print(f"初始配置下的末端执行器位置: {init_pose.translation}")
        print(f"初始配置下的末端执行器旋转矩阵:\n{init_pose.rotation}")
        
        # 使用CasADi创建优化问题
        self.cmodel = cpin.Model(self.robot.model)
        self.cdata = self.cmodel.createData()
        
        # 定义优化的符号变量
        self.q_sym = casadi.SX.sym("q", self.robot.model.nq)
        self.target_pose_sym = casadi.SX.sym("pose", 4, 4)
        
        # 用符号配置更新框架
        cpin.forwardKinematics(self.cmodel, self.cdata, self.q_sym)
        cpin.updateFramePlacements(self.cmodel, self.cdata)
        
        # 获取当前姿态和目标姿态
        current_pose = self.cdata.oMf[self.ee_frame_id]
        desired_pose = cpin.SE3(self.target_pose_sym)
        
        # 分离位置和方向误差
        position_error = current_pose.translation - desired_pose.translation
        
        # 对于方向，使用相对旋转的对数
        rel_rot = current_pose.rotation.T @ desired_pose.rotation
        orientation_error = cpin.log3(rel_rot)
        
        # 添加正则化项，使关节角度更接近初始配置
        initial_q = casadi.SX(self.q_init)
        regularization = 0.01 * casadi.sumsqr(self.q_sym - initial_q)
        
        # 目标函数：位置误差 + 方向误差 + 正则化
        position_cost = casadi.sumsqr(position_error)
        orientation_cost = 0.1 * casadi.sumsqr(orientation_error)  # 大幅减小方向权重
        objective = position_cost + orientation_cost + regularization
        
        # 设置NLP求解器，增加稳定性选项
        self.nlp = {'x': self.q_sym, 'f': objective, 'p': self.target_pose_sym}
        
        # IPOPT选项
        ipopt_options = {
            'ipopt.print_level': 0,
            'ipopt.sb': 'yes',
            'ipopt.max_iter': 200,           # 增加最大迭代次数
            'ipopt.tol': 1e-3,               # 放宽收敛容差
            'ipopt.acceptable_tol': 1e-2,    # 放宽可接受容差
            'ipopt.hessian_approximation': 'limited-memory',  # 使用BFGS近似Hessian矩阵
            'ipopt.mu_strategy': 'adaptive', # 自适应障碍参数更新
            'ipopt.check_derivatives_for_naninf': 'yes',
            'print_time': 0
        }
        
        self.solver = casadi.nlpsol('solver', 'ipopt', self.nlp, ipopt_options)
        
        print("IK求解器初始化成功")
    
    def solve_ik(self, target_pose, q_init=None):
        """为目标末端执行器姿态求解逆运动学"""
        # 使用提供的初始猜测或默认值
        if q_init is None:
            q_init = self.q_init
        
        # 定义关节角度的约束 (使用机器人模型中的限制)
        lbx = self.robot.model.lowerPositionLimit
        ubx = self.robot.model.upperPositionLimit
        
        # 确保约束有效
        for i in range(len(lbx)):
            if np.isnan(lbx[i]) or np.isinf(lbx[i]):
                lbx[i] = -3.14
            if np.isnan(ubx[i]) or np.isinf(ubx[i]):
                ubx[i] = 3.14
        
        # 求解优化问题
        try:
            sol = self.solver(x0=q_init, p=target_pose, lbx=lbx, ubx=ubx)
            q_sol = np.array(sol['x']).flatten()
            
            # 检查解是否包含NaN
            if np.any(np.isnan(q_sol)):
                print("警告: 解包含NaN值")
                return q_init, False
            
            # 检查解的质量
            position_error, orientation_error, _ = self.check_solution(q_sol, target_pose)
            
            # 如果位置误差太大，可能是优化停在了局部最优解
            if position_error > 0.05:  # 如果误差大于5厘米
                print(f"警告: 位置误差较大 ({position_error:.3f}m)")
                
                # 如果误差很大，返回失败
                if position_error > 0.1:  # 如果误差大于10厘米
                    return q_sol, False
            
            # 存储解作为下一次的初始猜测
            self.q_init = q_sol.copy()
            
            return q_sol, True
            
        except Exception as e:
            print(f"IK求解器错误: {e}")
            return q_init, False
    
    def check_solution(self, q, target_pose):
        """检查IK解的质量"""
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
        try:
            rel_rot = actual_pose.rotation.T @ target_rotation
            orientation_error = np.linalg.norm(pin.log3(rel_rot))
        except:
            orientation_error = np.pi
        
        return position_error, orientation_error, actual_pose

class FairinoIKPublisher(Node):
    def __init__(self, urdf_path):
        super().__init__('fairino_ik_publisher')
        
        # 创建IK求解器
        self.ik_solver = Arm_IK(urdf_path)
        
        # 创建关节状态发布器
        self.joint_publisher = self.create_publisher(JointState, 'joint_states', 10)
        
        # 创建定时器
        self.timer = self.create_timer(0.1, self.timer_callback)
        
        # 关节名称列表
        self.joint_names = ['j1', 'j2', 'j3', 'j4', 'j5', 'j6']
        
        # 轨迹参数
        self.trajectory_time = 0.0
        self.speed_factor = 1  # 较慢的速度
        
        # 使用初始配置下的末端执行器位置作为轨迹中心
        data = self.ik_solver.robot.model.createData()
        pin.forwardKinematics(self.ik_solver.robot.model, data, self.ik_solver.q_init)
        pin.updateFramePlacements(self.ik_solver.robot.model, data)
        init_pos = data.oMf[self.ik_solver.ee_frame_id].translation
        
        # 使用初始位置作为轨迹中心
        self.center_position = init_pos.copy()
        self.get_logger().info(f"轨迹中心位置: {self.center_position}")
        
        # 轨迹类型与参数
        self.trajectory_type = 0  # 默认为静态位置
        self.radius = 0.10  # 5厘米半径
        self.vertical_range = 0.10  # 上下运动范围
        
        # 设置自然的末端执行器方向 (使用初始配置下的方向)
        self.target_rotation = data.oMf[self.ik_solver.ee_frame_id].rotation
        
        # 创建初始齐次变换矩阵
        self.target_pose = np.eye(4)
        self.target_pose[:3, :3] = self.target_rotation
        self.target_pose[:3, 3] = self.center_position
        
        # 测试初始姿态是否可达
        init_test_q, init_success = self.ik_solver.solve_ik(self.target_pose)
        if init_success:
            pos_error, ori_error, _ = self.ik_solver.check_solution(init_test_q, self.target_pose)
            self.get_logger().info(f"初始姿态可达，位置误差: {pos_error:.3f}m")
            
            # 立即发布初始位置
            joint_state_msg = JointState()
            joint_state_msg.header.stamp = self.get_clock().now().to_msg()
            joint_state_msg.name = self.joint_names
            joint_state_msg.position = self.ik_solver.q_init.tolist()
            self.joint_publisher.publish(joint_state_msg)
            self.get_logger().info("已发布初始关节角度")
        else:
            self.get_logger().error("初始姿态不可达！请检查初始关节角度配置")
        
        # 记录最后一次成功的关节角度
        self.last_successful_joints = self.ik_solver.q_init.copy()
        
        # 记录连续失败计数
        self.consecutive_failures = 0
        self.reset_needed = False
        
        self.get_logger().info('Fairino5 IK发布器已初始化')
        
        # 询问用户选择轨迹类型
        self.ask_trajectory_type()
    
    def ask_trajectory_type(self):
        """询问用户选择轨迹类型"""
        self.get_logger().info("请选择轨迹类型:")
        self.get_logger().info("0: 静态位置 (固定姿态)")
        self.get_logger().info("1: 小圆形轨迹 (半径10厘米)")
        self.get_logger().info("2: 垂直运动 (上下10厘米)")
        self.get_logger().info("3: 水平矩形 (10厘米 x 5厘米)")
        self.get_logger().info("4: 8字")
        
        try:
            trajectory_type = int(input("请输入轨迹类型编号 (0-3): "))
            if 0 <= trajectory_type <= 4:
                self.trajectory_type = trajectory_type
                self.get_logger().info(f"已选择轨迹类型: {trajectory_type}")
            else:
                self.get_logger().warn("无效的轨迹类型，将使用默认的静态位置")
        except:
            self.get_logger().warn("输入错误，将使用默认的静态位置")
    
    def update_target_pose(self):
        """基于当前轨迹类型和时间更新目标位姿"""
        
        # 如果需要重置，回到中心位置
        if self.reset_needed:
            self.target_pose[:3, 3] = self.center_position
            self.reset_needed = False
            return
        
        # 静态位置
        if self.trajectory_type == 0:
            return  # 保持目标位姿不变
        
        t = self.trajectory_time * self.speed_factor
        
        # 小圆形轨迹
        if self.trajectory_type == 1:
            x = self.center_position[0] + self.radius * np.cos(t)
            y = self.center_position[1] + self.radius * np.sin(t)
            z = self.center_position[2]
        
        # 垂直运动
        elif self.trajectory_type == 2:
            x = self.center_position[0]
            y = self.center_position[1]
            z = self.center_position[2] + self.vertical_range * np.sin(t)
        elif self.trajectory_type == 4:  # 8字形轨迹
            # 参数方程：x = r*sin(2t), y = r*sin(t)
            x = self.center_position[0] + 0.1 * np.sin(2 * t)
            y = self.center_position[1] + 0.1 * np.sin(t)
            z = self.center_position[2]
        # 水平矩形
        elif self.trajectory_type == 3:
            period = 2 * np.pi
            phase = (t % period) / period  # 0 到 1 的相对位置
            
            rect_width = 0.10  # 10厘米宽
            rect_height = 0.05  # 5厘米高
            
            if phase < 0.25:  # 第一条边
                progress = phase * 4
                x = self.center_position[0] + rect_width * (progress - 0.5)
                y = self.center_position[1] - rect_height / 2
            elif phase < 0.5:  # 第二条边
                progress = (phase - 0.25) * 4
                x = self.center_position[0] + rect_width / 2
                y = self.center_position[1] + rect_height * (progress - 0.5)
            elif phase < 0.75:  # 第三条边
                progress = (phase - 0.5) * 4
                x = self.center_position[0] + rect_width * (0.5 - progress)
                y = self.center_position[1] + rect_height / 2
            else:  # 第四条边
                progress = (phase - 0.75) * 4
                x = self.center_position[0] - rect_width / 2
                y = self.center_position[1] + rect_height * (0.5 - progress)
            
            z = self.center_position[2]
        
        # 更新目标位置
        self.target_pose[:3, 3] = [x, y, z]
    
    def timer_callback(self):
        # 更新轨迹时间
        self.trajectory_time += 0.1
        
        # 更新目标位姿
        self.update_target_pose()
        
        # 求解IK
        q_sol, success = self.ik_solver.solve_ik(self.target_pose, self.last_successful_joints)
        
        if success:
            # 检查解的质量
            pos_error, ori_error, actual_pose = self.ik_solver.check_solution(q_sol, self.target_pose)
            
            if pos_error < 0.05:  # 位置误差小于5厘米
                # 重置失败计数
                self.consecutive_failures = 0
                
                # 存储成功的关节角度
                self.last_successful_joints = q_sol.copy()
                
                # 创建并发布关节状态消息
                joint_state_msg = JointState()
                joint_state_msg.header.stamp = self.get_clock().now().to_msg()
                joint_state_msg.name = self.joint_names
                joint_state_msg.position = q_sol.tolist()
                
                self.joint_publisher.publish(joint_state_msg)
                
                # 计算实际末端位置与目标位置
                target_pos = self.target_pose[:3, 3]
                actual_pos = actual_pose.translation
                
                self.get_logger().info(f'目标: {target_pos}, 实际: {actual_pos}, 误差: {pos_error:.3f}m')
            else:
                self.get_logger().warn(f'解有效但误差较大: {pos_error:.3f}m')
                self.consecutive_failures += 1
        else:
            self.get_logger().warn('无法找到IK解')
            self.consecutive_failures += 1
            
        # 如果连续失败太多次，重置轨迹
        if self.consecutive_failures > 5:
            self.get_logger().warn('连续多次求解失败，重置轨迹')
            self.reset_needed = True
            self.consecutive_failures = 0

def main(args=None):
    rclpy.init(args=args)
    
    # URDF文件路径 - 更新为正确的路径
    urdf_path = "/home/jhw/sim_ws/src/robot_description/urdf/qingloong/fairino5_v6.urdf"
    
    # 检查文件是否存在
    if not os.path.exists(urdf_path):
        print(f"错误: URDF文件不存在: {urdf_path}")
        print("请确保文件路径正确，或提供绝对路径。")
        return
    
    try:
        node = FairinoIKPublisher(urdf_path)
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('用户中断')
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        # 清理
        rclpy.shutdown()

if __name__ == '__main__':
    main()