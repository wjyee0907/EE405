import time
import cv2
import rclpy
import numpy as np
import jax.numpy as jnp

from .kinematics import *
from .utils.kinematics_utils import *
from .utils.grasping_base import GraspingNodeBase
from .marker_detector import MarkerDetectionResult


class GraspingNode(GraspingNodeBase):
    def __init__(self, name):
        super().__init__(name)
        self.running = True
        
        self.last_q = None
        self.grasping = False
        self.grasp_targets = []

        #####
        ## TODO : Define custom class variables, if needed.
        self.gripper_duration = 1.5
        self.servo_duration = 5.0
        self.rest_duration = 0.1



        # =========================================================================
        # Custom Variables for Placing & Stacking Mission
        # =========================================================================
        self.block_height = 0.02       # (2 cm = 0.02 m)
        self.stack_clearance = 0.0015  # Buffer Distance for Stacking (1.5 mm)
       
        # Fixed Place Drop Zone (4x4 Task Space Matrix) (Need Adjustment)
        self.fixed_place_targets = {
            "blue_3": np.array([
                [1, 0, 0,  0.10],  # x = 0.20 m
                [0, 1, 0, -0.10],  # y = -0.15 m
                [0, 0, 1,  0.03],  # z = 0.03 m      # Drop Zone for Blue
                [0, 0, 0,  1.00]
            ]),
            "green_3": np.array([
                [1, 0, 0,  0.10],  # x = 0.25 m
                [0, 1, 0,  0.10],  # y = 0.00 m
                [0, 0, 1,  0.03],  # z = 0.03 m      # Drop Zone for Green
                [0, 0, 0,  1.00]
            ]),
            "red_3": np.array([
                [1, 0, 0,  0.10],  # x = 0.20 m
                [0, 1, 0,  0.00],  # y = 0.15 m
                [0, 0, 1,  0.03],  # z = 0.03 m      # Drop Zone for Red
                [0, 0, 0,  1.00]
            ])
        }
       
        # Safe RTB q (Home Pose) (Need Adjustment)
        self.home_q = pulse2angle([500, 736, 40, 180, 500])
        
        self.set_joint_positions(self.home_q, 5.0)
        time.sleep(5.0)

        self.place_q = (
            np.array(pulse2angle([955,408,137,240,500])),
            np.array(pulse2angle([866,408,137,240,500])),
            np.array(pulse2angle([801,409,138,240,500])),
        )
        self.block_cnt = 0

        #####

    ### Utility functions for controlling the robot arm and gripper
    ### DO NOT MODIFY THESE FUNCTIONS

    def gripper_close(self, duration=1.5):
        '''
        Close the gripper.
        '''
        self._set_position_pulse([(10, 550)], duration)

    def gripper_open(self, duration=1.5):
        '''
        Open the gripper.
        '''
        self._set_position_pulse([(10, 100)], duration)

    def get_joint_positions(self):
        '''
        Returns: current joint positions in "radians" as a numpy array
        '''
        q = self.get_joint_positions_pulse() # Base function method
        return pulse2angle(q)
    
    def set_joint_positions(self, q, duration):
        '''
        q: "radians", list or numpy array of joint angles
        duration: time to move in seconds
        '''
        pulse = angle2pulse(q)
        self.set_joint_positions_pulse(pulse, duration) # Base function method

    def get_detected_markers(self):
        rclpy.spin_once(self, timeout_sec=0.01)
        if self.image is not None:
            detected_markers = self.marker_detector.detect_markers_with_pose(self.image)
            self.image = None
        else:
            detected_markers = {}
        return detected_markers

    def rvec_tvec_to_T(self, rvec, tvec):
        """
        Convert rotation vector and translation vector to a 4x4 transformation matrix.
        """
        R = SO3.exp(rvec).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()
        return T

    def interpolate_T(self, T_start, T_end, alpha):
        """
        Interpolate between two 4x4 transformation matrices.
        alpha: interpolation factor (0.0 to 1.0)
        """
        R_start = T_start[:3, :3]
        R_end = T_end[:3, :3]
        t_start = T_start[:3, 3]
        t_end = T_end[:3, 3]

        R_interp = SO3.interpolate(R_start, R_end, alpha).as_matrix()
        t_interp = (1 - alpha) * t_start + alpha * t_end

        T_interp = np.eye(4)
        T_interp[:3, :3] = R_interp
        T_interp[:3, 3] = t_interp
        return T_interp

    #########################################################################################################################
    ## TODO : Implement the following functions to complete the grasping functionality.

    def get_block_pose(self, 
        detection_result : MarkerDetectionResult, 
    ):
        """
            Calculate the block's pose in the world coordinate frame 
            given the marker's rotation and translation vectors.
            Returns a 4x4 transformation matrix representing the block's pose.
        """
        ######
        ## TODO : Implement this function to compute the block's pose, given the marker's rvec and tvec.
        T_block = None

        rvec = detection_result.rvec
        tvec = detection_result.tvec

        R_cam_marker = SO3.exp(rvec).as_matrix()
        
        B_aug = jnp.asarray(tvec).reshape(3, 1)
        C_aug = jnp.array([[0, 0, 0]])
        D_aug = jnp.array([[1]])
        T_cam_marker = jnp.block([[R_cam_marker, B_aug], [C_aug, D_aug]])

        T_world_cam = forward_kinematics(self.get_joint_positions(), 'cam')
        
        T_block = jnp.matmul(T_world_cam, T_cam_marker)
        T_block = np.array(T_block)

        ######

        return T_block
    
    def solve_config(
        self, 
        T_target: np.ndarray, 
        seed: np.ndarray, 
        tolerance: float = 0.03
    ):
        """
            Calculate valid configuration from target T matrix in a given tolerance.
            Various conditions for rejection : including None output of IK itself, and rotation-position over-tradeoff
            Returns valid configuration especially for grasping, and total error.
        """
        result = inverse_kinematics(seed, T_target)
        
	# If IK itself fails by some reason
        if result is None or result.get("sol") is None:
            return None

        pos_err = result.get("pos_error")
        rot_err = result.get("rot_error")
        
        # Ensure that pos_err and rot_err is 'a number'.
        if pos_err is None or not np.isfinite(pos_err):
            return None
        
        if rot_err is None or not np.isfinite(rot_err):
            return None
            
        print(f"rotational error : {rot_err}, positional error : {pos_err}")
        
        # If rotational error exceeds 0.15 rad/s * 1s, pos error exceeds total 1.5cm
        # To prevent over-tradeoff of pos/rot error.
        if rot_err > 0.015 or pos_err > 0.015:
            return None
        
        # Further weigh rotational error
        tot_err = pos_err + 1.3 * rot_err
        print(tot_err)
        
        # Check if total error is within tolerance
        if tot_err > tolerance:
            return None
        
        q = np.asarray(result["sol"], dtype=float)

        return q, float(tot_err)

    def get_grasp_q(
        self,
        T_world_marker: np.ndarray,
        grasp_depth: float = 0.01,
        approach_height: float = 0.02,
        tolerance: float = 0.03
    ):
        """
            Get a configuration of robotic manipulator from T matrix of marker within given tolerance.
            Assess multiple grasping poses including:
              1) Grasping the cube normally.
              2) Tilted grasping
            Return the minimum error configuration among the grasping poses satisfying the tolerance constraint.
        """

        T_world_marker = np.asarray(T_world_marker, dtype=float)

        q_seed = np.asarray(self.get_joint_positions(), dtype=float)
        
        best_score = float("inf")
        best_q = None

        for yaw_deg in (0, 10, 80, 90, 100, 170, 180, 190, 260, 270, 280, 350):

            yaw_deg = np.deg2rad(yaw_deg)

            R_gripper = np.array([[-np.cos(yaw_deg), np.sin(yaw_deg), 0],
                                [np.sin(yaw_deg), np.cos(yaw_deg), 0],
                                [0, 0, -1]])
            
            T_marker_tcp = np.eye(4)
            T_marker_tcp[:3, :3] = R_gripper
            T_marker_tcp[:3, 3] = [0.0, 0.0, -0.015]
            
            T_tcp_realtcp = np.eye(4)
            T_tcp_realtcp[:3, 3] = [-0.005, 0.0, +grasp_depth]
            
            T_marker_tcp = np.matmul(T_marker_tcp, T_tcp_realtcp)

            # World T grasp calculation
            T_grasp = T_world_marker @ T_marker_tcp

            sol_res = self.solve_config(T_grasp, q_seed, tolerance)
            
            if sol_res is None:
                continue
                
            q_grasp = sol_res[0]
            tot_err = sol_res[1]

            if  q_grasp is not None and best_score > tot_err:
                best_score = tot_err
                best_q = q_grasp
                
        for tilt_deg in (-45, -30, -15, 15, 30, 45):
            
            tilt_deg = np.deg2rad(tilt_deg)
            
            for yaw_deg in (0, 90, 180, 270):
                yaw_deg = np.deg2rad(yaw_deg)
                
                R_gripper = np.array([[-np.cos(yaw_deg), np.sin(yaw_deg), 0],
                                [np.sin(yaw_deg), np.cos(yaw_deg), 0],
                                [0, 0, -1]])
            
                T_marker_tcp = np.eye(4)
                T_marker_tcp[:3, :3] = R_gripper
                T_marker_tcp[:3, 3] = [0.0, 0.0, -0.015]
                
                T_tcp_tilt = np.eye(4)
                
                R_tilt = np.array([[np.cos(tilt_deg), 0, -np.sin(tilt_deg)],
                                [0, 1, 0],
                                [np.sin(tilt_deg), 0, np.cos(tilt_deg)]])
                                
                T_tcp_tilt[:3, :3] = R_tilt
                
                T_marker_tcp = np.matmul(T_marker_tcp, T_tcp_tilt)
            
                T_tcp_realtcp = np.eye(4)
                T_tcp_realtcp[:3, 3] = [-0.005, 0.0, +grasp_depth]
            
                T_marker_tcp = np.matmul(T_marker_tcp, T_tcp_realtcp)

                # World T grasp calculation
                T_grasp = T_world_marker @ T_marker_tcp
                
                print(f"tilted grasp degree : {tilt_deg}")
                
                sol_res = self.solve_config(T_grasp, q_seed, tolerance)
                
                if sol_res is None:
                    continue
                
                q_grasp = sol_res[0]
                tot_err = sol_res[1]

                if  q_grasp is not None and best_score > tot_err:
                    # return q_approach, q_grasp
                    best_score = tot_err
                    best_q = q_grasp
        
        return best_q

    def grasp(self, target_marker_id: int | str) -> bool:
        """
            Execute grasping action for the object with the specified marker ID.
            If target_marker_id is a string, it should map to a predefined marker ID.
            Returns True if successful, False otherwise.
        """
        ######
        ## TODO : Implement this function to perform the complete grasping sequence.
        is_success = False

        self.gripper_open()
        time.sleep(self.gripper_duration+self.rest_duration)
        
        detected_markers_list = []
        detected_markers = {} 
        target_marker_id = int(target_marker_id)

        for _ in range(50): # Collect every attempt containing the target marker.
            detected_markers = self.get_detected_markers()
            if target_marker_id in detected_markers:
                detected_markers_list.append(detected_markers.copy())

        if not detected_markers_list:
            print(f"marker {target_marker_id} is not detected.")
            return is_success
            
        if len(detected_markers_list) <= 3:
            print(f"marker {target_marker_id} is rarely detected.")
            return is_success
            
        T_block_cand_list = []
        
        for dt_mk in detected_markers_list:
            temp_T_block = self.get_block_pose(dt_mk[target_marker_id])
            T_block_cand_list.append(temp_T_block)

        # Use the latest valid observation until robust selection is added. (Decided not to include)
        detected_markers = detected_markers_list[-1]

        T_block = self.get_block_pose(detected_markers[target_marker_id])
        
        q_grasp = self.get_grasp_q(T_block)
        
        # If there is no valid grasping configuration, return False.
        if q_grasp is None:
            return is_success

        q_home = self.get_joint_positions()
        
        print(f"targeting {q_grasp}")

        self.set_joint_positions(q_grasp,self.servo_duration) #move to block and...
        time.sleep(self.servo_duration+self.rest_duration)

        self.gripper_close() #grasp block...
        time.sleep(self.gripper_duration+self.rest_duration)

        self.set_joint_positions(q_home,self.servo_duration) #return to original position
        time.sleep(self.servo_duration+self.rest_duration)

        self.last_q = q_home
        
        is_success = True
        
        ######
        return is_success

    def place(self, action_name) -> bool:   # output : is_success
        """
        Execute placing action given an action name ("blue_3", "green_3", "red_3").
        """
        pre_q = self.place_q[self.block_cnt].copy();pre_q[3] = 0.0
        self.set_joint_positions(pre_q, self.servo_duration) #move to pre-place position
        time.sleep(self.servo_duration+self.rest_duration)

        self.set_joint_positions(self.place_q[self.block_cnt], self.servo_duration) #move to target position
        time.sleep(self.servo_duration+self.rest_duration)
        self.gripper_open(self.gripper_duration) #release block
        time.sleep(self.gripper_duration+self.rest_duration)

        self.set_joint_positions(self.home_q, self.servo_duration) #return to home position
        time.sleep(self.servo_duration+self.rest_duration)
        self.gripper_close(self.gripper_duration) #close gripper
        time.sleep(self.gripper_duration+self.rest_duration)

        self.block_cnt += 1

        return True
