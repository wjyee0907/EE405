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
        self.home_q = pulse2angle([500, 736, 40, 219, 500])
        
        self.set_joint_positions(self.home_q, 5.0)
        time.sleep(5.0)


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
        
        print(rvec)

        R_cam_marker = SO3.exp(rvec).as_matrix()
        
        B_aug = jnp.asarray(tvec).reshape(3, 1)
        print(B_aug)
        C_aug = jnp.array([[0, 0, 0]])
        D_aug = jnp.array([[1]])
        T_cam_marker = jnp.block([[R_cam_marker, B_aug], [C_aug, D_aug]])
        print(T_cam_marker)
        
        print("help")
        
        print(self.get_joint_positions())

        T_world_cam = forward_kinematics(self.get_joint_positions(), 'cam')
        
        print(T_world_cam)

        T_block = jnp.matmul(T_world_cam, T_cam_marker)

        T_block = np.array(T_block)

        ######

        return T_block
    
    def solve_waypoint(self, T_target, seed, tolerance) -> np.ndarray | None:
        result = inverse_kinematics(seed, T_target)
        
        print(result.get("pos_error"))
    
        if result is None or result.get("sol") is None or result.get("pos_error") > tolerance:
            return None
                
        q = np.asarray(result["sol"], dtype=float)
                
        return q

    def get_grasp_waypoints(
        self,
        T_world_marker: np.ndarray,
        grasp_depth: float = 0.025,
        approach_height: float = 0.02,
        tolerance: float = 0.007
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Return (T_approach, T_grasp), or None if no sampled pair passes IK.

        Assume a centered marker on the top face, with +z pointing outward.
        Sample jaw orientations with the approach at 90 degrees to that face.
        The TCP grasp point is grasp_depth metres below the marker; the
        approach point is approach_height metres above it. Both poses have
        the same orientation, with the approach point behind TCP along -z.

        Checks position, orientation and joint limits at both endpoints.
        This computes poses only; it does not execute motion or check collisions.
        """
        T_world_marker = np.asarray(T_world_marker, dtype=float)

        q_seed = np.asarray(self.get_joint_positions(), dtype=float)
        
        print("before for loop")

        for yaw_deg in (0, 10, 80, 90, 100, 170, 180, 190, 260, 270, 280, 350):

            yaw_deg = np.deg2rad(yaw_deg)

            R_gripper = np.array([[-np.cos(yaw_deg), np.sin(yaw_deg), 0],
                                [np.sin(yaw_deg), np.cos(yaw_deg), 0],
                                [0, 0, -1]])
            
            T_marker_tcp = np.eye(4)
            T_marker_tcp[:3, :3] = R_gripper
            T_marker_tcp[:3, 3] = [0.0, 0.0, 0.0]
            
            T_tcp_realtcp = np.eye(4)
            T_tcp_realtcp[:3, 3] = [-0.005, 0.0, +grasp_depth]
            
            T_marker_tcp = np.matmul(T_marker_tcp, T_tcp_realtcp)

            # World T grasp calculation
            T_grasp = T_world_marker @ T_marker_tcp

            T_approach = T_grasp.copy()
            # place the approach pose, +approach height z direction.
            T_approach[2, 3] += approach_height

            q_approach = self.solve_waypoint(T_approach, q_seed, tolerance)
            
            print(q_approach)

            if q_approach is None:
                continue

            q_grasp = self.solve_waypoint(T_grasp, q_approach, tolerance)
            
            print(q_approach, q_grasp)

            if  q_grasp is not None:
                return q_approach, q_grasp

        return None

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


        detected_markers = {} 
        target_marker_id = int(target_marker_id)

        for _ in range(50): #retry 50 times if fail to recognize marker
            detected_markers = self.get_detected_markers()
            if target_marker_id in detected_markers:
                break


        if not target_marker_id in detected_markers.keys():
            print(f"marker {target_marker_id} is not detected.\ndetected markers are {list(detected_markers.keys())}")
            return is_success
            


        T_block = self.get_block_pose(detected_markers[target_marker_id])
        
        print(T_block)


        q_res_tuple = self.get_grasp_waypoints(T_block)

        if q_res_tuple is None:
            return is_success

        q_app = q_res_tuple[0]
        q_grasp = q_res_tuple[1]
        
        print(q_app, q_grasp)
        
        q_home = self.get_joint_positions()
        
        print("calculation good")

        self.set_joint_positions(q_app,self.servo_duration) #move to app and...
        time.sleep(self.servo_duration+self.rest_duration)

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
        # Define Target Pose & Approach Pose (World Z +8 cm)
        T_place = self.fixed_place_targets[action_name].copy()

        T_pre_place = T_place.copy()
        T_pre_place[2, 3] += 0.08

        # current_q for IK
        curr_q = self.get_joint_positions()

        # Approach IK (pos_error < 0.01 m)
        res_pre = inverse_kinematics(curr_q, T_pre_place)
        if res_pre is None or res_pre["sol"] is None or res_pre["pos_error"] > 0.01:
            print(f"[Place Error] IK pre-place failed or pos_error ({res_pre['pos_error'] if res_pre else 'None'}) > 0.01")
            return False
        q_pre = res_pre["sol"]

        # Target IK
        res_place = inverse_kinematics(q_pre, T_place)   # use q_pre for continuity
        if res_place is None or res_place["sol"] is None or res_place["pos_error"] > 0.01:
            print(f"[Place Error] IK place failed or pos_error ({res_place['pos_error'] if res_place else 'None'}) > 0.01")
            return False
        q_place = res_place["sol"]

        # 5. Motion Sequence
        # Step 1: Go 8 cm above drop zone first
        '''
        self.set_joint_positions(q_pre, duration=2.0)
        time.sleep(2.2)
        '''

        # Step 2: Direct Descend
        # Modified Step 1~2 : Go directly to drop zone
        self.set_joint_positions(q_place, duration=1.2)
        # time.sleep(1.4)
        time.sleep(3.0)

        # Step 3: Open Gripper
        self.gripper_open(duration=1.2)
        time.sleep(1.3)

        # Step 4: Direct Ascend
        self.set_joint_positions(q_pre, duration=1.2)
        time.sleep(1.4)

        # Step 5: RTB (Return to Base)
        if self.home_q is not None:
            self.set_joint_positions(self.home_q, duration=1.8)
            time.sleep(1.9)

        print(f"[Place Success] Completed: {action_name}")
        return True
