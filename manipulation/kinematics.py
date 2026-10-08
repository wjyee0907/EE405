import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from scipy.optimize import least_squares
from .utils.kinematics_utils import get_urdf
from jaxlie import SO3

#####

def forward_kinematics(
    q: np.ndarray, 
    target: str,
) -> np.ndarray:
    """
    Compute the forward kinematics for the robotic arm.
    Args:
        q (np.ndarray): Joint angles (1D array of size 5) in radian.
        target (str): Target frame to compute the pose for. Options are:
                      'j1', 'j2', 'j3', 'j4', 'j5', 'cam', 'tcp'.
                      Default is 'j1'.
    Returns:
        jnp.ndarray: 4x4 transformation matrix of the target frame.
    """
    
    ######
    # Double precision is needed to match the URDF reference at numerical accuracy.
    jax.config.update("jax_enable_x64", True)
    urdf_dict = get_urdf()
    rigidTFs = ["T_base_joint_1", "T_joint1_joint_2", "T_joint2_joint_3", "T_joint3_joint_4", "T_joint4_joint_5"]

    def get_rotation(w, theta): # w : ndarray, theta : float
        omega = jnp.array(theta * w)
        R_res = SO3.exp(w * theta).as_matrix()

        B_aug = jnp.array([[0],[0],[0]])
        C_aug = jnp.array([[0, 0, 0]])
        D_aug = jnp.array([[1]])
        T_res = jnp.block([[R_res, B_aug], [C_aug, D_aug]])

        return T_res

    pose = urdf_dict["T_base"]

    pose = (pose@urdf_dict["T_base_joint_1"])@get_rotation(q[0],urdf_dict["Axis_joint_1"])
    if target == "j1":
        return pose
    pose = (pose@urdf_dict["T_joint1_joint_2"])@get_rotation(q[1],urdf_dict["Axis_joint_2"])
    if target == "j2":
        return pose
    pose = (pose@urdf_dict["T_joint2_joint_3"])@get_rotation(q[2],urdf_dict["Axis_joint_3"])
    if target == "j3":
        return pose
    pose = (pose@urdf_dict["T_joint3_joint_4"])@get_rotation(q[3],urdf_dict["Axis_joint_4"])
    if target == "j4":
        return pose
    elif target =="cam":
        pose = pose@urdf_dict["T_joint4_cam_offset"]
        return pose
    pose = (pose@urdf_dict["T_joint4_joint_5"])@get_rotation(q[4],urdf_dict["Axis_joint_5"])
    if target == "j5":
        return pose
    elif target == "tcp":
        pose = pose@urdf_dict["T_joint5_TCP_offset"]
        return pose
    return pose


    #####
def err_fn(q,tar):
    q = jnp.array(q)
    T_ee = forward_kinematics(q,"tcp")
    tar = jnp.array(tar)
    R_ee = T_ee[0:3,0:3]
    P_ee = T_ee[0:3,3]
    R_tar = tar[0:3,0:3]
    P_tar = tar[0:3,3]
    rot_err = SO3.from_matrix(R_ee.T @ R_tar).log()
    pos_err = P_ee-P_tar
    return jnp.concatenate([rot_err, pos_err])

jac_jit = jax.jit(jax.jacfwd(err_fn))
err_jit = jax.jit(err_fn)

def inverse_kinematics(
    q: np.ndarray, 
    T_target: np.ndarray
) -> dict | None:
    """
    Compute the inverse kinematics for the robotic arm.
    Args:
        q (np.ndarray): Initial joint angles (1D array of size 5) in radian.
        T_target (np.ndarray): 4x4 transformation matrix of the target end-effector pose (tcp).
    
        Returns:
            dict | None: A dictionary containing the solution joint angles and position error, or None if failed.
                {
                    "sol": np.ndarray,  # Solution joint angles (1D array of size 5) in radian or None if failed
                    "pos_error": float  # Position error in meters or None if failed
                }
    """

    #####
    jax.config.update("jax_enable_x64", True)

    

    res = least_squares(lambda q:np.asarray(err_jit(q,T_target)),
                        np.asarray(q),
                        jac = lambda q:np.asarray(jac_jit(q,T_target)))

    ik_solution = res.x
    pos_error = float(jnp.linalg.norm(err_fn(ik_solution,T_target)[3:])) + 0.3 * float(jnp.linalg.norm(err_fn(ik_solution,T_target)[:3]))
    # pos_error = float(np.linalg.norm(res.fun))

    #####

    result = {}
    result["sol"] = ik_solution
    result["pos_error"] = pos_error

    return result
