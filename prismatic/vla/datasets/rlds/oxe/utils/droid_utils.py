"""Episode transforms for DROID dataset."""

from typing import Any, Dict
import math

import tensorflow as tf
#import tensorflow_graphics.geometry.transformation as tfg

def euler_from_rotation_matrix(
    rotation_matrix,
    name: str = "euler_from_rotation_matrix",
    eps: float = 1e-3,
    add_asserts_to_graph: bool = True,
):
    """Converts rotation matrices to Euler angles (x-y-z order). https://github.com/tensorflow/graphics/blob/36e707fe51c8d48b0849eed0dea38d63990fc765/tensorflow_graphics/geometry/transformation/euler.py#L140

    Args:
      rotation_matrix: Tensor shape [..., 3, 3]
      name: name scope
      eps: tolerance for rotation matrix validity checks
      add_asserts_to_graph: if True, add tf.debugging asserts (like TFG flag)

    Returns:
      Tensor shape [..., 3] with angles (theta_x, theta_y, theta_z)
    """

    def assert_rotation_matrix_normalized(matrix, eps=1e-3, name="assert_rotation_matrix_normalized"):
        """Checks whether a matrix is a (proper) rotation matrix: R^T R = I and det(R)=1."""
        if not add_asserts_to_graph:
            return matrix

        with tf.name_scope(name):
            matrix = tf.convert_to_tensor(value=matrix)

            # Basic shape checks: rank>=2 and last dims are 3x3
            tf.debugging.assert_rank_at_least(matrix, 2, message="rotation_matrix must have rank >= 2")
            tf.debugging.assert_equal(tf.shape(matrix)[-2], 3, message="rotation_matrix last-2 dim must be 3")
            tf.debugging.assert_equal(tf.shape(matrix)[-1], 3, message="rotation_matrix last-1 dim must be 3")

            # Orthonormal check: R^T R ≈ I
            rt = tf.linalg.matrix_transpose(matrix)
            should_be_I = tf.matmul(rt, matrix)
            I = tf.eye(3, batch_shape=tf.shape(matrix)[:-2], dtype=matrix.dtype)
            ortho_ok = tf.reduce_all(tf.abs(should_be_I - I) <= tf.cast(eps, matrix.dtype), axis=[-2, -1])

            # Proper rotation check: det(R) ≈ 1
            det = tf.linalg.det(matrix)
            det_ok = tf.abs(det - tf.ones_like(det)) <= tf.cast(eps, matrix.dtype)

            is_matrix_normalized = tf.logical_and(ortho_ok, det_ok)

            with tf.control_dependencies([
                tf.debugging.assert_equal(
                    is_matrix_normalized,
                    tf.ones_like(is_matrix_normalized, dtype=tf.bool),
                    message="rotation_matrix is not normalized (not a valid SO(3) rotation matrix).",
                )
            ]):
                return tf.identity(matrix)

    def nonzero_sign(x):
        # returns +1 for x>=0 else -1 (avoids returning 0)
        one = tf.ones_like(x)
        return tf.where(x >= 0, one, -one)

    def select_eps_for_addition(dtype):
        # Similar spirit to tfg asserts.select_eps_for_addition
        # Keep it small but dtype-aware.
        if dtype == tf.float16:
            return tf.cast(1e-3, dtype)
        if dtype == tf.bfloat16:
            return tf.cast(1e-3, dtype)
        if dtype == tf.float32:
            return tf.cast(1e-6, dtype)
        # float64 etc.
        return tf.cast(1e-12, dtype)

    def general_case(rotation_matrix, r20, eps_addition):
        """Handles the general case."""
        theta_y = -tf.asin(r20)
        sign_cos_theta_y = nonzero_sign(tf.cos(theta_y))

        r00 = rotation_matrix[..., 0, 0]
        r10 = rotation_matrix[..., 1, 0]
        r21 = rotation_matrix[..., 2, 1]
        r22 = rotation_matrix[..., 2, 2]

        r00 = nonzero_sign(r00) * eps_addition + r00
        r22 = nonzero_sign(r22) * eps_addition + r22

        # cos(theta_y) can be 0 at gimbal lock; masked out later.
        theta_z = tf.atan2(r10 * sign_cos_theta_y, r00 * sign_cos_theta_y)
        theta_x = tf.atan2(r21 * sign_cos_theta_y, r22 * sign_cos_theta_y)

        return tf.stack((theta_x, theta_y, theta_z), axis=-1)

    def gimbal_lock(rotation_matrix, r20, eps_addition):
        """Handles Gimbal locks."""
        r01 = rotation_matrix[..., 0, 1]
        r02 = rotation_matrix[..., 0, 2]
        sign_r20 = nonzero_sign(r20)

        r02 = nonzero_sign(r02) * eps_addition + r02
        theta_x = tf.atan2(-sign_r20 * r01, -sign_r20 * r02)
        theta_y = -sign_r20 * tf.cast(math.pi / 2.0, dtype=r20.dtype)
        theta_z = tf.zeros_like(theta_x)

        return tf.stack((theta_x, theta_y, theta_z), axis=-1)

    with tf.name_scope(name):
        rotation_matrix = tf.convert_to_tensor(value=rotation_matrix)

        # Shape checks (static-ish): we still do runtime asserts above in assert fn.
        rotation_matrix = assert_rotation_matrix_normalized(rotation_matrix, eps=eps)

        r20 = rotation_matrix[..., 2, 0]
        eps_addition = select_eps_for_addition(rotation_matrix.dtype)

        general_solution = general_case(rotation_matrix, r20, eps_addition)
        gimbal_solution = gimbal_lock(rotation_matrix, r20, eps_addition)

        # Same condition as你贴的tfg版本：abs(r20) == 1 视为 gimbal
        is_gimbal = tf.equal(tf.abs(r20), tf.ones_like(r20))
        gimbal_mask = tf.stack((is_gimbal, is_gimbal, is_gimbal), axis=-1)

        return tf.where(gimbal_mask, gimbal_solution, general_solution)

def rmat_to_euler(rot_mat):
    #return tfg.euler.from_rotation_matrix(rot_mat)
    return euler_from_rotation_matrix(rot_mat)

def rotation_matrix_3d_from_euler(
    angles,
    name: str = "rotation_matrix_3d_from_euler",
) -> tf.Tensor:
  r"""Convert Euler angles to a 3D rotation matrix.

  The resulting matrix is R = Rz * Ry * Rx.

  Args:
    angles: Tensor with shape [..., 3], angles about x, y, z in radians.
    name: Name scope.

  Returns:
    Tensor with shape [..., 3, 3].
  """

  def _check_static_last_dim_is_3(tensor, tensor_name: str):
    # 仅做一个轻量的静态检查（不依赖 tfg.shape.check_static）
    ts = tensor.shape
    if ts.rank is not None and ts.rank >= 1:
      if ts[-1] is not None and ts[-1] != 3:
        raise ValueError(
            f"{tensor_name} must have last dimension 3, but got shape {ts}"
        )

  def _build_matrix_from_sines_and_cosines(sin_angles, cos_angles):
    # angles = [x, y, z]
    sx, sy, sz = tf.unstack(sin_angles, axis=-1)
    cx, cy, cz = tf.unstack(cos_angles, axis=-1)

    # R = Rz * Ry * Rx
    # [ [ cz*cy,  cz*sy*sx - sz*cx,  cz*sy*cx + sz*sx ],
    #   [ sz*cy,  sz*sy*sx + cz*cx,  sz*sy*cx - cz*sx ],
    #   [  -sy ,        cy*sx     ,        cy*cx      ] ]
    r00 = cz * cy
    r01 = cz * sy * sx - sz * cx
    r02 = cz * sy * cx + sz * sx

    r10 = sz * cy
    r11 = sz * sy * sx + cz * cx
    r12 = sz * sy * cx - cz * sx

    r20 = -sy
    r21 = cy * sx
    r22 = cy * cx

    row0 = tf.stack([r00, r01, r02], axis=-1)
    row1 = tf.stack([r10, r11, r12], axis=-1)
    row2 = tf.stack([r20, r21, r22], axis=-1)
    return tf.stack([row0, row1, row2], axis=-2)  # [..., 3, 3]

  with tf.name_scope(name):
    angles = tf.convert_to_tensor(value=angles)
    _check_static_last_dim_is_3(angles, "angles")

    sin_angles = tf.sin(angles)
    cos_angles = tf.cos(angles)
    return _build_matrix_from_sines_and_cosines(sin_angles, cos_angles)

def euler_to_rmat(euler):
    #return tfg.rotation_matrix_3d.from_euler(euler)
    return rotation_matrix_3d_from_euler(euler)

def rotation_matrix_3d_inverse(
    matrix,
    name: str = "rotation_matrix_3d_inverse",
    eps: float = 1e-3,
    add_asserts: bool = True,
):
    """Computes the inverse of a 3D rotation matrix (inverse == transpose).

    Args:
      matrix: TensorLike, shape [..., 3, 3]
      name: op name scope
      eps: tolerance for orthonormality & det check
      add_asserts: whether to add tf.debugging asserts into the graph

    Returns:
      Tensor, shape [..., 3, 3]
    """

    def _check_shape(m: tf.Tensor) -> tf.Tensor:
        # Static checks if possible
        if m.shape.rank is not None:
            if m.shape.rank < 2:
                raise ValueError(f"`matrix` rank must be >= 2, got rank={m.shape.rank}")
            if m.shape[-2] != 3 or m.shape[-1] != 3:
                raise ValueError(f"`matrix` last dims must be (3,3), got {m.shape[-2:]}")

        # Dynamic checks (works even when shape unknown)
        shape_dyn = tf.shape(m)
        with tf.control_dependencies([
            tf.debugging.assert_greater_equal(tf.rank(m), 2, message="`matrix` rank must be >= 2"),
            tf.debugging.assert_equal(shape_dyn[-2], 3, message="`matrix` must have shape [..., 3, 3]"),
            tf.debugging.assert_equal(shape_dyn[-1], 3, message="`matrix` must have shape [..., 3, 3]"),
        ]):
            return tf.identity(m)

    def _assert_rotation_matrix_normalized(m: tf.Tensor) -> tf.Tensor:
        # Equivalent spirit to tfg.rotation_matrix_3d.assert_rotation_matrix_normalized:
        # checks R^T R ≈ I and det(R) ≈ 1.
        if not add_asserts:
            return m

        # Compute R^T R
        rt = tf.linalg.matrix_transpose(m)
        should_be_I = tf.matmul(rt, m)

        I = tf.eye(3, batch_shape=tf.shape(should_be_I)[:-2], dtype=m.dtype)

        # max |R^T R - I|
        ortho_err = tf.reduce_max(tf.abs(should_be_I - I), axis=[-2, -1])

        # det(R) close to 1
        det = tf.linalg.det(m)
        det_err = tf.abs(det - tf.ones_like(det))

        with tf.control_dependencies([
            tf.debugging.assert_less_equal(
                ortho_err, tf.cast(eps, m.dtype),
                message="Rotation matrix not orthonormal within eps",
            ),
            tf.debugging.assert_less_equal(
                det_err, tf.cast(eps, m.dtype),
                message="Rotation matrix determinant not 1 within eps",
            ),
        ]):
            return tf.identity(m)

    with tf.name_scope(name):
        matrix = tf.convert_to_tensor(matrix)
        matrix = _check_shape(matrix)
        matrix = _assert_rotation_matrix_normalized(matrix)

        # inverse of rotation matrix == transpose on the last two dims
        return tf.linalg.matrix_transpose(matrix)

def invert_rmat(rot_mat):
    #return tfg.rotation_matrix_3d.inverse(rot_mat)
    return rotation_matrix_3d_inverse(rot_mat)


def rotmat_to_rot6d(mat):
    """
    Converts rotation matrix to R6 rotation representation (first two rows in rotation matrix).
    Args:
        mat: rotation matrix

    Returns: 6d vector (first two rows of rotation matrix)

    """
    r6 = mat[..., :2, :]
    r6_0, r6_1 = r6[..., 0, :], r6[..., 1, :]
    r6_flat = tf.concat([r6_0, r6_1], axis=-1)
    return r6_flat


def velocity_act_to_wrist_frame(velocity, wrist_in_robot_frame):
    """
    Translates velocity actions (translation + rotation) from base frame of the robot to wrist frame.
    Args:
        velocity: 6d velocity action (3 x translation, 3 x rotation)
        wrist_in_robot_frame: 6d pose of the end-effector in robot base frame

    Returns: 9d velocity action in robot wrist frame (3 x translation, 6 x rotation as R6)

    """
    R_frame = euler_to_rmat(wrist_in_robot_frame[:, 3:6])
    R_frame_inv = invert_rmat(R_frame)

    # world to wrist: dT_pi = R^-1 dT_rbt
    vel_t = (R_frame_inv @ velocity[:, :3][..., None])[..., 0]

    # world to wrist: dR_pi = R^-1 dR_rbt R
    dR = euler_to_rmat(velocity[:, 3:6])
    dR = R_frame_inv @ (dR @ R_frame)
    dR_r6 = rotmat_to_rot6d(dR)
    return tf.concat([vel_t, dR_r6], axis=-1)


def rand_swap_exterior_images(img1, img2):
    """
    Randomly swaps the two exterior images (for training with single exterior input).
    """
    return tf.cond(tf.random.uniform(shape=[]) > 0.5, lambda: (img1, img2), lambda: (img2, img1))


def droid_baseact_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    DROID dataset transformation for actions expressed in *base* frame of the robot.
    """
    dt = trajectory["action_dict"]["cartesian_velocity"][:, :3]
    dR = trajectory["action_dict"]["cartesian_velocity"][:, 3:6]

    trajectory["action"] = tf.concat(
        (
            dt,
            dR,
            1 - trajectory["action_dict"]["gripper_position"],
        ),
        axis=-1,
    )
    trajectory["observation"]["exterior_image_1_left"], trajectory["observation"]["exterior_image_2_left"] = (
        rand_swap_exterior_images(
            trajectory["observation"]["exterior_image_1_left"],
            trajectory["observation"]["exterior_image_2_left"],
        )
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["cartesian_position"],
            trajectory["observation"]["gripper_position"],
        ),
        axis=-1,
    )
    return trajectory


def droid_wristact_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    DROID dataset transformation for actions expressed in *wrist* frame of the robot.
    """
    wrist_act = velocity_act_to_wrist_frame(
        trajectory["action_dict"]["cartesian_velocity"], trajectory["observation"]["cartesian_position"]
    )
    trajectory["action"] = tf.concat(
        (
            wrist_act,
            trajectory["action_dict"]["gripper_position"],
        ),
        axis=-1,
    )
    trajectory["observation"]["exterior_image_1_left"], trajectory["observation"]["exterior_image_2_left"] = (
        rand_swap_exterior_images(
            trajectory["observation"]["exterior_image_1_left"],
            trajectory["observation"]["exterior_image_2_left"],
        )
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["cartesian_position"],
            trajectory["observation"]["gripper_position"],
        ),
        axis=-1,
    )
    return trajectory


def droid_finetuning_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    DROID dataset transformation for actions expressed in *base* frame of the robot.
    """
    dt = trajectory["action_dict"]["cartesian_velocity"][:, :3]
    dR = trajectory["action_dict"]["cartesian_velocity"][:, 3:6]
    trajectory["action"] = tf.concat(
        (
            dt,
            dR,
            1 - trajectory["action_dict"]["gripper_position"],
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["cartesian_position"],
            trajectory["observation"]["gripper_position"],
        ),
        axis=-1,
    )
    return trajectory


def zero_action_filter(traj: Dict) -> bool:
    """
    Filters transitions whose actions are all-0 (only relative actions, no gripper action).
    Note: this filter is applied *after* action normalization, so need to compare to "normalized 0".
    """
    DROID_Q01 = tf.convert_to_tensor(
        [
            -0.7776297926902771,
            -0.5803514122962952,
            -0.5795090794563293,
            -0.6464047729969025,
            -0.7041108310222626,
            -0.8895104378461838,
        ]
    )
    DROID_Q99 = tf.convert_to_tensor(
        [
            0.7597932070493698,
            0.5726242214441299,
            0.7351000607013702,
            0.6705610305070877,
            0.6464948207139969,
            0.8897542208433151,
        ]
    )
    DROID_NORM_0_ACT = 2 * (tf.zeros_like(traj["action"][:, :6]) - DROID_Q01) / (DROID_Q99 - DROID_Q01 + 1e-8) - 1

    return tf.reduce_any(tf.math.abs(traj["action"][:, :6] - DROID_NORM_0_ACT) > 1e-5)
