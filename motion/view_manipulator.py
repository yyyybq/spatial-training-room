from annotated_types import T
import numpy as np
from scipy.spatial.transform import Rotation as R


# from view_suite.scannet.utils.pose_utils import (
#     extrinsic_c2w_to_w2c,
#     extrinsic_w2c_to_c2w,
#     c2w_extrinsic_to_se3,
#     w2c_extrinsic_to_se3,
#     check_4x4,
#     assert_rotation,
# )

# ================================
# Matrix <-> Matrix conversions
# ================================
def extrinsic_c2w_to_w2c(camera_to_world: np.ndarray) -> np.ndarray:
    """
    Convert a camera-to-world (c2w) homogeneous matrix into a world-to-camera (w2c) extrinsic.
    Args:
        camera_to_world: (4,4) camera-to-world matrix
    Returns:
        (4,4) world-to-camera matrix
    """
    check_4x4(camera_to_world, name="camera_to_world")
    return np.linalg.inv(camera_to_world)


def extrinsic_w2c_to_c2w(world_to_camera: np.ndarray) -> np.ndarray:
    """
    Convert a world-to-camera (w2c) extrinsic into a camera-to-world (c2w) homogeneous matrix.
    Args:
        world_to_camera: (4,4) world-to-camera matrix
    Returns:
        (4,4) camera-to-world matrix
    """
    check_4x4(world_to_camera, name="world_to_camera")
    return np.linalg.inv(world_to_camera)


def check_4x4(M: np.ndarray, name: str = "matrix"):
    """Validate a (4,4) homogeneous matrix."""
    if not isinstance(M, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, got {type(M)}")
    if M.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4,4), got {M.shape}")


def assert_rotation(Rm: np.ndarray, name: str = "R", atol: float = 1e-6):
    """Basic orthonormality check for a rotation matrix."""
    if Rm.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3,3), got {Rm.shape}")
    should_be_I = Rm.T @ Rm
    if not np.allclose(should_be_I, np.eye(3), atol=atol):
        raise ValueError(f"{name} is not orthonormal within atol={atol}")
    if not np.isclose(np.linalg.det(Rm), 1.0, atol=atol):
        raise ValueError(f"{name} must have det=1, got det={np.linalg.det(Rm)}")

class ViewManipulator:
    """
    Camera pose controller for InteriorGS/ScanNet-style environments.
    Keeps the canonical state as camera-to-world (c2w).
    Movement is along the camera's own axes.
    """

    def __init__(
        self,
        step_translation: float = 0.3,
        step_rotation_deg: float = 30.0,
        # **保持 'Z' 作为默认值，以兼容大多数 ScanNet/Habitat 设置**
        world_up_axis: str = "Z", 
        is_discrete: bool = False,
        # 假设相机遵循 OpenCV/PyTorch3D 惯例 (Y轴向下)
        image_y_down: bool = True, 
    ):
        """
        Args:
            step_translation: translation step length in world units.
            step_rotation_deg: rotation step size in degrees.
            world_up_axis: 'Z' (ScanNet-style) or 'Y'.
            is_discrete: snap rotations to multiples of step_rotation_deg (on c2w).
            image_y_down: if True, screen-up = camera (0,-1,0); else (0,+1,0).
        """
        self.step_t = float(step_translation)
        self.step_r_deg = float(step_rotation_deg)
        self.step_r = np.radians(self.step_r_deg)
        self.up_axis = world_up_axis.upper()
        assert self.up_axis in ("Z", "Y"), "world_up_axis must be 'Z' or 'Y'"
        self.is_discrete = bool(is_discrete)
        self.image_y_down = bool(image_y_down)

        # Canonical pose: camera-to-world (c2w)
        self.c2w = np.eye(4, dtype=np.float64)

    # -------------------------------------------------------------------------
    # Initialization / getters / setters
    # -------------------------------------------------------------------------
    def reset(self, initial_extrinsic_c2w: np.ndarray | None = None) -> np.ndarray:
        """
        Reset to identity or provided camera-to-world (4x4). 
        In discrete mode, the rotation is snapped.
        """
        if initial_extrinsic_c2w is None:
            self.c2w = np.eye(4, dtype=np.float64)
        else:
            check_4x4(initial_extrinsic_c2w)
            self.c2w = initial_extrinsic_c2w.astype(np.float64)
        if self.is_discrete:
            self._snap_rotation_in_place()
        return self.get_pose(mode="c2w")

    def get_pose(self, mode: str = "c2w") -> np.ndarray:
        """
        Return current extrinsic:
          - mode='c2w': camera-to-world (4x4)
          - mode='w2c': world-to-camera (4x4)
        """
        if mode == "c2w":
            return self.c2w.copy()
        elif mode == "w2c":
            return extrinsic_c2w_to_w2c(self.c2w)
        else:
            raise ValueError("mode must be 'c2w' or 'w2c'")

    # -------------------------------------------------------------------------
    # Discrete action API
    # -------------------------------------------------------------------------
    def step(self, action: str) -> np.ndarray:
        """Executes a discrete movement action."""
        a = action.strip().lower()
        if   a == "w": self.move_forward(+self.step_t)
        elif a == "s": self.move_forward(-self.step_t)
        elif a == "a": self.move_right(-self.step_t)
        elif a == "d": self.move_right(+self.step_t)
        elif a == "y": self.move_screen_up(+self.step_t)
        elif a == "h": self.move_screen_up(-self.step_t)
        elif a == "q": self.yaw_camera(-self.step_r)  # turn left (local +Y)
        elif a == "e": self.yaw_camera(+self.step_r)  # turn right
        elif a == "r":
            # look up
            ang = (+self.step_r) if self.image_y_down else (-self.step_r)
            self.pitch_camera(ang)
        elif a == "f":
            # look down
            ang = (-self.step_r) if self.image_y_down else (+self.step_r)
            self.pitch_camera(ang)
        elif a == "t":
            # on-screen CCW => camera roll negative
            self.roll_camera(-self.step_r)
        elif a == "g":
            # on-screen CW => camera roll positive
            self.roll_camera(+self.step_r)
        else:
            raise ValueError(f"Unsupported action: {action}")
        return self.get_pose(mode="c2w")


    # -------------------------------------------------------------------------
    # Movements (Translation)
    # -------------------------------------------------------------------------
    def move_forward(self, distance: float):
        """Translate along camera +Z by `distance` (world = c2w[:3,:3] @ [0,0,1])."""
        R_c2w, C_world = self._Rc_t_from_c2w(self.c2w)
        dir_world = R_c2w @ np.array([0.0, 0.0, 1.0])  # camera +Z in world
        self._translate_camera_center(C_world, R_c2w, dir_world * distance)

    def move_right(self, distance: float):
        """Translate along screen-right (camera +X) by `distance`."""
        R_c2w, C_world = self._Rc_t_from_c2w(self.c2w)
        dir_world = R_c2w @ np.array([1.0, 0.0, 0.0])  # camera +X in world
        self._translate_camera_center(C_world, R_c2w, dir_world * distance)

    def move_screen_up(self, distance: float):
        """Translate along screen-up (camera ±Y) by `distance`."""
        R_c2w, C_world = self._Rc_t_from_c2w(self.c2w)
        cam_up = np.array([0.0, -1.0, 0.0]) if self.image_y_down else np.array([0.0, 1.0, 0.0])
        dir_world = R_c2w @ cam_up
        self._translate_camera_center(C_world, R_c2w, dir_world * distance)

    # -------------------------------------------------------------------------
    # Rotations (about camera center)
    # -------------------------------------------------------------------------
    def yaw_camera(self, angle_rad: float):
        """Yaw around the camera's local +Y axis by angle_rad."""
        R_c2w, C_world = self._Rc_t_from_c2w(self.c2w)
        R_local = R.from_euler("y", angle_rad, degrees=False).as_matrix()
        R_new = R_c2w @ R_local
        if self.is_discrete:
            R_new = self._snap_rotation_matrix_c2w(R_new)
        self.c2w = self._compose_c2w(R_new, C_world)

    def pitch_camera(self, angle_rad: float):
        """Pitch around the camera's local +X axis by angle_rad."""
        R_c2w, C_world = self._Rc_t_from_c2w(self.c2w)
        R_local = R.from_euler("x", angle_rad, degrees=False).as_matrix()
        R_new = R_c2w @ R_local
        if self.is_discrete:
            R_new = self._snap_rotation_matrix_c2w(R_new)
        self.c2w = self._compose_c2w(R_new, C_world)

    def roll_camera(self, angle_rad: float):
        """Roll around the camera's local +Z axis by angle_rad."""
        R_c2w, C_world = self._Rc_t_from_c2w(self.c2w)
        R_local = R.from_euler("z", angle_rad, degrees=False).as_matrix()
        R_new = R_c2w @ R_local
        if self.is_discrete:
            R_new = self._snap_rotation_matrix_c2w(R_new)
        self.c2w = self._compose_c2w(R_new, C_world)


    # -------------------------------------------------------------------------
    # 6-DoF conversions (SE(3))
    # -------------------------------------------------------------------------
    def get_se3(self, degrees: bool = True) -> np.ndarray:
        """Return the camera-to-world pose as SE(3) = [cx, cy, cz, rx, ry, rz]."""
        R_c2w = self.c2w[:3, :3]
        C_world = self.c2w[:3, 3]
        eul = R.from_matrix(R_c2w).as_euler('xyz', degrees=degrees)
        return np.concatenate([C_world.astype(np.float64), eul.astype(np.float64)])

    def set_se3(self, pose6: np.ndarray, degrees: bool = True):
        """Set the camera-to-world pose from SE(3) = [cx, cy, cz, rx, ry, rz]."""
        pose6 = np.asarray(pose6, dtype=np.float64).reshape(-1)
        if pose6.shape[0] != 6:
            raise ValueError(f"pose6 must have shape (6,), got {pose6.shape}")
        C_world = pose6[:3]
        angles = pose6[3:]
        R_c2w = R.from_euler('xyz', angles, degrees=degrees).as_matrix()
        
        if self.is_discrete:
            e = R.from_matrix(R_c2w).as_euler('xyz', degrees=False)
            e = self.step_r * np.round(e / self.step_r)
            R_c2w = R.from_euler('xyz', e, degrees=False).as_matrix()
            
        M = np.eye(4, dtype=np.float64)
        M[:3, :3] = R_c2w
        M[:3, 3] = C_world
        self.c2w = M


    # -------------------------------------------------------------------------
    # Internal helpers (Static and utilities)
    # -------------------------------------------------------------------------
    @staticmethod
    def _Rc_t_from_c2w(M: np.ndarray):
        """Extract (R_c2w, C_world) from a 4x4 c2w matrix."""
        check_4x4(M)
        R_c2w = M[:3, :3]
        t = M[:3, 3]
        assert_rotation(R_c2w)
        return R_c2w, t

    @staticmethod
    def _compose_c2w(R_c2w: np.ndarray, C_world: np.ndarray) -> np.ndarray:
        """Compose a 4x4 c2w from rotation and camera center."""
        assert_rotation(R_c2w)
        M = np.eye(4, dtype=np.float64)
        M[:3, :3] = R_c2w
        M[:3, 3] = C_world.astype(np.float64)
        return M

    def _translate_camera_center(self, C_world: np.ndarray, R_c2w: np.ndarray, delta_world: np.ndarray):
        """Move camera center by delta in world coordinates, preserving orientation."""
        C_new = C_world + delta_world
        self.c2w = self._compose_c2w(R_c2w, C_new)

    # ----- discrete snapping on c2w -----
    def _snap_angles(self, eul_xyz_rad: np.ndarray) -> np.ndarray:
        """Snap each Euler angle (rad) to nearest multiple of step_r."""
        return self.step_r * np.round(eul_xyz_rad / self.step_r)

    def _snap_rotation_matrix_c2w(self, R_c2w: np.ndarray) -> np.ndarray:
        """Snap a c2w rotation matrix via Euler 'xyz' rounding."""
        e = R.from_matrix(R_c2w).as_euler('xyz', degrees=False)
        e = self._snap_angles(e)
        return R.from_euler('xyz', e, degrees=False).as_matrix()

    def _snap_rotation_in_place(self):
        """Snap current rotation (on c2w) while preserving camera center."""
        R_c2w, C_world = self._Rc_t_from_c2w(self.c2w)
        R_snapped = self._snap_rotation_matrix_c2w(R_c2w)
        self.c2w = self._compose_c2w(R_snapped, C_world)