"""
Math Utilities
============
"""

from typing import Tuple

import numpy as np

# ============================================================================
# Quaternion and Rotation Matrix Conversions
# ============================================================================

def batch_rotation(matrices: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Apply batch of rotation matrices to batch of vectors.

    Args:
        matrices: (B, 3, 3) Batch of rotation matrices
        vectors: (B, 3) Batch of vectors

    Returns:
        rotated_vectors: (B, 3) Batch of rotated vectors
    """
    if matrices.ndim - vectors.ndim == 1:
        return (matrices @ vectors[..., None]).squeeze(-1)
    elif matrices.ndim - vectors.ndim == -1:
        return (np.expand_dims(matrices, axis=1) @ vectors[..., None]).squeeze(-1)
    else:
        try:
            return np.einsum('...ij,...j->...i', matrices, vectors)
        except ValueError:
            raise ValueError(f"Cannot batch rotate with matrices shape {matrices.shape} and vector shape {vectors.shape}.")

def transpose(R: np.ndarray) -> np.ndarray:
    """Transpose rotation matrix or batch of rotation matrices."""
    return R.swapaxes(-1, -2)

def to_xyzw(quat_wxyz):
    """Convert wxyz to xyzw."""
    return quat_wxyz[..., [1, 2, 3, 0]]

def to_wxyz(quat_xyzw):
    """Convert xyzw to wxyz."""
    return quat_xyzw[..., [3, 0, 1, 2]]


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to rotation matrix (3, 3).

    Supports both single quaternion (1D array) and batch of quaternions (any shape ending in 4).
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    xw = x * w
    yw = y * w
    zw = z * w
    
    row0 = np.stack([1 - 2 * (yy + zz), 2 * (xy - zw), 2 * (xz + yw)], axis=-1)
    row1 = np.stack([2 * (xy + zw), 1 - 2 * (xx + zz), 2 * (yz - xw)], axis=-1)
    row2 = np.stack([2 * (xz - yw), 2 * (yz + xw), 1 - 2 * (xx + yy)], axis=-1)
    
    return np.stack([row0, row1, row2], axis=-2)


def rot_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to quaternion (w, x, y, z).

    Supports batched inputs of shape (..., 3, 3)
    Returns (..., 4)
    """
    R = np.asarray(R)
    assert R.shape[-2:] == (3, 3)

    m00 = R[..., 0, 0]
    m11 = R[..., 1, 1]
    m22 = R[..., 2, 2]

    trace = m00 + m11 + m22

    q = np.empty(R.shape[:-2] + (4,), dtype=R.dtype)

    # Case 1: trace > 0
    mask = trace > 0
    s = np.zeros_like(trace)
    s[mask] = 0.5 / np.sqrt(trace[mask] + 1.0)

    q[..., 0][mask] = 0.25 / s[mask]
    q[..., 1][mask] = (R[..., 2, 1][mask] - R[..., 1, 2][mask]) * s[mask]
    q[..., 2][mask] = (R[..., 0, 2][mask] - R[..., 2, 0][mask]) * s[mask]
    q[..., 3][mask] = (R[..., 1, 0][mask] - R[..., 0, 1][mask]) * s[mask]

    # Case 2: m00 is largest
    mask2 = (~mask) & (m00 > m11) & (m00 > m22)
    s[mask2] = 2.0 * np.sqrt(1.0 + m00[mask2] - m11[mask2] - m22[mask2])

    q[..., 0][mask2] = (R[..., 2, 1][mask2] - R[..., 1, 2][mask2]) / s[mask2]
    q[..., 1][mask2] = 0.25 * s[mask2]
    q[..., 2][mask2] = (R[..., 0, 1][mask2] + R[..., 1, 0][mask2]) / s[mask2]
    q[..., 3][mask2] = (R[..., 0, 2][mask2] + R[..., 2, 0][mask2]) / s[mask2]

    # Case 3: m11 is largest
    mask3 = (~mask) & (~mask2) & (m11 > m22)
    s[mask3] = 2.0 * np.sqrt(1.0 + m11[mask3] - m00[mask3] - m22[mask3])

    q[..., 0][mask3] = (R[..., 0, 2][mask3] - R[..., 2, 0][mask3]) / s[mask3]
    q[..., 1][mask3] = (R[..., 0, 1][mask3] + R[..., 1, 0][mask3]) / s[mask3]
    q[..., 2][mask3] = 0.25 * s[mask3]
    q[..., 3][mask3] = (R[..., 1, 2][mask3] + R[..., 2, 1][mask3]) / s[mask3]

    # Case 4: m22 is largest
    mask4 = (~mask) & (~mask2) & (~mask3)
    s[mask4] = 2.0 * np.sqrt(1.0 + m22[mask4] - m00[mask4] - m11[mask4])

    q[..., 0][mask4] = (R[..., 1, 0][mask4] - R[..., 0, 1][mask4]) / s[mask4]
    q[..., 1][mask4] = (R[..., 0, 2][mask4] + R[..., 2, 0][mask4]) / s[mask4]
    q[..., 2][mask4] = (R[..., 1, 2][mask4] + R[..., 2, 1][mask4]) / s[mask4]
    q[..., 3][mask4] = 0.25 * s[mask4]

    return q



# Alias for backward compatibility
rot_to_quat = rot_to_quat_wxyz


# ============================================================================
# 6D Rotation Representation
# ============================================================================


def rot_to_6d(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to 6D representation (first two columns).

    Supports both single rotation matrix (2D array) and batch of matrices (3D array).
    """
    return np.concatenate([R[..., 0], R[..., 1]], axis=-1)


def rot6d_to_rot(r6: np.ndarray) -> np.ndarray:
    """Convert 6D representation back to rotation matrix (Gram-Schmidt)."""
    if r6.ndim == 1:
        a1 = r6[:3]
        a2 = r6[3:]
    else:
        a1 = r6[..., :3]
        a2 = r6[..., 3:]
        
    r1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    r2_tilde = a2 - np.sum(r1 * a2, axis=-1, keepdims=True) * r1
    r2 = r2_tilde / (np.linalg.norm(r2_tilde, axis=-1, keepdims=True) + 1e-8)
    r3 = np.cross(r1, r2, axis=-1)
    
    return np.stack([r1, r2, r3], axis=-1)


# ============================================================================
# Yaw Angle Utilities
# ============================================================================


def yaw_from_quat(q: np.ndarray) -> np.ndarray:
    """Extract yaw angle from quaternion (wxyz format).
    
    Args:
        q: (4,) or (N, 4) quaternion
        
    Returns:
        yaw: float or (N,) array
    """
    if q.ndim == 1:
        w, x, y, z = q
    else:
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)


def yaw_to_rot_matrix(yaw: float) -> np.ndarray:
    """Create rotation matrix for rotation around Z axis.
    
    Args:
        yaw: float or (N,) array
        
    Returns:
        R: (3, 3) or (N, 3, 3) rotation matrix
    """
    c, s = np.cos(yaw), np.sin(yaw)
    
    if np.ndim(yaw) == 0:
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    
    zeros = np.zeros_like(yaw)
    ones = np.ones_like(yaw)
    
    # Stack along the last two dimensions to form (N, 3, 3) or (..., 3, 3)
    row1 = np.stack([c, -s, zeros], axis=-1)
    row2 = np.stack([s, c, zeros], axis=-1)
    row3 = np.stack([zeros, zeros, ones], axis=-1)
    
    return np.stack([row1, row2, row3], axis=-2)


def remove_yaw_from_rot(R: np.ndarray) -> np.ndarray:
    """Remove yaw component from rotation matrix.
    
    Args:
        R: (3, 3) or (N, 3, 3) rotation matrix
        
    Returns:
        R_no_yaw: (3, 3) or (N, 3, 3) rotation matrix
    """
    if R.ndim == 2:
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        yaw = np.arctan2(R[..., 1, 0], R[..., 0, 0])
        
    R_yaw_inv = yaw_to_rot_matrix(-yaw)
    return R_yaw_inv @ R


# ============================================================================
# Angle Utilities
# ============================================================================

def normalize_angle(angle: np.ndarray) -> np.ndarray:
    """Normalize angle to [-pi, pi].
    
    Args:
        angle: float or array of angles
        
    Returns:
        normalized_angle: float or array of normalized angles
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


# ============================================================================
# Direction Utilities
# ============================================================================


def direction_to_quat_wxyz(direction: np.ndarray) -> np.ndarray:
    """Convert a direction vector to a quaternion that aligns Z-axis with the direction."""
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        return np.array([1, 0, 0, 0], dtype=np.float64)
    direction = direction / norm

    up = np.array([0, 0, 1])
    if abs(np.dot(direction, up)) > 0.99:
        up = np.array([1, 0, 0])

    z_axis = direction
    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)
    y_axis = np.cross(z_axis, x_axis)

    R = np.column_stack([x_axis, y_axis, z_axis])
    return rot_to_quat_wxyz(R)


# ============================================================================
# SE(3) Utilities
# ============================================================================


def compute_relative_se3(
    pos_a: np.ndarray,
    quat_a: np.ndarray,
    pos_b: np.ndarray,
    quat_b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute SE(3) of B relative to A.

    Args:
        pos_a: Position of frame A (3,)
        quat_a: Quaternion of frame A (w, x, y, z)
        pos_b: Position of frame B (3,)
        quat_b: Quaternion of frame B (w, x, y, z)

    Returns:
        rel_pos: Position of B relative to A (3,)
        rel_rot: Rotation matrix of B relative to A (3, 3)
    """
    R_a = quat_to_rot(quat_a)
    R_b = quat_to_rot(quat_b)
    rel_pos = (transpose(R_a) @ (pos_b - pos_a)[..., None]).squeeze(-1)
    rel_rot = transpose(R_a) @ R_b
    return rel_pos, rel_rot

def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Hamilton product of two quaternions.
    Both q1, q2 are [qw, qx, qy, qz].
    """    
    w1, x1, y1, z1 = q1[...,0], q1[...,1], q1[...,2], q1[...,3]
    w2, x2, y2, z2 = q2[...,0], q2[...,1], q2[...,2], q2[...,3]
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])