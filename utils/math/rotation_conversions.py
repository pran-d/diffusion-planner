import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

# batch*n
def normalize_vector( v):
    batch=v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))# batch
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).to(v.device)))
    v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
    v = v/v_mag
    return v
    
# u, v batch*n
def cross_product( u, v):
    batch = u.shape[0]
    #print (u.shape)
    #print (v.shape)
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
        
    out = torch.cat((i.view(batch,1), j.view(batch,1), k.view(batch,1)),1)#batch*3
        
    return out
        
#poses batch*6
def compute_rotation_matrix_from_ortho6d(poses):
    x_raw = poses[:,0:3]#batch*3
    y_raw = poses[:,3:6]#batch*3
        
    x = normalize_vector(x_raw) #batch*3
    z = cross_product(x,y_raw) #batch*3
    z = normalize_vector(z)#batch*3
    y = cross_product(z,x)#batch*3
        
    x = x.view(-1,3,1)
    y = y.view(-1,3,1)
    z = z.view(-1,3,1)
    matrix = torch.cat((x,y,z), 2) #batch*3*3
    return matrix


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation to 3x3 rotation matrix.
    Wrapper around RotationContinuity/sanity_test/code/tools.py
    Input:
        d6: (B, 6) Batch of 6D rotation representations
    Output:
        matrix: (B, 3, 3) Batch of rotation matrices
    """
    # The tools.py implementation expects (Batch, 6)
    # If input has more dimensions (e.g. B, T, 6), flatten and reshape
    original_shape = d6.shape
    if len(original_shape) > 2:
        d6_flat = d6.reshape(-1, 6)
        matrix_flat = compute_rotation_matrix_from_ortho6d(d6_flat)
        return matrix_flat.reshape(original_shape[:-1] + (3, 3))
    else:
        return compute_rotation_matrix_from_ortho6d(d6)

def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts 3x3 rotation matrix to 6D rotation representation.
    Input:
        matrix: (B, 3, 3) Batch of rotation matrices
    Output:
        d6: (B, 6) Batch of 6D rotation representations
    """
    # Extract first two columns
    # matrix[..., :, 0] is the first column (B, 3)
    # matrix[..., :, 1] is the second column (B, 3)
    return torch.cat([matrix[..., :, 0], matrix[..., :, 1]], dim=-1)

def quaternion_to_rotation_6d(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to 6D rotation representation.
    Args:
        quaternions: quaternions with real part first, as tensor of shape (..., 4).
    Returns:
        Batch of 6D rotation representations, as tensor of shape (..., 6).
    """
    return matrix_to_rotation_6d(quaternion_to_matrix(quaternions))

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:

    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first, as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.
    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m = matrix.view(batch_dim + (9,))

    q = torch.empty(batch_dim + (4,), dtype=matrix.dtype, device=matrix.device)

    # Algorithm from http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
    # Adapted for batch processing
    
    # Trace
    trace = m[..., 0] + m[..., 4] + m[..., 8]
    
    # Case 1: Trace > 0
    mask_pos = trace > 0
    if mask_pos.any():
        s = torch.sqrt(trace[mask_pos] + 1.0) * 2
        inv_s = 1.0 / s
        q[mask_pos, 0] = 0.25 * s
        q[mask_pos, 1] = (m[mask_pos, 7] - m[mask_pos, 5]) * inv_s
        q[mask_pos, 2] = (m[mask_pos, 2] - m[mask_pos, 6]) * inv_s
        q[mask_pos, 3] = (m[mask_pos, 3] - m[mask_pos, 1]) * inv_s

    # Case 2: Trace <= 0
    mask_neg = ~mask_pos
    if mask_neg.any():
        # Find major diagonal element
        m00 = m[mask_neg, 0]
        m11 = m[mask_neg, 4]
        m22 = m[mask_neg, 8]
        
        # Case 2a: m00 is largest
        mask_0 = (m00 > m11) & (m00 > m22)
        mask_0_global = mask_neg.clone()
        mask_0_global[mask_neg] = mask_0
        
        if mask_0_global.any():
            s = torch.sqrt(1.0 + m[mask_0_global, 0] - m[mask_0_global, 4] - m[mask_0_global, 8]) * 2
            inv_s = 1.0 / s
            q[mask_0_global, 0] = (m[mask_0_global, 7] - m[mask_0_global, 5]) * inv_s
            q[mask_0_global, 1] = 0.25 * s
            q[mask_0_global, 2] = (m[mask_0_global, 1] + m[mask_0_global, 3]) * inv_s
            q[mask_0_global, 3] = (m[mask_0_global, 2] + m[mask_0_global, 6]) * inv_s
            
        # Case 2b: m11 is largest
        mask_1 = (m11 > m00) & (m11 > m22)
        mask_1_global = mask_neg.clone()
        mask_1_global[mask_neg] = mask_1
        
        if mask_1_global.any():
            s = torch.sqrt(1.0 + m[mask_1_global, 4] - m[mask_1_global, 0] - m[mask_1_global, 8]) * 2
            inv_s = 1.0 / s
            q[mask_1_global, 0] = (m[mask_1_global, 2] - m[mask_1_global, 6]) * inv_s
            q[mask_1_global, 1] = (m[mask_1_global, 1] + m[mask_1_global, 3]) * inv_s
            q[mask_1_global, 2] = 0.25 * s
            q[mask_1_global, 3] = (m[mask_1_global, 5] + m[mask_1_global, 7]) * inv_s
            
        # Case 2c: m22 is largest
        mask_2 = (m22 > m00) & (m22 > m11)
        mask_2_global = mask_neg.clone()
        mask_2_global[mask_neg] = mask_2
        
        if mask_2_global.any():
            s = torch.sqrt(1.0 + m[mask_2_global, 8] - m[mask_2_global, 0] - m[mask_2_global, 4]) * 2
            inv_s = 1.0 / s
            q[mask_2_global, 0] = (m[mask_2_global, 3] - m[mask_2_global, 1]) * inv_s
            q[mask_2_global, 1] = (m[mask_2_global, 2] + m[mask_2_global, 6]) * inv_s
            q[mask_2_global, 2] = (m[mask_2_global, 5] + m[mask_2_global, 7]) * inv_s
            q[mask_2_global, 3] = 0.25 * s

    return q


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.
    Args:
        axis_angle: Rotations given as axis/angle, as tensor of shape (..., 3).
    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2. so sin(x/2)/x is about 1/2.
    sin_half_angles_over_angles[small_angles] = 0.5 - (angles[small_angles] * angles[small_angles]) / 48
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions


def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.
    Args:
        quaternions: quaternions with real part first, as tensor of shape (..., 4).
    Returns:
        Rotations given as axis/angle, as tensor of shape (..., 3).
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2. so sin(x/2)/x is about 1/2.
    sin_half_angles_over_angles[small_angles] = 0.5 - (angles[small_angles] * angles[small_angles]) / 48
    return quaternions[..., 1:] / sin_half_angles_over_angles


# Helpers
def transform_pos(r_inv, pos_seq, pos0):
    # Apply rotation to relative position
    return r_inv.apply(pos_seq - pos0)

def untransform_pos(r, pos_seq, pos0):
    # Apply inverse rotation to relative position
    return r.apply(pos_seq) + pos0
    
def transform_vec(r_inv, vec_seq):
    # Apply rotation only
    return r_inv.apply(vec_seq)
    
def transform_quat(r_inv, quat_seq_mj):
    # mj(wxyz) -> scipy(xyzw)
    q_scipy = quat_seq_mj[..., [1, 2, 3, 0]]
    r_curr = R.from_quat(q_scipy)
    # Rotate: R_new = R_inv * R_curr
    r_new = r_inv * r_curr
    # scipy(xyzw) -> mj(wxyz)
    q_new_scipy = r_new.as_quat()
    return q_new_scipy[..., [3, 0, 1, 2]]