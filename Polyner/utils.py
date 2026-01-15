import numpy as np

def cone_beam_ray(proj_pos_u, proj_pos_v, SOD):
    """
    Generate cone beam ray sampling point coordinates
    
    Parameters:
    -----------
    proj_pos_u : array-like
        Horizontal detector angles (degrees), shape: (num_det_u,)
    proj_pos_v : array-like
        Vertical detector angles (degrees), shape: (num_det_v,)
    SOD : float
        Source-to-object distance (used to determine number of sampling points)
    
    Returns:
    --------
    xyz : ndarray
        Ray sampling point coordinates, shape: (num_det_u, num_det_v, 2*SOD, 3)
    """
    # Source position (normalized coordinates)
    origin_x = 0
    origin_y = -1
    origin_z = 0
    
    # Reference ray: along y-axis from -1 to 1
    num_samples = int(2 * SOD) #训练时三个方向的采样点都是2*SOD，test（输出）时也要保持一致
    y = np.linspace(-1, 1, num_samples).reshape(-1, 1)  # (num_samples, 1)
    x = np.zeros_like(y)
    z = np.zeros_like(y)
    
    # Homogeneous coordinates: (x, y, z, 1)
    xyz_temp = np.concatenate([x, y, z, np.ones_like(x)], axis=-1)  # (num_samples, 4)
    
    num_det_u = len(proj_pos_u)
    num_det_v = len(proj_pos_v)
    
    # Output array: (num_det_u, num_det_v, num_samples, 3)
    xyz = np.zeros(shape=(num_det_u, num_det_v, num_samples, 3))
    
    for i in range(num_det_u):
        for j in range(num_det_v):
            # Get rotation angles
            theta_u = np.deg2rad(proj_pos_u[num_det_u - i - 1])  # horizontal angle
            theta_v = np.deg2rad(proj_pos_v[num_det_v - j - 1])  # vertical angle
            
            # Build 3D rotation transformation matrix M = T2 @ Rv @ Ru @ T1
            M = build_3d_rotation_matrix(theta_u, theta_v, origin_x, origin_y, origin_z)
            
            # Transform sampling points
            temp = xyz_temp @ M.T  # (num_samples, 4) @ (4, 4) -> (num_samples, 4)
            xyz[i, j, :, :] = temp[:, :3]  # take first 3 columns (x, y, z)
    
    return xyz


def build_3d_rotation_matrix(theta_u, theta_v, ox, oy, oz):
    """
    Build 3D rotation matrix around arbitrary point (ox, oy, oz)
    First rotate around z-axis by theta_u, then around x-axis by theta_v
    
    M = T2 @ Rv @ Ru @ T1
    
    Parameters:
    -----------
    theta_u : float
        Rotation angle around z-axis (radians)
    theta_v : float
        Rotation angle around x-axis (radians)
    ox, oy, oz : float
        Rotation center coordinates
    
    Returns:
    --------
    M : ndarray
        4x4 homogeneous transformation matrix
    """
    cos_u, sin_u = np.cos(theta_u), np.sin(theta_u)
    cos_v, sin_v = np.cos(theta_v), np.sin(theta_v)
    
    # Ru: rotation around z-axis
    Ru = np.array([
        [cos_u, -sin_u, 0, 0],
        [sin_u,  cos_u, 0, 0],
        [0,      0,     1, 0],
        [0,      0,     0, 1]
    ])
    
    # Rv: rotation around x-axis
    Rv = np.array([
        [1, 0,      0,     0],
        [0, cos_v, -sin_v, 0],
        [0, sin_v,  cos_v, 0],
        [0, 0,      0,     1]
    ])
    
    # T1: translate to origin
    T1 = np.array([
        [1, 0, 0, -ox],
        [0, 1, 0, -oy],
        [0, 0, 1, -oz],
        [0, 0, 0,  1]
    ])
    
    # T2: translate back
    T2 = np.array([
        [1, 0, 0, ox],
        [0, 1, 0, oy],
        [0, 0, 1, oz],
        [0, 0, 0, 1]
    ])
    
    # Composite transformation: M = T2 @ Rv @ Ru @ T1
    M = T2 @ Rv @ Ru @ T1
    
    return M


def grid_coordinate_3d(h, w, d):
    """
    Generate 3D grid coordinates
    
    Parameters:
    -----------
    h, w, d : int
        Dimensions along three axes
    
    Returns:
    --------
    xyz : ndarray, shape (h*w*d, 3)
        Flattened 3D coordinates in range [-1, 1]
    """
    x = np.linspace(-1, 1, h)
    y = np.linspace(-1, 1, w)
    z = np.linspace(-1, 1, d)
    x, y, z = np.meshgrid(x, y, z, indexing='ij')  # (h, w, d) each
    xyz = np.stack([x, y, z], axis=-1).reshape(-1, 3)  # (h*w*d, 3)
    return xyz


def rotate_ray_3d(xyz, angle):
    """
    Rotate 3D coordinates around z-axis (gantry rotation for CT projection views)
    
    Parameters:
    -----------
    xyz : ndarray
        Input coordinates, shape (..., 3)
    angle : float
        Gantry rotation angle around z-axis (degrees) - projection view angle
    
    Returns:
    --------
    xyz_rotated : ndarray
        Rotated coordinates, same shape as input
    """
    xyz_shape = xyz.shape
    xyz = xyz.reshape(-1, 3)
    
    angle_rad = np.deg2rad(angle)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    
    # Rotation around z-axis (gantry rotation)
    R = np.array([
        [cos_a, -sin_a, 0],
        [sin_a,  cos_a, 0],
        [0,      0,     1]
    ])
    
    xyz_rotated = (xyz @ R.T).reshape(xyz_shape)
    return xyz_rotated
