def guidance_smoothness(xk, pred_x0, alpha_cumprod):
    """
    Smoothness guidance: penalize jumps between consecutive timesteps.

    Args:
        xk:          (B, T, D) noisy trajectory at step k (unused here)
        pred_x0:     (B, T, D) predicted clean trajectory
        alpha_cumprod: scalar or (B,) tensor (unused here)

    Returns:
        scalar guidance loss
    """
    pos_scaling = 1.0
    joint_scaling = 0.2
    
    # First-order temporal differences
    loss = ((pos_scaling * (xk[:, 1:, :3] - xk[:, :-1, :3])) ** 2).mean() + \
            ((joint_scaling * (xk[:, 1:, 7:-7] - xk[:, :-1, 7:-7])) ** 2).mean() + \
            ((pos_scaling * (xk[:, 1:, -7:-4] - xk[:, :-1, -7:-4])) ** 2).mean() 

    return loss
