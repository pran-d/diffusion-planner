import torch

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


def guidance_goal_mse(xk, pred_x0, alpha_cumprod, goal):
    """
    Mean Squared Error guidance towards a specific goal state at the final timestep.
    
    Args:
        xk: (B, T, D) - current noisy sample (unused)
        pred_x0: (B, T, D) - predicted clean sample
        alpha_cumprod: (B, T, 1) - alpha cumulative product (unused)
        goal: (D_sub,) - target values for the specific indices (tensor)
        indices: (D_sub,) or List[int] - indices of the feature dimension to apply loss on (tensor or list)
    """
    # pred_x0: (B, T, D)
    # Select last timestep
    pred_final = pred_x0[:, -1, :] # (B, D)
    
    # Select target features
    indices = torch.tensor([39, 40])
    pred_final = pred_final[:, indices]
    
    # Expand goal to batch
    while goal.dim() < pred_final.dim():
        goal = goal.unsqueeze(0)
    target = goal.expand_as(pred_final)
    
    return ((pred_final - target) ** 2).mean()
