import torch


def init_edge_rot_mat(edge_distance_vec):
    edge_vec_0 = edge_distance_vec
    edge_vec_0_distance = torch.sqrt(torch.sum(edge_vec_0**2, dim=1))

    # Make sure the atoms are far enough apart
    #assert torch.min(edge_vec_0_distance) < 0.0001
    
    # 检查是否为空张量
    if edge_vec_0_distance.numel() > 0:
        if torch.min(edge_vec_0_distance) < 0.0001:
            print(
                "Error edge_vec_0_distance: {}".format(
                    torch.min(edge_vec_0_distance)
                )
            )
    else:
        # 如果是空张量，直接返回空的旋转矩阵
        return torch.empty(0, 3, 3, device=edge_distance_vec.device, dtype=edge_distance_vec.dtype)
        
    norm_x = edge_vec_0 / (edge_vec_0_distance.view(-1, 1))
    
    edge_vec_2 = torch.rand_like(edge_vec_0) - 0.5
    edge_vec_2 = edge_vec_2 / (
        torch.sqrt(torch.sum(edge_vec_2**2, dim=1)).view(-1, 1)
    )
    # Create two rotated copys of the random vectors in case the random vector is aligned with norm_x
    # With two 90 degree rotated vectors, at least one should not be aligned with norm_x
    edge_vec_2b = edge_vec_2.clone()
    edge_vec_2b[:, 0] = -edge_vec_2[:, 1]
    edge_vec_2b[:, 1] = edge_vec_2[:, 0]
    edge_vec_2c = edge_vec_2.clone()
    edge_vec_2c[:, 1] = -edge_vec_2[:, 2]
    edge_vec_2c[:, 2] = edge_vec_2[:, 1]
    vec_dot_b = torch.abs(torch.sum(edge_vec_2b * norm_x, dim=1)).view(
        -1, 1
    )
    vec_dot_c = torch.abs(torch.sum(edge_vec_2c * norm_x, dim=1)).view(
        -1, 1
    )

    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
    edge_vec_2 = torch.where(
        torch.gt(vec_dot, vec_dot_b), edge_vec_2b, edge_vec_2
    )
    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
    edge_vec_2 = torch.where(
        torch.gt(vec_dot, vec_dot_c), edge_vec_2c, edge_vec_2
    )

    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1))
    
    # Check the vectors aren't aligned, if they are, add small perturbation
    max_dot = torch.max(vec_dot)
    if max_dot >= 0.99:
        # 添加小的随机扰动来避免向量过于对齐
        perturbation = torch.randn_like(edge_vec_2) * 0.01
        edge_vec_2 = edge_vec_2 + perturbation
        edge_vec_2 = edge_vec_2 / (torch.sqrt(torch.sum(edge_vec_2**2, dim=1, keepdim=True)))
        vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1))
        max_dot = torch.max(vec_dot)
        
        # 如果扰动后仍然对齐，使用备用向量
        if max_dot >= 0.99:
            edge_vec_2 = torch.cross(norm_x, torch.tensor([[1.0, 0.0, 0.0]], device=norm_x.device).expand_as(norm_x), dim=1)
            edge_vec_2_norm = torch.sqrt(torch.sum(edge_vec_2**2, dim=1, keepdim=True))
            # 如果叉积结果为零向量，使用另一个方向
            zero_mask = edge_vec_2_norm.squeeze() < 1e-6
            if zero_mask.any():
                edge_vec_2[zero_mask] = torch.cross(norm_x[zero_mask], torch.tensor([[0.0, 1.0, 0.0]], device=norm_x.device).expand_as(norm_x[zero_mask]), dim=1)
                edge_vec_2_norm[zero_mask] = torch.sqrt(torch.sum(edge_vec_2[zero_mask]**2, dim=1, keepdim=True))
            edge_vec_2 = edge_vec_2 / edge_vec_2_norm

    norm_z = torch.cross(norm_x, edge_vec_2, dim=1)
    norm_z = norm_z / (
        torch.sqrt(torch.sum(norm_z**2, dim=1, keepdim=True))
    )
    norm_z = norm_z / (
        torch.sqrt(torch.sum(norm_z**2, dim=1)).view(-1, 1)
    )
    norm_y = torch.cross(norm_x, norm_z, dim=1)
    norm_y = norm_y / (
        torch.sqrt(torch.sum(norm_y**2, dim=1, keepdim=True))
    )

    # Construct the 3D rotation matrix
    norm_x = norm_x.view(-1, 3, 1)
    norm_y = -norm_y.view(-1, 3, 1)
    norm_z = norm_z.view(-1, 3, 1)

    edge_rot_mat_inv = torch.cat([norm_z, norm_x, norm_y], dim=2)
    edge_rot_mat = torch.transpose(edge_rot_mat_inv, 1, 2)

    return edge_rot_mat.detach()