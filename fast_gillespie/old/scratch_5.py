import torch
import networkx as nx
import numpy as np
import random

def custom_energy(pos_tensor, edge_index, fixed_index, k):
    # Repulsive forces
    diff = pos_tensor.unsqueeze(0) - pos_tensor.unsqueeze(1)
    distance = torch.norm(diff, dim=2)
    distance = torch.clamp(distance, min=0.01)  # Avoid division by zero
    repulsive = torch.sum(k / distance)

    # Attractive forces
    start, end = edge_index
    edge_diff = pos_tensor[start] - pos_tensor[end]
    attractive = 0.5 * k * torch.sum(torch.norm(edge_diff, dim=1) ** 2)

    # Fix fixed_index to 0
    fixed_node_energy = 20 * torch.norm(pos_tensor[fixed_index, :])

    return repulsive + attractive + fixed_node_energy


def jbk_energy(pos_tensor, edge_index, fixed_index,
               mean_edge_length=1.0,
               repulsion_energy_scale=3.0,
               boundary_radius=10,
               vertical_center_energy_scale=3.0,
               drift_energy_scale=10.0,
               edge_energy_scale=100.0,
               fixed_node_energy_scale=100.0,
               boundary_energy_scale=100.0,
               debug=False):

    # Term draws the fixed index to zero
    fixed_node_energy = fixed_node_energy_scale * torch.sum(torch.abs(pos_tensor[fixed_index, :]))

    # Term pushes all nodes to the right
    drift_energy = -drift_energy_scale * torch.sum(pos_tensor[:, 0] )

    # Center vertically
    vertical_center_energy = vertical_center_energy_scale * torch.sum( torch.abs(pos_tensor[:, 1])**2 )

    # This seems to be working
    start, end = edge_index
    edge_diff = pos_tensor[start] - pos_tensor[end]
    edge_energy = edge_energy_scale * torch.sum(
        torch.abs(edge_diff - mean_edge_length))

    # Compute pairwise distances and corresponding energies
    diffs = pos_tensor[:, None, :] - pos_tensor[None, :, :]
    distances = torch.norm(diffs, dim=2)
    distances = torch.clamp(distances, min=0.01)  # Avoid division by zero
    repulsion_energy = repulsion_energy_scale * torch.sum(1 / distances)

    # Enforce boundary
    boundary_energy = boundary_energy_scale * torch.sum(torch.norm(pos_tensor, dim=1) > boundary_radius)

    if debug:
        print('debugging')

    return fixed_node_energy + drift_energy + boundary_energy + edge_energy + repulsion_energy + vertical_center_energy


def custom_layout_pytorch(G, k=1.0, iterations=100, learning_rate=0.1, pos_dict_init=None):
    # Convert graph to edge index
    edge_index = torch.tensor([[u, v] for u, v in G.edges()]).t().to(torch.long)

    # Initialize positions, with a a randomly chosen node fixed at position 0,0
    fixed_node = random.choice(list(G.nodes))
    fixed_index = list(G.nodes).index(fixed_node)
    print(f'fixed node = {fixed_node}')

    if pos_dict_init is None:
        pos_dict = nx.spring_layout(G, pos={fixed_node: (0, 0)}, fixed=[fixed_node])
        debug=False
    else:
        pos_dict = pos_dict_init
        debug=True

    pos_tensor = torch.tensor(
        [[pos_dict[n][0], pos_dict[n][1]] for n in G.nodes()],
        requires_grad=True, dtype=torch.float)

    optimizer = torch.optim.Adam([pos_tensor], lr=learning_rate)

    for _ in range(iterations):
        optimizer.zero_grad()
        # energy = custom_energy(pos_tensor, edge_index, fixed_index, k)
        energy = jbk_energy(pos_tensor=pos_tensor,
                            edge_index=edge_index,
                            fixed_index=fixed_index,
                            debug=debug)
        if debug:
            break;
        energy.backward()
        optimizer.step()

    # Convert back to dictionary
    return {n: pos_tensor[i].detach().numpy() for i, n in enumerate(G.nodes())}
