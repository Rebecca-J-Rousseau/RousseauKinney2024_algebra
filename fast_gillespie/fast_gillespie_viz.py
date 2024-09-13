import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter
from networkx import connected_components
import random
import torch

def show_complex_sizes(nodes, edges, ax=None, figsize=(5,5)):
    # Store these in a directed graph
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Get dictionary listing connected components
    connected_components = list(nx.weakly_connected_components(G))
    component_sizes = [len(c) for c in nx.weakly_connected_components(G)]
    size_dict = {size: component_sizes.count(size) for size in
                 set(component_sizes)}

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    counts = size_dict.values()
    sizes = size_dict.keys()
    ax.loglog(sizes, counts, marker='o', linestyle='none')
    ax.set_xlabel('complex size')
    ax.set_ylabel('number of complexes of given size')

def find_peripheral_node(G):
    """
    This function takes a NetworkX graph G as input (either directed or undirected),
    and returns the name of a single peripheral node, which is the node with the maximum eccentricity.

    :param G: NetworkX graph (either directed or undirected)
    :return: Name of the peripheral node
    """
    # If the graph is directed, convert it to an undirected graph
    if G.is_directed():
        UG = G.to_undirected(as_view=True)
    else:
        UG = G

    if not nx.is_connected(UG):
        raise ValueError("The graph must be connected")

    # Compute the all-pairs shortest path lengths
    shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(UG))

    # Compute the eccentricity of each node
    node_eccentricity = nx.eccentricity(UG, sp=shortest_path_lengths)

    # Find the node with the maximum eccentricity
    peripheral_node = max(node_eccentricity, key=node_eccentricity.get)

    return peripheral_node


def pin_and_flow_potential(pos_tensor,
                           edge_index,
                           fixed_index,
                           fixed_location=(0,0),
                           edge_length=1.0,
                           repulsion_energy_scale=0.2,
                           drift_energy_scale=3.0,
                           edge_energy_scale=100.0,
                           fixed_node_energy_scale=100.0):
    # Get number of nodes
    num_nodes = pos_tensor.shape[0]

    # Term that pins the fixed index to at the origin
    fixed_location = torch.tensor(fixed_location, dtype=torch.float)
    fixed_node_displacement = pos_tensor[fixed_index,:] - fixed_location
    fixed_node_energy = fixed_node_energy_scale * num_nodes * torch.sum(
        torch.abs(fixed_node_displacement)**2)

    # Term that maintains a fixed distance between nodes
    if edge_index is not None:
        start, end = edge_index
        edge_diff = torch.norm(pos_tensor[start] - pos_tensor[end], dim=1)
        edge_energy = 0.5 * edge_energy_scale * torch.sum(
            torch.abs(edge_diff - edge_length) ** 2)
    else:
        edge_energy = 0.0

    # Term that causes nodes to repel one another
    diffs = pos_tensor[:, None, :] - pos_tensor[None, :, :]
    num_nodes = pos_tensor.shape[0]
    distances = torch.norm(diffs, dim=2)
    distances = torch.clamp(distances, min=0.01)  # Avoid division by zero
    repulsion_contributions = 0.5 * repulsion_energy_scale * edge_length * (
                1.0 - torch.eye(num_nodes)) / distances
    repulsion_energy = torch.sum(repulsion_contributions) / 2

    # Term that pushes all nodes to the right
    drift_energy = -drift_energy_scale * torch.sum(pos_tensor[:, 0])

    # Return sum of energy terms
    return fixed_node_energy + edge_energy + repulsion_energy + drift_energy


def pin_and_flow_layout(G,
                        fixed_node=None,
                        fixed_location=(0,0),
                        iterations=1000,
                        learning_rate=0.1,
                        center_complex=True):

    # Choose peripheral node as the node to pin
    if fixed_node is None:
        fixed_node = find_peripheral_node(G)
    fixed_index = list(G.nodes).index(fixed_node)

    # Convert graph to edge_index
    node_list = list(G.nodes())
    num_edges = len(G.edges())
    if num_edges > 0:
        edge_index = np.array(
            [[node_list.index(u), node_list.index(v)] for u, v in G.edges()])
        edge_index = torch.tensor(edge_index).t().to(torch.long)
    else:
        edge_index = None

    # Randomly initialize positions of nodes
    pos_dict = {node: np.random.randn(2) for node in G.nodes()}
    pos_tensor = torch.tensor(
        [[pos_dict[n][0], pos_dict[n][1]] for n in G.nodes()],
        requires_grad=True, dtype=torch.float)

    # Set optimizer
    optimizer = torch.optim.Adam([pos_tensor], lr=learning_rate)

    # Perform specified number of iterations
    for _ in range(iterations):
        # I guess this removes memory
        optimizer.zero_grad()

        # Compute energy
        energy = pin_and_flow_potential(pos_tensor=pos_tensor,
                                        edge_index=edge_index,
                                        fixed_index=fixed_index,
                                        fixed_location=fixed_location)

        # Perform backpropagation to compute gradient
        energy.backward()

        # Update pos_tensor based on gradient using optimizer
        optimizer.step()

    # If center, center pos_tensor on fixed_location
    pos_arr = pos_tensor.detach().numpy()
    if center_complex:
        complex_center = 0.5*(pos_arr.min(axis=0) + pos_arr.max(axis=0))
        pos_arr = pos_arr \
                  - complex_center[np.newaxis, :] \
                  + np.array(fixed_location)[np.newaxis, :]

    # Convert pos_Tensor back to dictionary
    pos_dict = {n: pos_arr[i,:] for i, n in enumerate(G.nodes())}
    return pos_dict


# # Visualize the complexes
# def visualize_complexes_old(nodes,
#                             edges,
#                             node_properties,
#                             edge_properties,
#                             min_component_length=1,
#                             max_component_length=np.inf,
#                             aspect_ratio=5,
#                             subplots_shape=(5,1),
#                             figsize=(8, 8)):
#     # Create a new graph
#     G = nx.DiGraph()
#     G.add_nodes_from(nodes)
#     G.add_edges_from(edges)
#
#     # Get weakly connected components
#     components = list(nx.weakly_connected_components(G))
#
#     # Get number to plot
#     num_subplots = subplots_shape[0]*subplots_shape[1]
#
#     # Filter for components within a specified size
#     components = [c for c in components if
#                   len(c) >= min_component_length and
#                   len(c) <= max_component_length]
#
#     # Limit the number of components if more than number of subplots
#     if num_subplots < len(components):
#         components = random.sample(components, num_subplots)
#     num_components = len(components)
#
#     # Define figure and axes
#     fig, axs = plt.subplots(subplots_shape[0], subplots_shape[1], figsize=figsize)
#
#     # Package axs into a 1d array
#     if num_subplots == 1:
#         axs = [axs]
#     else:
#         axs = axs.flatten()
#
#     # Plot each connected component
#     pos_arr = np.array([[0,0]])
#     for ax, component in zip(axs[:num_components], components):
#
#         # Get subgraph
#         subG = G.subgraph(component).copy()
#
#         # Set location at which to pin complex
#         pos = pin_and_flow_layout(subG,
#                                   fixed_location=(0,0),
#                                   iterations=1000,
#                                   learning_rate=0.1)
#
#         # Draw nodes
#         for node_class, props in node_properties.items():
#             node_list = [node for node, data in subG.nodes(data=True) if
#                          data['class'] == node_class]
#             nx.draw_networkx_nodes(subG, pos, nodelist=node_list, ax=ax, **props)
#             nx.draw_networkx_nodes(subG, pos, nodelist=node_list, ax=ax, **props)
#
#         # Draw edges
#         for edge_class, props in edge_properties.items():
#             edge_list = [(u, v) for (u, v, d) in subG.edges(data=True) if
#                          d['class'] == edge_class]
#             nx.draw_networkx_edges(subG, pos, edgelist=edge_list, ax=ax, **props)
#
#         this_pos_arr = np.array([p for p in pos.values()])
#         pos_arr = np.concatenate((pos_arr, this_pos_arr), axis=0)
#
#     # Determine shared x lims and y lims for all plots
#     xlim = [np.min(pos_arr[:, 0])-1, np.max(pos_arr[:, 0])+1]
#     xrange = xlim[1] - xlim[0]
#     yrange = xrange / aspect_ratio
#     ylim = [-yrange/2, yrange/2]
#
#     # Format each axes
#     for _, ax in enumerate(axs):
#
#         # If there is nothing to plot on ax, turn it off
#         if _ >= num_components:
#             ax.axis('off')
#             continue
#
#         # Set limits
#         ax.set_xlim(xlim)
#         ax.set_ylim(ylim)
#
#         # Set equal aspect
#         ax.set_aspect('equal')
#
#         # Draw gridlines
#         #ax.axhline(0, color='gray')
#         #ax.axvline(0, color='gray')
#
#         # Remove ticks
#         ax.set_xticks([])
#         ax.set_yticks([])
#         #ax.set_title(f'Component {k+1} of {num_components}', fontsize=16)
#
#     # Remove axis
#     #plt.axis('off')
#
#     # Show the plot
#     plt.tight_layout()
#     plt.show()
#
#

# Visualize the complexes
def visualize_complexes(nodes,
                        edges,
                        node_properties,
                        edge_properties,
                        min_component_length=1,
                        max_component_length=np.inf,
                        iterations=100,
                        learning_rate=0.1,
                        grid_shape=(5,3),
                        x_spacing=10,
                        y_spacing=3,
                        ax=None,
                        figsize=(10, 5)):
    # Create a new graph
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Get weakly connected components
    components = list(nx.weakly_connected_components(G))
    tot_num_components = len(components)

    # Get number to plot
    num_gridpoints = grid_shape[0]*grid_shape[1]

    # Filter for components within a specified size
    components = [c for c in components if
                  len(c) >= min_component_length and
                  len(c) <= max_component_length]

    # Limit the number of components if more than number of subplots
    if num_gridpoints < len(components):
        components = random.sample(components, num_gridpoints)
    num_components = len(components)

    # Define figure and axes
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot each complex
    n_rows, n_cols = grid_shape
    pos_arr = np.array([[0, 0]])
    for i in range(n_rows):
        for j in range(n_cols):
            if i*n_cols + j >= num_components:
                break

            # Get component
            component = components[i*n_cols + j]

            # Get subgraph
            subG = G.subgraph(component).copy()

            # Set location at which to pin complex
            fixed_location = (j*x_spacing, i*y_spacing)
            pos = pin_and_flow_layout(subG,
                                      fixed_location=fixed_location,
                                      iterations=iterations,
                                      learning_rate=learning_rate)

            # Draw nodes
            for node_class, props in node_properties.items():
                node_list = [node for node, data in subG.nodes(data=True) if
                             data['class'] == node_class]
                nx.draw_networkx_nodes(subG, pos, nodelist=node_list, ax=ax, **props)
                nx.draw_networkx_nodes(subG, pos, nodelist=node_list, ax=ax, **props)

            # Draw edges
            for edge_class, props in edge_properties.items():
                edge_list = [(u, v) for (u, v, d) in subG.edges(data=True) if
                             d['class'] == edge_class]
                nx.draw_networkx_edges(subG, pos, edgelist=edge_list, ax=ax, **props)

            this_pos_arr = np.array([p for p in pos.values()])
            pos_arr = np.concatenate((pos_arr, this_pos_arr), axis=0)

    xlim = [np.min(pos_arr[:, 0]) - 1, np.max(pos_arr[:, 0]) + 1]
    ylim = [np.min(pos_arr[:, 1]) - 1, np.max(pos_arr[:, 1]) + 1]

    # Set limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Set equal aspect
    ax.set_aspect('equal')

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_title(f'{num_components} of {tot_num_components} complexes')


def color_connected_components(G):
    # Get connected components
    components = list(nx.weakly_connected_components(G))

    # Generate a list of distinct colors
    colors = 0.8 * plt.cm.rainbow(np.linspace(0, 1, len(components)))
    colors[:, 3] = 1.0
    np.random.shuffle(colors)

    # Create a dictionary mapping each node to its color
    color_map = {}
    for component, color in zip(components, colors):
        for node in component:
            color_map[node] = color

    return [color_map[node] for node in G.nodes()]


# def plot_state(particle,
#                interaction=None,
#                directed=True,
#                k=0.2,
#                node_size=30,
#                system_name=None,
#                max_connected_components=50,
#                figsize=(8,8)):
#
#     fig, ax = plt.subplots(figsize=figsize)
#
#     # Create a graph
#     G = nx.DiGraph()
#
#     # Add connected nodes if interaction is included
#     if interaction is not None:
#         G.add_edges_from(interaction.indices)
#         # Add monomer nodes
#         particle_field_indices = set(i[0] for i in particle.indices)
#         interaction_indices = set(i[0] for i in interaction.indices) | set(
#             i[1] for i in interaction.indices)
#         monomer_indices = particle_field_indices - interaction_indices
#         G.add_nodes_from(monomer_indices)
#     else:
#         particle_indices = set(i[0] for i in particle.indices)
#         G.add_nodes_from(particle_indices)
#
#     # Create subG from a subset of the connected components of G
#     connected_components = list(nx.weakly_connected_components(G))
#     num_components = int(min(len(connected_components), max_connected_components))
#     sub_components = np.random.choice(connected_components, num_components, replace=False)
#     print(sub_components)
#     subG = nx.DiGraph()
#     for component in sub_components:
#         edges = G.subgraph(component).edges
#         nodes = G.subgraph(component).nodes
#         subG.add_edges_from(edges)
#         subG.add_nodes_from(nodes)
#
#     # Use spring_layout with custom parameters
#     pos = nx.spring_layout(subG, k=k, iterations=100)
#
#     # Get colors for nodes based on their connected component
#     node_colors = color_connected_components(subG)
#
#     # Draw the graph
#     if directed:
#         arrowstyle = '->'
#     else:
#         arrowstyle = '-'
#
#     # Draw graph
#     nx.draw(subG, pos, with_labels=False, node_color=node_colors,
#             node_size=node_size, arrowstyle=arrowstyle, ax=ax)
#
#     if system_name is None:
#         system_name = 'the'
#     ax.set_title(f'{system_name} system, {len(sub_components)} of {len(connected_components)} complexes shown')


def show_sim_stats(sim, x_is_time=False, figsize=(8, 8)):

    step_num = np.array([d["step_num"] for d in sim.step_info])
    rule_num = np.array([d["rule_num"] for d in sim.step_info])
    update_duration = np.array(
        [d["update_duration"] for d in sim.step_info])
    rest_duration = np.array([d["rest_duration"] for d in sim.step_info])
    custom_stat_dict = {}
    for stat_name in sim.custom_stat_names:
        custom_stat_dict[stat_name] = np.array(
            [d[stat_name] for d in sim.step_info])
    t = np.array([d["t"] for d in sim.step_info])
    rule_probs = np.array([d["rule_probs"] for d in sim.step_info])
    eligible_rates = np.array([d["eligible_rates"] for d in sim.step_info])
    num_eligible_indices = np.array([d["num_eligible_indices"] for d in sim.step_info])

    if x_is_time:
        x = t
        xlabel = 'time'
    else:
        x = step_num
        xlabel = 'step number'
    xmax = max(x)

    # Create a figure
    fig = plt.figure(figsize=figsize)

    # Create a GridSpec with 2 rows and 2 columns
    gs = gridspec.GridSpec(3, 1)

    # Plot user-specified stats
    ax = fig.add_subplot(gs[0, 0])
    for stat_name, stat_vals in custom_stat_dict.items():
        ax.plot(x, stat_vals, label=stat_name)
    ax.set_xlim(0, xmax)
    _, upper_lim = ax.get_ylim()
    ax.set_ylim(bottom=0, top=upper_lim)
    ax.set_ylabel('number')
    ax.legend(loc='upper right')

    # Plot eligible rates
    ax = fig.add_subplot(gs[1, 0])
    for i in range(eligible_rates.shape[1]):
        ax.plot(x, eligible_rates[:, i], label=sim.rules.rules[i].name)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, np.max(np.ravel(eligible_rates)) * 1.2)
    ax.set_ylabel('eligible rate')
    ax.legend(loc='upper right')

    # Plot compute time
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(x, 1E6*update_duration, label='rules.update_eligibility()')
    ax.plot(x, 1E6*rest_duration, label='rest of gillespie loop')
    ax.legend(loc='upper right')
    ymax = 1E6*np.percentile(update_duration + rest_duration, q=95)
    ax.set_ylim(0, ymax)
    ax.set_xlim(0, xmax)
    ax.set_ylabel('compute time (μs)')

    # Plot x-label
    ax.set_xlabel(xlabel)

    # Fix spacing between plots
    fig.tight_layout()