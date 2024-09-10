import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter


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


def plot_state(particle,
               interaction=None,
               directed=True,
               k=0.2,
               node_size=30,
               system_name=None,
               figsize=(8,8)):

    fig, ax = plt.subplots(figsize=figsize)

    # Create a graph
    G = nx.DiGraph()

    # Add connected nodes if interaction is included
    if interaction is not None:
        G.add_edges_from(interaction.indices)
        # Add monomer nodes
        particle_field_indices = set(i[0] for i in particle.indices)
        interaction_indices = set(i[0] for i in interaction.indices) | set(
            i[1] for i in interaction.indices)
        monomer_indices = particle_field_indices - interaction_indices
        G.add_nodes_from(monomer_indices)
    else:
        particle_indices = set(i[0] for i in particle.indices)
        G.add_nodes_from(particle_indices)

    # Use spring_layout with custom parameters
    pos = nx.spring_layout(G, k=k, iterations=50)

    # Get colors for nodes based on their connected component
    node_colors = color_connected_components(G)

    # Draw the graph
    if directed:
        arrowstyle = '->'
    else:
        arrowstyle = '-'

    nx.draw(G, pos, with_labels=False, node_color=node_colors,
            node_size=node_size, arrowstyle=arrowstyle, ax=ax)

    if system_name is None:
        system_name = 'the'
    ax.set_title(f'Final state of {system_name} system')

# def plot_state_v2(ax,
#                particles,
#                interactions=(),
#                arrows=None,
#                k=0.2, node_size=30, arrowsize=10):
#
#     if arrows is None:
#         arrows = ['->']*len(interactions)
#     else:
#         assert len(arrows) == len(interactions)
#
#     # Create a directed graph
#     G = nx.DiGraph()
#
#     node_shape_options = ['.', 's', '^', 'h']
#     edge_color_options = ['k', 'C0', 'C1', 'C2', 'C3']
#
#     # Need to assign different names to nodes for different particles
#     particle_info = []
#     for k, particle in enumerate(particles):
#         indices = list(i[0] for i in particle.indices)
#         G.add_nodes_from(indices)
#         d = dict(indices=indices,
#                  shape=node_shape_options[k%len(node_shape_options)])
#         particle_info.append(d)
#
#     interaction_info = []
#     for k, interaction in enumerate(interactions):
#         pairs = list(interaction.indices)
#         G.add_edges_from(pairs)
#         d = dict(indices=pairs,
#                  arrows=arrows[k],
#                  color=edge_color_options[k%len(edge_color_options)])
#         interaction_info.append(d)
#
#     # Use spring_layout with custom parameters
#     pos = nx.spring_layout(G, k=k, iterations=50)
#
#     # Get colors for nodes based on their connected component
#     # node_colors = color_connected_components(G)
#
#     for k in range(len(particles)):
#         d = particle_info[k]
#         nx.draw_networkx_nodes(G, pos, ax=ax,
#                                nodelist=d['indices'],
#                                node_shape=d['shape'],
#                                node_size=node_size)
#
#     for k in range(len(interactions)):
#         d = interaction_info[k]
#         nx.draw_networkx_edges(G, pos, ax=ax,
#                                edgelist=d['indices'],
#                                edge_color=d['color'],
#                                arrowsize=arrowsize,
#                                arrowstyle=d['arrows'],
#                                width=2)


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
    max_stat_val = 0
    for stat_name, stat_vals in custom_stat_dict.items():
        ax.plot(x, stat_vals, label=stat_name)
        max_stat_val = max(max_stat_val, np.max(stat_vals))
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, max_stat_val * 1.2)
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