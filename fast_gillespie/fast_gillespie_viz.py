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


def plot_state(ax, particle, interaction=None, directed=True, k=0.2, node_size=30):
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

def show_sim_stats(sim,
                   particle,
                   interaction,
                   directed,
                   x_is_time=False, system_name=None, figsize=(16, 8)):

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
    gs = gridspec.GridSpec(3, 2)

    ax = fig.add_subplot(gs[:, 0])
    plot_state(ax=ax,
               particle=particle,
               interaction=interaction,
               directed=directed,
               k=0.2, node_size=30)
    if system_name is None:
        system_name = 'System'
    ax.set_title(f"{system_name} after {sim.num_steps:,d} steps")

    # Plot user-specified stats
    ax = fig.add_subplot(gs[0, 1])
    max_stat_val = 0
    for stat_name, stat_vals in custom_stat_dict.items():
        ax.plot(x, stat_vals, label=stat_name)
        max_stat_val = max(max_stat_val, np.max(stat_vals))
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, max_stat_val * 1.2)
    ax.set_ylabel('number')
    ax.legend(loc='upper right')

    # Plot eligible rates
    ax = fig.add_subplot(gs[1, 1])
    for i in range(eligible_rates.shape[1]):
        ax.plot(x, eligible_rates[:, i], label=sim.rules.rules[i].name)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, np.max(np.ravel(eligible_rates)) * 1.2)
    ax.set_ylabel('eligible rate')
    ax.legend(loc='upper right')

    # # Plot number of eligible indices
    # ax = fig.add_subplot(gs[2, 1])
    # for i in range(num_eligible_indices.shape[1]):
    #     ax.semilogy(x, num_eligible_indices[:, i], label=sim.rules.rules[i].name)
    # ax.set_xlim(0, xmax)
    # finite_mask = np.isfinite(num_eligible_indices)
    # ax.set_ylim(0, np.max(np.ravel(finite_mask*num_eligible_indices)) * 1.2)
    # ax.set_ylabel('num eligible indices')
    # ax.legend(loc='upper right')

    # Plot compute time
    ax = fig.add_subplot(gs[2, 1])
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