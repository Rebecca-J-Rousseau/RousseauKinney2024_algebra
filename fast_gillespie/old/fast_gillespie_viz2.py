import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter
from networkx import connected_components
import random


# Visualize the complexes
def visualize_complexes(nodes,
                        edges,
                        node_properties,
                        edge_properties,
                        max_components=None,
                        figsize=(8, 8)):
    # Create a new graph
    G = nx.DiGraph()

    # Add nodes with their classes
    G.add_nodes_from(nodes.keys())
    nx.set_node_attributes(G, nodes, 'class')

    # Add edges with their classes
    G.add_edges_from((u, v) for u, v, _ in edges)
    nx.set_edge_attributes(G, {(u, v): {'class': t} for u, v, t in edges})

    # Get weakly connected components
    components = list(nx.weakly_connected_components(G))

    # Limit the number of components if specified
    if max_components is not None and max_components < len(components):
        selected_components = random.sample(components, max_components)
        nodes_to_keep = set().union(*selected_components)
        G = G.subgraph(nodes_to_keep).copy()

    # Set up the plot
    plt.figure(figsize=figsize)
    #pos = nx.spring_layout(G)
    pos = nx.multipartite_layout(G)

    # Draw nodes
    for node_class, props in node_properties.items():
        node_list = [node for node, data in G.nodes(data=True) if
                     data['class'] == node_class]
        nx.draw_networkx_nodes(G, pos, nodelist=node_list, **props)
        nx.draw_networkx_nodes(G, pos, nodelist=node_list, **props)

    # Draw edges
    for edge_class, props in edge_properties.items():
        edge_list = [(u, v) for (u, v, d) in G.edges(data=True) if
                     d['class'] == edge_class]
        nx.draw_networkx_edges(G, pos, edgelist=edge_list, **props)

    # Remove axis
    plt.axis('off')

    # Show the plot
    plt.tight_layout()
    plt.show()


# def visualize_network(nodes, edges, node_shapes, node_colors, edge_colors,
#                       edge_styles, max_components=None, figsize=(8, 8),
#                       k=0.2, node_size=30, arrowsize=10):
#     # Create a new graph
#     G = nx.DiGraph()
#
#     # Add nodes with different classes
#     G.add_nodes_from(nodes.keys())
#     nx.set_node_attributes(G, nodes, 'class')
#
#     # Add edges with different classes
#     G.add_edges_from((u, v) for u, v, _ in edges)
#     nx.set_edge_attributes(G, {(u, v): {'type': t} for u, v, t in edges})
#
#     # Get weakly connected components
#     components = list(nx.weakly_connected_components(G))
#
#     # Limit the number of components if specified
#     if max_components is not None and max_components < len(components):
#         selected_components = random.sample(components, max_components)
#         nodes_to_keep = set().union(*selected_components)
#         G = G.subgraph(nodes_to_keep).copy()
#
#     # Set up the plot
#     plt.figure(figsize=figsize)
#     pos = nx.spring_layout(G, k=k)
#
#     # Draw nodes
#     for node_class, shape in node_shapes.items():
#         node_list = [node for node, data in G.nodes(data=True) if
#                      data['class'] == node_class]
#         nx.draw_networkx_nodes(G, pos, nodelist=node_list,
#                                node_color=node_colors[node_class],
#                                node_shape=shape, node_size=50)
#
#     # Draw edges
#     for edge_type, color in edge_colors.items():
#         edge_list = [(u, v) for (u, v, d) in G.edges(data=True) if
#                      d['type'] == edge_type]
#         nx.draw_networkx_edges(G, pos,
#                                edgelist=edge_list,
#                                edge_color=color,
#                                width=2,
#                                arrowstyle=edge_styles[edge_type],
#                                arrowsize=arrowsize)
#
#     # Remove axis
#     plt.axis('off')
#
#     # Show the plot
#     plt.tight_layout()
#     plt.show()


# # Example usage:
# if __name__ == "__main__":
#     # Define nodes
#     nodes = {
#         'A': 'class1', 'B': 'class1', 'C': 'class2', 'D': 'class2',
#         'E': 'class3', 'F': 'class3', 'G': 'class1', 'H': 'class2',
#         'I': 'class3', 'J': 'class1', 'K': 'class2', 'L': 'class3'
#     }
#
#     # Define edges
#     edges = [
#         ('A', 'B', 'type1'), ('B', 'C', 'type1'), ('C', 'D', 'type2'),
#         ('D', 'E', 'type2'), ('E', 'F', 'type3'), ('F', 'A', 'type3'),
#         ('G', 'H', 'type1'), ('H', 'I', 'type2'), ('I', 'G', 'type3'),
#         ('J', 'K', 'type1'), ('K', 'L', 'type2'), ('L', 'J', 'type3')
#     ]
#
#     # Define node shapes and colors
#     node_shapes = {'class1': 'o', 'class2': 's', 'class3': '^'}
#     node_colors = {'class1': 'skyblue', 'class2': 'lightgreen', 'class3': 'salmon'}
#
#     # Define edge colors and styles
#     edge_colors = {'type1': 'red', 'type2': 'blue', 'type3': 'green'}
#     edge_styles = {'type1': '->', 'type2': '-[', 'type3': '-|>'}


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
               max_connected_components=50,
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

    # Create subG from a subset of the connected components of G
    connected_components = list(nx.weakly_connected_components(G))
    num_components = int(min(len(connected_components), max_connected_components))
    sub_components = np.random.choice(connected_components, num_components, replace=False)
    print(sub_components)
    subG = nx.DiGraph()
    for component in sub_components:
        edges = G.subgraph(component).edges
        nodes = G.subgraph(component).nodes
        subG.add_edges_from(edges)
        subG.add_nodes_from(nodes)

    # Use spring_layout with custom parameters
    pos = nx.spring_layout(subG, k=k, iterations=100)

    # Get colors for nodes based on their connected component
    node_colors = color_connected_components(subG)

    # Draw the graph
    if directed:
        arrowstyle = '->'
    else:
        arrowstyle = '-'

    # Draw graph
    nx.draw(subG, pos, with_labels=False, node_color=node_colors,
            node_size=node_size, arrowstyle=arrowstyle, ax=ax)

    if system_name is None:
        system_name = 'the'
    ax.set_title(f'{system_name} system, {len(sub_components)} of {len(connected_components)} complexes shown')

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