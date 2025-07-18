# trendspotter_reddit/visualizer.py

import networkx as nx
import matplotlib.pyplot as plt
import random

class GraphVisualizer:
    def __init__(self, graph):
        self.graph = graph

    def draw_network_with_louvain_communities(self, communities):
        """
        Draws the graph and returns a tuple: (figure, reason_string).
        On success, reason_string is None. On failure, figure is None.
        """
        if not self.graph or self.graph.number_of_nodes() == 0:
            return (None, "The fetched data contained no user interactions to build a graph from.")

        graph_to_draw = self.graph.copy()
        root_nodes = [node for node in graph_to_draw.nodes() if node.startswith('r/')]
        graph_to_draw.remove_nodes_from(root_nodes)
        
        if not graph_to_draw.nodes():
            return (None, f"The initial graph only contained the root node(s) ({', '.join(root_nodes)}) and no other users. No graph to draw after filtering.")
        
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.set_title("Reddit Conversation Network", fontsize=25)

        num_communities = len(set(communities.values()))
        colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(num_communities)]
        
        community_color_map = {cid: color for cid, color in zip(sorted(list(set(communities.values()))), colors)}
        node_colors = [community_color_map.get(communities.get(node), '#808080') for node in graph_to_draw.nodes()]

        node_sizes = [graph_to_draw.degree(node) * 40 + 50 for node in graph_to_draw.nodes()]
        
        pos = nx.spring_layout(graph_to_draw, k=0.2, iterations=50)
        
        nx.draw_networkx_edges(graph_to_draw, pos, alpha=0.2, edge_color='gray', ax=ax)
        nx.draw_networkx_nodes(graph_to_draw, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9, ax=ax)
        nx.draw_networkx_labels(graph_to_draw, pos, font_size=8, font_color='black', ax=ax)
        
        plt.axis('off')
        
        return (fig, None) # Success: return the figure and a None reason