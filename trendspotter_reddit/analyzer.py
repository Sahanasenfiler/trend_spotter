import networkx as nx
import pandas as pd
import community as community_louvain
import numpy as np
from typing import Dict, List, Any, Tuple

class NetworkAnalyzer:
    """Builds and analyzes a social network graph from interactions."""

    def __init__(self, interactions_df: pd.DataFrame):
        """Initializes the analyzer with a DataFrame of interactions."""
        self.df = interactions_df if interactions_df is not None else pd.DataFrame()
        self.graph = self._create_network()
        self.communities = None # Cache communities for performance

    def _create_network(self) -> nx.DiGraph:
        """Builds a directed graph from user replies."""
        G = nx.DiGraph()
        if self.df.empty:
            return G
        for _, row in self.df.iterrows():
            author = str(row['author'])
            parent = str(row['parent_author'])
            if G.has_edge(author, parent):
                G[author][parent]['weight'] += 1
            else:
                G.add_edge(author, parent, weight=1)
        print(f"Network created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
        return G
    
    def _normalize_dict_values(self, scores: Dict[Any, float]) -> Dict[Any, float]:
        """Normalizes dictionary values to a 0-1 scale."""
        if not scores: return {}
        min_val = min(scores.values())
        max_val = max(scores.values())
        if max_val == min_val:
            return {k: 0.0 for k in scores}
        return {k: (v - min_val) / (max_val - min_val) for k, v in scores.items()}

    def detect_louvain_communities(self) -> Dict[Any, int]:
        """Detects communities using the Louvain algorithm. Caches the result."""
        if self.communities is not None:
            return self.communities
        if not self.graph.nodes:
            self.communities = {}
            return {}
        undirected_graph = self.graph.to_undirected()
        partition = community_louvain.best_partition(undirected_graph, weight='weight')
        print(f"Detected {len(set(partition.values()))} communities using Louvain algorithm.")
        self.communities = partition
        return partition

    def calculate_hybrid_centrality(self, top_n: int = 10, weights: Dict[str, float] = None) -> List[Tuple[str, float]]:
        """
        Calculates a hybrid centrality score combining Degree, PageRank, Betweenness, and Louvain structure.
        """
        if weights is None:
            weights = {'degree': 0.25, 'pagerank': 0.25, 'betweenness': 0.25, 'community': 0.25}
        
        if not self.graph.nodes:
            return []

        degree = nx.degree_centrality(self.graph)
        pagerank = nx.pagerank(self.graph, weight='weight')
        betweenness = nx.betweenness_centrality(self.graph, weight='weight')
        communities = self.detect_louvain_communities()

        community_influence = {}
        for node in self.graph.nodes():
            node_community = communities.get(node)
            if node_community is None:
                community_influence[node] = 0
                continue
            inter_community_edges = sum(1 for neighbor in self.graph.successors(node) if communities.get(neighbor) != node_community)
            intra_community_edges = sum(1 for neighbor in self.graph.successors(node) if communities.get(neighbor) == node_community)
            community_influence[node] = intra_community_edges + (1.5 * inter_community_edges)

        norm_degree = self._normalize_dict_values(degree)
        norm_pagerank = self._normalize_dict_values(pagerank)
        norm_betweenness = self._normalize_dict_values(betweenness)
        norm_community = self._normalize_dict_values(community_influence)

        hybrid_scores = {node: (weights['degree'] * norm_degree.get(node, 0) +
                                weights['pagerank'] * norm_pagerank.get(node, 0) +
                                weights['betweenness'] * norm_betweenness.get(node, 0) +
                                weights['community'] * norm_community.get(node, 0))
                         for node in self.graph.nodes()}
        return sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    def evaluate_centrality(self, metrics_to_eval: List[str], top_k: int = 10) -> pd.DataFrame:
        """
        Calculates and compares evaluation metrics for different centrality scores.
        """
        communities = self.detect_louvain_communities()
        if not communities:
            print("Cannot evaluate without communities.")
            return pd.DataFrame()

        num_communities = len(set(communities.values()))
        report = {}
        centrality_scores = {
            'degree': nx.degree_centrality(self.graph),
            'pagerank': nx.pagerank(self.graph, weight='weight'),
            'betweenness': nx.betweenness_centrality(self.graph, weight='weight'),
            'hybrid': dict(self.calculate_hybrid_centrality(top_n=self.graph.number_of_nodes()))
        }

        try:
            initial_lcc_size = len(max(nx.weakly_connected_components(self.graph), key=len))
        except ValueError:
            initial_lcc_size = 0

        for metric in metrics_to_eval:
            top_nodes = [node for node, score in sorted(centrality_scores[metric].items(), key=lambda x: x[1], reverse=True)[:top_k]]
            if not top_nodes: continue

            covered_communities = {communities.get(n) for n in top_nodes if n in communities}
            
            comm_counts = pd.Series([communities.get(n) for n in top_nodes if n in communities]).value_counts()
            probs = comm_counts / comm_counts.sum() if not comm_counts.empty else pd.Series()
            shannon = -np.sum(probs * np.log2(probs)) if not probs.empty else 0
            
            hubs = sum(1 for n in top_nodes if sum(1 for neighbor in self.graph.successors(n) if communities.get(neighbor) != communities.get(n)) <= sum(1 for neighbor in self.graph.successors(n) if communities.get(neighbor) == communities.get(n)))
            bridges = top_k - hubs
            
            G_copy = self.graph.copy()
            G_copy.remove_node(top_nodes[0])
            try:
                new_lcc_size = len(max(nx.weakly_connected_components(G_copy), key=len))
            except ValueError:
                new_lcc_size = 0
            
            report[metric] = {
                "Top 1 Node": top_nodes[0],
                "Community Coverage": f"{len(covered_communities)}/{num_communities}",
                "Shannon Diversity": round(shannon, 3),
                "Role Analysis (Hubs/Bridges)": f"{hubs}/{bridges}",
                "Network Disruption Score": initial_lcc_size - new_lcc_size
            }
        return pd.DataFrame.from_dict(report, orient='index')

# --- MAIN EXECUTION BLOCK ---
if __name__ == '__main__':
    file_path = 'dataset.csv'
    try:
        # --- THIS IS THE CORRECTED LINE ---
        df = pd.read_csv(
            file_path,
            engine='python',          # Use the more robust python engine
            on_bad_lines='skip',      # Skip any rows that are completely broken
            dtype={'author': str, 'parent_author': str, 'text': str}
        )
        # --- END OF CORRECTION ---

        df.dropna(subset=['author', 'parent_author'], inplace=True)
        print(f"Successfully loaded {len(df)} rows from {file_path}")
        if len(df) == 0:
             print("Warning: DataFrame is empty after loading and cleaning. No analysis will be run.")
             exit()

    except FileNotFoundError:
        print(f"--- ERROR: {file_path} not found. Please ensure the file exists in the same directory. ---")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        exit()
    
    analyzer = NetworkAnalyzer(df)
    metrics = ['degree', 'pagerank', 'betweenness', 'hybrid']
    print("\n--- Centrality Evaluation Report (Top 10 Nodes) ---")
    evaluation_df = analyzer.evaluate_centrality(metrics, top_k=10)
    
    if not evaluation_df.empty:
        try:
            print(evaluation_df.to_markdown(index=True))
        except ImportError:
            print(evaluation_df)