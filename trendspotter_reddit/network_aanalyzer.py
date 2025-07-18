import networkx as nx
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import community as community_louvain
from itertools import islice
from typing import Dict, List, Any, Tuple, Set

class NetworkAnalyzer:
    """Builds and analyzes a social network graph from Reddit interactions."""

    def __init__(self, interactions_df: pd.DataFrame):
        """
        Initializes the analyzer with a DataFrame of interactions.

        Args:
            interactions_df (pd.DataFrame): DataFrame with 'author' and 'parent_author' columns.
                                            It should also contain a 'text' column for topic modeling.
        """
        self.df = interactions_df if interactions_df is not None else pd.DataFrame()
        self.graph = self._create_network()

    def _create_network(self) -> nx.DiGraph:
        """
        Builds a directed graph from user replies.
        Edges go from the replier (author) to the replied-to user (parent_author).
        Edge weights represent the number of replies between users.
        """
        G = nx.DiGraph()
        if self.df.empty:
            return G

        for _, row in self.df.iterrows():
            # Ensure authors are strings and not NaN or other types
            author = str(row['author'])
            parent = str(row['parent_author'])

            if G.has_edge(author, parent):
                G[author][parent]['weight'] += 1
            else:
                G.add_edge(author, parent, weight=1)

        print(f"Network created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
        return G

    def get_centrality_measures(self, top_n: int = 10) -> Dict[str, List[Tuple[Any, float]]]:
        """
        Calculates key centrality measures for the network nodes.

        Calculates Degree, Betweenness, and PageRank centrality.

        Args:
            top_n (int): The number of top nodes to return for each measure.

        Returns:
            Dict[str, List[Tuple[Any, float]]]: A dictionary where keys are centrality measure names
                                                and values are sorted lists of (node, score) tuples.
        """
        if not self.graph.nodes:
            return {}

        centrality = {
            'degree': sorted(nx.degree_centrality(self.graph).items(), key=lambda x: x[1], reverse=True)[:top_n],
            'betweenness': sorted(nx.betweenness_centrality(self.graph, weight='weight').items(), key=lambda x: x[1], reverse=True)[:top_n],
            'pagerank': sorted(nx.pagerank(self.graph, weight='weight').items(), key=lambda x: x[1], reverse=True)[:top_n],
        }
        return centrality

    def detect_louvain_communities(self) -> Dict[Any, int]:
        """
        Detects communities using the Louvain algorithm on an undirected version of the graph.

        Returns:
            Dict[Any, int]: A dictionary mapping each node to a community ID.
        """
        if not self.graph.nodes:
            return {}
        # Louvain works on undirected graphs.
        undirected_graph = self.graph.to_undirected()
        partition = community_louvain.best_partition(undirected_graph, weight='weight')
        print(f"Detected {len(set(partition.values()))} communities using Louvain algorithm.")
        return partition

    def detect_girvan_newman_communities(self, level: int = 1) -> List[List[Any]]:
        """
        Detects communities using the Girvan-Newman algorithm (computationally expensive).
        This method returns the communities at a specific level of the hierarchy.

        Args:
            level (int): The level of community division to retrieve (1 is the first split).

        Returns:
            List[List[Any]]: A list of lists, where each inner list represents a community of sorted nodes.
        """
        if not self.graph.nodes:
            return []
        print(f"Running Girvan-Newman for level {level} (this can be very slow)...")
        comp_generator = nx.community.girvan_newman(self.graph)
        try:
            # islice gets the desired partition level from the generator without computing all of them.
            for communities in islice(comp_generator, level - 1, level):
                return [sorted(c) for c in communities]
        except (StopIteration, ValueError):
            # This can happen if the graph is empty or has fewer components than the level requested.
            print(f"Could not retrieve Girvan-Newman communities at level {level}.")
            return []
        return []

    def get_community_topics(self, communities: Dict[Any, int], top_n_words: int = 5) -> Dict[int, List[str]]:
        """
        Identifies top keywords for each community using TF-IDF.

        Args:
            communities (Dict[Any, int]): A dictionary mapping nodes to community IDs (from Louvain).
            top_n_words (int): The number of top keywords to extract for each community.

        Returns:
            Dict[int, List[str]]: A dictionary mapping community IDs to a list of top keywords.
        """
        if not communities or self.df.empty:
            return {}

        # Create a mapping from author to their community ID
        author_to_community = {author: cid for author, cid in communities.items()}
        # Ensure 'community' column uses a safe mapping, handling authors not in the community dict
        self.df['community'] = self.df['author'].map(author_to_community)

        community_topics = {}
        # Group the dataframe by the newly created 'community' column
        for cid, group in self.df.groupby('community'):
            if group.empty or 'text' not in group or group['text'].isnull().all():
                continue

            # Use a list comprehension for cleaner text filtering
            corpus = [text for text in group['text'] if isinstance(text, str)]
            if not corpus:
                community_topics[int(cid)] = ["not enough text"]
                continue

            try:
                vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, min_df=2)
                tfidf_matrix = vectorizer.fit_transform(corpus)
                feature_names = vectorizer.get_feature_names_out()
                word_scores = tfidf_matrix.sum(axis=0).A1
                top_word_indices = word_scores.argsort()[-top_n_words:][::-1]
                community_topics[int(cid)] = [feature_names[i] for i in top_word_indices]
            except ValueError:
                # This can happen if the vocabulary is too small (e.g., less than min_df)
                community_topics[int(cid)] = ["not enough text to analyze"]

        # Clean up the added column
        self.df.drop(columns=['community'], inplace=True, errors='ignore')
        return community_topics

    def identify_trends(self, centrality: Dict, communities: Dict, topics: Dict) -> Dict[str, Any]:
        """
        Synthesizes analysis results to identify key trends, influencers, and topics.

        Args:
            centrality (Dict): The output from get_centrality_measures.
            communities (Dict): The partition output from detect_louvain_communities.
            topics (Dict): The topics output from get_community_topics.

        Returns:
            Dict[str, Any]: A dictionary summarizing the key findings.
        """
        if not centrality or not communities:
            return {}

        # 1. Identify Key Influencer (highest PageRank for authority)
        key_influencer = "N/A"
        if centrality.get('pagerank'):
            key_influencer = centrality['pagerank'][0][0]

        # 2. Identify Key Connector (highest Betweenness for bridging communities)
        key_connector = "N/A"
        if centrality.get('betweenness'):
            key_connector = centrality['betweenness'][0][0]

        # 3. Identify Hottest Community (largest discussion group)
        most_active_community_id = None
        main_topic_keywords = []
        community_sizes = pd.Series(communities).value_counts()
        if not community_sizes.empty:
            most_active_community_id = int(community_sizes.index[0])
            if topics and most_active_community_id in topics:
                main_topic_keywords = topics[most_active_community_id]

        trends = {
            "key_influencer": {
                "user": key_influencer,
                "reason": "This user has the highest PageRank score, indicating they are a central authority whose interactions are considered important by other influential users."
            },
            "key_connector": {
                "user": key_connector,
                "reason": "This user has the highest Betweenness Centrality, meaning they act as a crucial bridge connecting different conversation groups and facilitating information flow."
            },
            "hottest_community": {
                "id": most_active_community_id,
                "size": int(community_sizes.iloc[0]) if not community_sizes.empty else 0,
                "topic": ", ".join(main_topic_keywords),
                "reason": "This is the largest identified community, representing the most significant and active conversation cluster in the analyzed posts."
            }
        }
        return trends