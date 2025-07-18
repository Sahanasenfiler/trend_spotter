# trendspotter_reddit/app.py

import streamlit as st
import pandas as pd
import networkx as nx
from reddit_fetcher import RedditFetcher
from network_analyzer import NetworkAnalyzer
from visualizer import GraphVisualizer

# --- DATA ANALYSIS FUNCTION (CACHED) ---
@st.cache_data(show_spinner=False)
def run_full_analysis(subreddit, post_limit):
    """
    Fetches and analyzes data, returning only simple data structures to avoid pickling errors.
    """
    fetcher = RedditFetcher()
    interactions_df = fetcher.fetch_subreddit_interactions(subreddit, post_limit)

    if interactions_df is None or interactions_df.empty:
        return {"error": "Could not fetch any interaction data. The subreddit might be inactive or private."}

    analyzer = NetworkAnalyzer(interactions_df)
    communities = analyzer.detect_louvain_communities()
    
    edge_list = list(analyzer.graph.edges(data=True))

    return {
        "edge_list": edge_list,
        "communities": communities,
        "centrality": analyzer.get_centrality_measures(),
        "topics": analyzer.get_community_topics(communities),
        "num_nodes": analyzer.graph.number_of_nodes(),
        "num_edges": analyzer.graph.number_of_edges(),
        "num_communities": len(set(communities.values())),
        "error": None
    }

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="TrendSpotter for Reddit", layout="wide", initial_sidebar_state="expanded")

# --- SIDEBAR (CONTROLS) ---
st.sidebar.title("🔎 TrendSpotter")
st.sidebar.markdown("Analyze conversation trends in any subreddit using Social Network Analysis.")

predefined_subreddits = [
    "python", "datascience", "machinelearning", 
    "gaming", "wallstreetbets", "technology", "programming",
    "Custom..."
]
selection = st.sidebar.selectbox("Choose a subreddit or select 'Custom'", predefined_subreddits)

if selection == "Custom...":
    subreddit_name = st.sidebar.text_input("Enter custom subreddit name", "askreddit").lower()
else:
    subreddit_name = selection

post_limit = st.sidebar.slider("Number of Posts to Analyze", 5, 50, 15, 5)

if st.sidebar.button("Analyze Subreddit", type="primary"):
    with st.spinner(f"Fetching and analyzing r/{subreddit_name}..."):
        st.session_state['results'] = run_full_analysis(subreddit_name, post_limit)

# --- MAIN PAGE (RESULTS) ---
st.title("Reddit Social Network Analysis")

if 'results' in st.session_state:
    results = st.session_state['results']

    if results.get("error"):
        st.error(results["error"])
    else:
        col1, col2 = st.columns([3, 1])

        with col1:
            st.header("Conversation Network Graph")

            stat_cols = st.columns(3)
            with stat_cols[0]:
                st.metric(label="👥 Nodes (Users)", value=results['num_nodes'])
            with stat_cols[1]:
                st.metric(label="🔗 Edges (Interactions)", value=results['num_edges'])
            with stat_cols[2]:
                st.metric(label="💬 Communities", value=results['num_communities'])
            
            if results["edge_list"]:
                reconstructed_graph = nx.DiGraph()
                reconstructed_graph.add_edges_from(results["edge_list"])
                visualizer = GraphVisualizer(reconstructed_graph)
                fig, reason = visualizer.draw_network_with_louvain_communities(results["communities"])
                if fig:
                    st.pyplot(fig, use_container_width=True)
                else:
                    st.warning(f"Could not generate the network graph. Reason: {reason}")
            else:
                st.info("The analysis returned no network connections to draw.")

        with col2:
            st.header("Top Influential Users")
            if results["centrality"] and results["centrality"].get("degree"):
                # Get the top N users based on degree centrality to build the table
                top_users = [user for user, score in results["centrality"]["degree"]]
                
                # Build the DataFrame with all 5 centrality measures
                centrality_df = pd.DataFrame({
                    "Degree": [dict(results["centrality"]["degree"]).get(u, 0) for u in top_users],
                    "Betweenness": [dict(results["centrality"]["betweenness"]).get(u, 0) for u in top_users],
                    "PageRank": [dict(results["centrality"]["pagerank"]).get(u, 0) for u in top_users],
                    "Closeness": [dict(results["centrality"]["closeness"]).get(u, 0) for u in top_users],
                    "Katz": [dict(results["centrality"]["katz"]).get(u, 0) for u in top_users],
                }, index=top_users)
                
                # Display the styled DataFrame
                st.dataframe(centrality_df.style.format("{:.4f}"))
            else:
                st.info("No user centrality scores to display.")

        st.header("Emerging Topics by Community")
        if not results['topics']:
            st.info("No distinct topics with keywords could be identified.")
        else:
            for cid, topics in sorted(results['topics'].items()):
                with st.expander(f"**Community {cid}**: {', '.join(topics[:2])}..."):
                    st.write(f"Top Keywords: `{'`, `'.join(topics)}`")
else:
    st.info("Enter a subreddit and click 'Analyze Subreddit' in the sidebar to begin.")