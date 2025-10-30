import networkx as nx
import matplotlib.pyplot as plt

def load_graph():
    # Default: built-in karate graph
    G = nx.karate_club_graph()

    # Optional: load from edgelist file
    # import pathlib
    # edgefile = pathlib.Path(__file__).parent.parent / "data" / "karate.edgelist"
    # if edgefile.exists():
    #     G = nx.read_edgelist(edgefile, nodetype=int)

    return G

def compute_basic_measures(G):
    diameter = nx.diameter(G)
    avg_distance = nx.average_shortest_path_length(G)
    return diameter, avg_distance

def plot_distributions(G):
    pagerank = nx.pagerank(G)
    degrees = dict(G.degree())

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.hist(list(degrees.values()), bins=10)
    plt.title("Degree Distribution")
    plt.xlabel("degree"); plt.ylabel("count")

    plt.subplot(1,2,2)
    plt.hist(list(pagerank.values()), bins=10)
    plt.title("PageRank Distribution")
    plt.xlabel("pagerank"); plt.ylabel("count")
    plt.tight_layout()
    plt.show()

def compare_with_random(G):
    N = G.number_of_nodes()
    E = G.number_of_edges()
    avg_k = 2 * E / N
    p = avg_k / (N - 1)

    Gr = nx.gnp_random_graph(N, p, seed=42)

    def stats(graph):
        return dict(
            avg_path = nx.average_shortest_path_length(graph),
            assortativity = nx.degree_assortativity_coefficient(graph),
            clustering = nx.average_clustering(graph),
        )

    return p, stats(G), stats(Gr)

def manual_triangle_count(G):
    triangles = {}
    for node in G.nodes():
        neighbors = set(G.neighbors(node))
        cnt = 0
        for u in neighbors:
            for v in neighbors:
                if u < v and G.has_edge(u, v):
                    cnt += 1
        triangles[node] = cnt
    total_manual = sum(triangles.values()) // 3
    total_nx = sum(nx.triangles(G).values()) // 3
    return total_manual, total_nx, triangles

def first_split_girvan_newman(G):
    from networkx.algorithms.community import girvan_newman
    comp = girvan_newman(G)
    communities = tuple(sorted(c) for c in next(comp))
    return communities

def main():
    G = load_graph()
    print(f"Nodes: {G.number_of_nodes()}  Edges: {G.number_of_edges()}")

    # Visualize basic graph
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(6,6))
    nx.draw(G, pos=pos, with_labels=True, node_size=500)
    plt.title("Zachary's Karate Club")
    plt.show()

    diameter, avg_distance = compute_basic_measures(G)
    print(f"Diameter: {diameter}")
    print(f"Average shortest path length: {avg_distance:.4f}")

    plot_distributions(G)

    p, s_real, s_rand = compare_with_random(G)
    print(f"Random graph p: {p:.4f}")
    print("Real graph stats:", s_real)
    print("Random graph stats:", s_rand)

    total_manual, total_nx, triangles = manual_triangle_count(G)
    print(f"Triangles (manual): {total_manual}")
    print(f"Triangles (networkx): {total_nx}")

    # Community detection
    communities = first_split_girvan_newman(G)
    print("First split communities:", communities)

    # Draw communities
    colors = []
    for n in G.nodes():
        for idx, comm in enumerate(communities):
            if n in comm:
                colors.append(idx); break
    plt.figure(figsize=(6,6))
    nx.draw(G, pos=pos, with_labels=True, node_color=colors, node_size=500, cmap=plt.cm.tab10)
    plt.title("Communities (first split, Girvan–Newman)")
    plt.show()

if __name__ == "__main__":
    main()
