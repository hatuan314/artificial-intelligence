import random
import networkx as nx

def load_graph():
    # Default: built-in karate graph
    G = nx.karate_club_graph()

    # Optional: load from edgelist file
    # import pathlib
    # edgefile = pathlib.Path(__file__).parent.parent / "data" / "karate.edgelist"
    # if edgefile.exists():
    #     G = nx.read_edgelist(edgefile, nodetype=int)

    return G

# --- Scoring functions ---
def jaccard_score(G, i, j):
    n_i, n_j = set(G.neighbors(i)), set(G.neighbors(j))
    union = n_i | n_j
    if not union:
        return 0.0
    return len(n_i & n_j) / len(union)

def preferential_attachment(G, i, j):
    return G.degree(i) * G.degree(j)

def inverse_distance(G, i, j):
    try:
        d = nx.shortest_path_length(G, i, j)
        if d == 0:
            return 0.0
        return 1.0 / d
    except nx.NetworkXNoPath:
        return 0.0

def random_score(G, i, j):
    return random.random()

# --- Link prediction ---
def link_prediction(G, k, i, score_func):
    candidates = [n for n in G.nodes() if n != i and not G.has_edge(i, n)]
    scores = [(n, score_func(G, i, n)) for n in candidates]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]

# --- Evaluation (leave-one-out) ---
def evaluate_link_prediction(G, k, score_func):
    correct, total = 0, 0
    for (u, v) in list(G.edges()):
        Gt = G.copy()
        Gt.remove_edge(u, v)
        preds = [x for x, _ in link_prediction(Gt, k, u, score_func)]
        if v in preds:
            correct += 1
        total += 1
    return correct / total if total else 0.0

def demo():
    G = load_graph()
    print(f"Nodes: {G.number_of_nodes()}  Edges: {G.number_of_edges()}")
    source = 0
    k = 5
    for func in [jaccard_score, preferential_attachment, inverse_distance, random_score]:
        preds = link_prediction(G, k, source, func)
        print(f"Top-{k} predictions from node {source} using {func.__name__}:")
        for (j, s) in preds:
            print(f"  ({source}, {j}) -> {round(s, 4)}")
        print()

    for func in [jaccard_score, preferential_attachment, inverse_distance]:
        acc = evaluate_link_prediction(G, k, func)
        print(f"Accuracy (k={k}) — {func.__name__}: {acc:.3f}")

if __name__ == "__main__":
    demo()
