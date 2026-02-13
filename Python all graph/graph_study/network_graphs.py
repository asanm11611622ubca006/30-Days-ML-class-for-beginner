import networkx as nx
import matplotlib.pyplot as plt

def network_graphs():
    # 1. Simple Undirected Graph
    G = nx.Graph()
    G.add_edges_from([(1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (4, 1)])
    
    plt.figure(figsize=(6, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1500, edge_color='gray', font_size=15, font_weight='bold')
    plt.title("Simple Network Graph")
    plt.show()
    print("Displayed Simple Network Graph")

    # 2. Directed Graph with Weights
    DG = nx.DiGraph()
    DG.add_edge('A', 'B', weight=4)
    DG.add_edge('B', 'C', weight=2)
    DG.add_edge('C', 'A', weight=3)
    DG.add_edge('A', 'C', weight=1)
    
    plt.figure(figsize=(6, 6))
    pos_d = nx.circular_layout(DG)
    nx.draw(DG, pos_d, with_labels=True, node_color='lightgreen', node_size=2000, arrowsize=20)
    edge_labels = nx.get_edge_attributes(DG, 'weight')
    nx.draw_networkx_edge_labels(DG, pos_d, edge_labels=edge_labels)
    plt.title("Directed Weighted Graph")
    plt.show()
    print("Displayed Directed Weighted Graph")

if __name__ == "__main__":
    print("Generating Network Graphs...")
    network_graphs()
