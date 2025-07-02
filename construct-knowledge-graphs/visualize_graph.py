import networkx as nx
import matplotlib.pyplot as plt

def visualize_graph(graph_doc: any) -> None:
    """"Visualizes graph document."""
    G = nx.DiGraph()

    for node in graph_doc.nodes:
        G.add_node(node.id, label=node.type)

    for rel in graph_doc.relationships:
        G.add_edge(rel.source.id, rel.target.id, label=rel.type)

    pos = nx.spring_layout(G)
    node_labels = {n: f"{n}\n({d.get('label', '')})" for n, d in G.nodes(data=True)}
    edge_labels = nx.get_edge_attributes(G, 'label')

    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, labels=node_labels, node_color='lightblue', node_size=2500, with_labels=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    plt.title("Graph Visualization from Document")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
