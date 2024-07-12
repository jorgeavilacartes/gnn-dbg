# https://graphviz.readthedocs.io/en/stable/manual.html
from graphviz import Digraph

class PlotDigraph:
    "Plot graph from a set of nodes and a list of edges"
    def __init__(self,):
        # Create an empty graph
        self.new_graph()

    def new_graph(self,):
        "Create an empty graph"
        self.digraph  = Digraph(comment='De Bruijn')
        self.nodes = set()
        self.edges = list()

    def create_dot_graph(self, nodes, edges):
        "Create graph in 'dot' language to be plotted"
        # Add nodes to the graph
        for node in nodes: 
            self.add_node(node)
        # Add edges to the graph
        for edge in edges: 
            u, v = edge
            self.add_edge(u,v)

    def add_node(self, node: str):
        "Add node with same name"
        self.digraph.node(node, node) 

    def add_edge(self, u, v):
        "Add edge u -> v"
        self.digraph.edge(u,v)

    def show(self,):
        "Print graph"
        self.create_dot_graph(self.nodes, self.edges)
        return self.digraph