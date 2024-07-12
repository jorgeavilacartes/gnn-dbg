from typing import Optional
# https://graphviz.readthedocs.io/en/stable/manual.html
# from graphviz import Digraph
from .plot import PlotDigraph

class DeBruijnGraph(PlotDigraph): 
    "Create de Bruijn Graph from a sequence"
    def __init__(self, sequence: Optional[str] = None, kmers: Optional[list] = None, k: int = 3):# Instantiate PlotDigraph attributes
        self.sequence = sequence
        self.kmers = kmers
        self.k  = k
        
        super().__init__() 
        self._build_graph()

    def graph_from_sequence(self,):
        "Given a sequence, create the De Bruijn Graph"
        print("From sequence")
        for j,c in enumerate(self.sequence):
            if j+self.k < len(self.sequence):
                # Nodes: each (k-1)-mer is a node
                u = self.sequence[j:j+self.k] 
                v = self.sequence[j+1:j+self.k+1]
                
                # Add Edge
                self.edges.append((u,v))
                
                # Add Nodes
                self.nodes.add(u)
                self.nodes.add(v)

    def graph_from_kmers(self,):
        "Given a list of kmers, create the De Bruijn Graph"
        print("From k-mers")
        # Each k-mer is an edge
        for kmer in self.kmers: 
            u,v = kmer[:-1], kmer[1:]
            if len(u) == len(v) == self.k:
                self.edges.append((u,v))
                self.nodes.add(u)
                self.nodes.add(v)

    def _build_graph(self,): 
        "Build graph from a sequence or a list of kmers"
        if self.sequence is not None:         
            self.graph_from_sequence()
        elif self.kmers is not None: 
            self.graph_from_kmers()