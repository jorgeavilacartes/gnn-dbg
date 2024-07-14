import os.path as osp
from pathlib import Path
from Bio import SeqIO
from collections import defaultdict
from typing import Union

import torch
from torch_geometric.data import Data, Dataset

from src.graphs import DeBruijnGraph


class dBGDataset(Dataset):

    def __init__(self, 
                 root,
                 k,
                 fastq: Union[str, list[str]]=None,                 
                 transform=None, pre_transform=None, pre_filter=None):
        self.k = k
        self.fastq = fastq #"../data/ecolik12_ont_0001.fastq"
        super().__init__(root, transform, pre_transform, pre_filter)
        

    @property
    def raw_file_names(self,):
        return self.fastq

    @property
    def processed_file_names(self,):

        # count sequences in fastq
        record_dict = SeqIO.index(self.fastq, format="fastq")
        seq_ids = list(record_dict.keys())
        processed_filename = [f"data_{x}" for x in range(len(seq_ids))]
        
        return processed_filename

    def process(self):
        idx = 0
        record_dict = SeqIO.index(self.fastq, format="fastq")
        seq_ids = list(record_dict.keys())

        for seq_id in seq_ids:
            print(f"Building {seq_id}")
            # Read data from `raw_path`.
            record = record_dict[seq_id]
            seq = record.seq
            dbg = DeBruijnGraph(sequence=seq, k=self.k)
            kmer_count = self.count_kmers(seq, self.k)
            data = self.create_torch_graph(dbg.nodes, dbg.edges, kmer_count)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            Path(self.processed_dir).mkdir(exist_ok=True, parents=True)
            torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
            idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data
    
    def create_torch_graph(self, nodes: set, edges: list, kmer_count: dict):

        idx_nodes = {kmer: idx  for idx, kmer in  enumerate(nodes)}
        
        # nodes
        feat_nodes = [ [kmer_count[kmer]] for kmer in nodes]
        x = torch.tensor(feat_nodes, dtype=torch.float)

        # edges    
        edges =  list(map( lambda x: (idx_nodes[x[0]],idx_nodes[x[1]]) , edges))
        edge_index = torch.tensor(edges, dtype=torch.long)    

        return Data(x=x, edge_index=edge_index.t().contiguous())

    def count_kmers(self, seq, k):

        kmer_count = defaultdict(int)
        for pos in range(len(seq)-k+1):
            kmer = seq[pos:pos+k]
            kmer_count[kmer] += 1

        return kmer_count    