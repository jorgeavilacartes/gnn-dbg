import os.path as osp
from pathlib import Path
from Bio import SeqIO
from collections import defaultdict, namedtuple
from typing import Union
from collections import Counter
from tqdm import tqdm

import torch
from torch_geometric.data import Data, Dataset

from src.graphs import DeBruijnGraph


Seq = namedtuple("Seq", ["filename","seqid","idx"])

class dBGDataset(Dataset):

    def __init__(self, 
                 root,
                 k,
                 files: Union[str, list[str]]=None,                 
                 format: str = "fasta",
                 transform=None, pre_transform=None, pre_filter=None):
        self.k = k
        self.files = [files] if isinstance(files,str) else files #"../data/ecolik12_ont_0001.fastq"
        self.files.sort()
        self.format = format
        self.idx2metadata = self.get_idx2metadata()
        self.filename2label = {Path(f).stem: num for num,f in enumerate(self.files)}
        super().__init__(root=root, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)
        

    @property
    def raw_file_names(self,):
        return self.files

    @property
    def processed_file_names(self,):

        processed_filename = [f"{m.filename}/data_{m.idx}.pt" for m in self.idx2metadata]
        
        return processed_filename

    def process(self):
        idx = 0

        for path_file in self.files:
        
            records = SeqIO.parse(path_file, format=self.format)
            filename = Path(path_file).stem
            ommit_ids = self.ommit_seqids(path_file)
            label = self.filename2label[filename]

            for r in tqdm(records, desc=f"{filename}"):

                if r.id in ommit_ids:
                    continue

                # Read data from `raw_path`.
                dbg = DeBruijnGraph(sequence=r.seq, k=self.k)
                kmer_count = self.count_kmers(r.seq, self.k)
                data = self.create_torch_graph(dbg.nodes, dbg.edges, kmer_count, label)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                Path(self.processed_dir).joinpath(f"{filename}").mkdir(exist_ok=True, parents=True)
                torch.save(data, Path(self.processed_dir).joinpath(f"{filename}/data_{idx}.pt"))
                idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        m = self.idx2metadata[idx]
        data = torch.load(Path(self.processed_dir).joinpath(f'{m.filename}/data_{m.idx}.pt'))
        return data
    
    def create_torch_graph(self, nodes: set, edges: list, kmer_count: dict, label: int):

        nodes = list(nodes)
        nodes.sort()
        idx_nodes = {kmer: idx  for idx, kmer in  enumerate(nodes)}
        
        # nodes
        # tot_kmers = sum(kmer_count.values())
        max_count = max(kmer_count.values())
        feat_nodes = [ [kmer_count[kmer]/max_count] for kmer in nodes]
        x = torch.tensor(feat_nodes, dtype=torch.float)

        # edges    
        edges =  list(map( lambda x: (idx_nodes[x[0]],idx_nodes[x[1]]) , edges))
        edge_index = torch.tensor(edges, dtype=torch.long)    

        # label 
        y = torch.tensor([label], dtype=torch.long) 

        return Data(x=x, edge_index=edge_index.t().contiguous(), y = y)

    def count_kmers(self, seq, k):

        kmer_count = defaultdict(int)
        for pos in range(len(seq)-k+1):
            kmer = seq[pos:pos+k]
            kmer_count[kmer] += 1

        return kmer_count    
    
    def get_idx2metadata(self,):

        idx2metadata = []
        idx = 0
        for f in self.files:
            records = SeqIO.parse(f, format=self.format)
            ommit_ids = self.ommit_seqids(f)
            filename = Path(f).stem
            for r in records:
                if r.id not in ommit_ids:
                    idx2metadata.append(
                        Seq(filename, r.id, idx)
                    )
                    idx += 1

        return idx2metadata
    
    def ommit_seqids(self,f):
        records = SeqIO.parse(f, format=self.format)
        l = []
        for f in records:
            l.append(f.id)
        count = Counter(l)
        
        return list(dict(filter(lambda t: t[1]>1, count.items())).keys())