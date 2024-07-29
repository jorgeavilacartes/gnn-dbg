# gnn-dbg

Classification of genome sequences: CNN vs GNN

___
### 1. Create conda environment
```bash
conda env create -n pyg -f envs/genv-gpu.yml
```

### 2. `notebooks/` 
- `create-fcgr.ipynb` from a fasta file with a set of sequences, it creates the FCGR in numpy format for each sequence
- `fake-fasta.ipynb` creates a fasta file with random sequences
- `load-dbg.ipynb` example of de Bruijn Graph DataLoader
- `subsample-fasta.ipynb` extract a subset of sequences from a fasta file
- `train-fcgr.ipynb` train and tes for CNN + FCGR
- `train-dbg.ipynb` train and test for GNN + dBG

### 3. `src/`
```bash
src
├── datasets                         # torch geometric dataset for de Bruijn Graph, and torch dataset for FCGR
│    
├── graphs                           # de Bruijn Graph
│
└── models                           # CNN for FCGR, and GNN for de Bruijn Graphs
```