from utils.sparse_molecular_dataset import SparseMolecularDataset, SparseMolecularDatasetWithRewards

if __name__ == '__main__':
    data = SparseMolecularDataset()
    data.generate('data/gdb9.sdf', filters=lambda x: x.GetNumAtoms() <= 9, size=100)  ## QM9 filter
    data.save('data/gdb9_9nodes.sparsedataset')

    data = SparseMolecularDatasetWithRewards()
    data.generate('data/gdb9.sdf', filters=lambda x: x.GetNumAtoms() <= 9, size=100)
    data.save('data/gdb9_9nodes.rewarddataset')

