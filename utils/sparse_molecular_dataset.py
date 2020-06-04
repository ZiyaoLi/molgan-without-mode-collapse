import pickle
import numpy as np

from rdkit import Chem

# if __name__ == '__main__':
#     from progress_bar import ProgressBar
#     from utils import reward
# else:
from utils.progress_bar import ProgressBar
from utils.utils import reward

from datetime import datetime


class SparseMolecularDataset():

    def load(self, filename, proportion=1):

        with open(filename, 'rb') as f:
            self.__dict__.update(pickle.load(f))

        if proportion < 1:
            self.train_idx = np.random.choice(
                self.train_idx, int(len(self.train_idx) * proportion), replace=False)
            self.validation_idx = np.random.choice(
                self.validation_idx, int(len(self.validation_idx) * proportion), replace=False)
            self.test_idx = np.random.choice(
                self.test_idx, int(len(self.test_idx) * proportion), replace=False)

        self.train_count = len(self.train_idx)
        self.validation_count = len(self.validation_idx)
        self.test_count = len(self.test_idx)

        self.__len = self.train_count + self.validation_count + self.test_count

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def generate(self, filename, add_h=False, filters=lambda x: True, size=None, validation=0.1, test=0.1):
        self.log('Extracting {}..'.format(filename))

        if filename.endswith('.sdf'):
            self.data = list(filter(lambda x: x is not None, Chem.SDMolSupplier(filename)))
        elif filename.endswith('.smi'):
            self.data = [Chem.MolFromSmiles(line) for line in open(filename, 'r').readlines()]

        self.data = list(map(Chem.AddHs, self.data)) if add_h else self.data
        self.data = list(filter(filters, self.data))
        self.data = self.data[:size]

        self.log('Extracted {} out of {} molecules {}adding Hydrogen!'.format(len(self.data),
                                                                              len(Chem.SDMolSupplier(filename)),
                                                                              '' if add_h else 'not '))

        self._generate_encoders_decoders()
        self._generate_AX()

        # it contains the all the molecules stored as rdkit.Chem objects
        self.data = np.array(self.data)

        # it contains the all the molecules stored as SMILES strings
        self.smiles = np.array(self.smiles)

        # a (N, L) matrix where N is the length of the dataset and each L-dim vector contains the 
        # indices corresponding to a SMILE sequences with padding wrt the max length of the longest 
        # SMILES sequence in the dataset (see self._genS)
        self.data_S = np.stack(self.data_S)

        # a (N, 9, 9) tensor where N is the length of the dataset and each 9x9 matrix contains the 
        # indices of the positions of the ones in the one-hot representation of the adjacency tensor
        # (see self._genA)
        self.data_A = np.stack(self.data_A)

        # a (N, 9) matrix where N is the length of the dataset and each 9-dim vector contains the 
        # indices of the positions of the ones in the one-hot representation of the annotation matrix
        # (see self._genX)
        self.data_X = np.stack(self.data_X)

        # a (N, 9) matrix where N is the length of the dataset and each  9-dim vector contains the 
        # diagonal of the correspondent adjacency matrix
        self.data_D = np.stack(self.data_D)

        # a (N, F) matrix where N is the length of the dataset and each F vector contains features 
        # of the correspondent molecule (see self._genF)
        self.data_F = np.stack(self.data_F)

        # a (N, 9) matrix where N is the length of the dataset and each  9-dim vector contains the
        # eigenvalues of the correspondent Laplacian matrix
        self.data_Le = np.stack(self.data_Le)

        # a (N, 9, 9) matrix where N is the length of the dataset and each  9x9 matrix contains the 
        # eigenvectors of the correspondent Laplacian matrix
        self.data_Lv = np.stack(self.data_Lv)

        self.vertexes = self.data_F.shape[-2]
        self.features = self.data_F.shape[-1]

        self._generate_train_validation_test(validation, test)

    def _generate_encoders_decoders(self):
        self.log('Creating atoms encoder and decoder..')
        atom_labels = sorted(set([atom.GetAtomicNum() for mol in self.data for atom in mol.GetAtoms()] + [0]))
        self.atom_encoder_m = {l: i for i, l in enumerate(atom_labels)}
        self.atom_decoder_m = {i: l for i, l in enumerate(atom_labels)}
        self.atom_num_types = len(atom_labels)
        self.log('Created atoms encoder and decoder with {} atom types and 1 PAD symbol!'.format(
            self.atom_num_types - 1))

        self.log('Creating bonds encoder and decoder..')
        bond_labels = [Chem.rdchem.BondType.ZERO] + list(sorted(set(bond.GetBondType()
                                                                    for mol in self.data
                                                                    for bond in mol.GetBonds())))

        self.bond_encoder_m = {l: i for i, l in enumerate(bond_labels)}
        self.bond_decoder_m = {i: l for i, l in enumerate(bond_labels)}
        self.bond_num_types = len(bond_labels)
        self.log('Created bonds encoder and decoder with {} bond types and 1 PAD symbol!'.format(
            self.bond_num_types - 1))

        self.log('Creating SMILES encoder and decoder..')
        smiles_labels = ['E'] + list(set(c for mol in self.data for c in Chem.MolToSmiles(mol)))
        self.smiles_encoder_m = {l: i for i, l in enumerate(smiles_labels)}
        self.smiles_decoder_m = {i: l for i, l in enumerate(smiles_labels)}
        self.smiles_num_types = len(smiles_labels)
        self.log('Created SMILES encoder and decoder with {} types and 1 PAD symbol!'.format(
            self.smiles_num_types - 1))

    def _generate_AX(self):
        self.log('Creating features and adjacency matrices..')
        pr = ProgressBar(60, len(self.data))

        data = []
        smiles = []
        data_S = []
        data_A = []
        data_X = []
        data_D = []
        data_F = []
        data_Le = []
        data_Lv = []

        max_length = max(mol.GetNumAtoms() for mol in self.data)
        max_length_s = max(len(Chem.MolToSmiles(mol)) for mol in self.data)

        for i, mol in enumerate(self.data):
            A = self._genA(mol, connected=True, max_length=max_length)
            D = np.count_nonzero(A, -1)
            if A is not None:
                data.append(mol)
                smiles.append(Chem.MolToSmiles(mol))
                data_S.append(self._genS(mol, max_length=max_length_s))
                data_A.append(A)
                data_X.append(self._genX(mol, max_length=max_length))
                data_D.append(D)
                data_F.append(self._genF(mol, max_length=max_length))

                L = D - A
                Le, Lv = np.linalg.eigh(L)

                data_Le.append(Le)
                data_Lv.append(Lv)

            pr.update(i + 1)

        self.log(date=False)
        self.log('Created {} features and adjacency matrices  out of {} molecules!'.format(len(data),
                                                                                           len(self.data)))

        self.data = data
        self.smiles = smiles
        self.data_S = data_S
        self.data_A = data_A
        self.data_X = data_X
        self.data_D = data_D
        self.data_F = data_F
        self.data_Le = data_Le
        self.data_Lv = data_Lv
        self.__len = len(self.data)

    def _genA(self, mol, connected=True, max_length=None):

        max_length = max_length if max_length is not None else mol.GetNumAtoms()

        A = np.zeros(shape=(max_length, max_length), dtype=np.int32)

        begin, end = [b.GetBeginAtomIdx() for b in mol.GetBonds()], [b.GetEndAtomIdx() for b in mol.GetBonds()]
        bond_type = [self.bond_encoder_m[b.GetBondType()] for b in mol.GetBonds()]

        A[begin, end] = bond_type
        A[end, begin] = bond_type

        degree = np.sum(A[:mol.GetNumAtoms(), :mol.GetNumAtoms()], axis=-1)

        return A if connected and (degree > 0).all() else None

    def _genX(self, mol, max_length=None):

        max_length = max_length if max_length is not None else mol.GetNumAtoms()

        return np.array([self.atom_encoder_m[atom.GetAtomicNum()] for atom in mol.GetAtoms()] + [0] * (
                    max_length - mol.GetNumAtoms()), dtype=np.int32)

    def _genS(self, mol, max_length=None):

        max_length = max_length if max_length is not None else len(Chem.MolToSmiles(mol))

        return np.array([self.smiles_encoder_m[c] for c in Chem.MolToSmiles(mol)] + [self.smiles_encoder_m['E']] * (
                    max_length - len(Chem.MolToSmiles(mol))), dtype=np.int32)

    def _genF(self, mol, max_length=None):

        max_length = max_length if max_length is not None else mol.GetNumAtoms()

        features = np.array([[*[a.GetDegree() == i for i in range(5)],
                              *[a.GetExplicitValence() == i for i in range(9)],
                              *[int(a.GetHybridization()) == i for i in range(1, 7)],
                              *[a.GetImplicitValence() == i for i in range(9)],
                              a.GetIsAromatic(),
                              a.GetNoImplicit(),
                              *[a.GetNumExplicitHs() == i for i in range(5)],
                              *[a.GetNumImplicitHs() == i for i in range(5)],
                              *[a.GetNumRadicalElectrons() == i for i in range(5)],
                              a.IsInRing(),
                              *[a.IsInRingSize(i) for i in range(2, 9)]] for a in mol.GetAtoms()], dtype=np.int32)

        return np.vstack((features, np.zeros((max_length - features.shape[0], features.shape[1]))))

    def matrices2mol(self, node_labels, edge_labels, strict=False):
        mol = Chem.RWMol()

        for node_label in node_labels:
            mol.AddAtom(Chem.Atom(self.atom_decoder_m[node_label]))

        for start, end in zip(*np.nonzero(edge_labels)):
            if start > end:
                mol.AddBond(int(start), int(end), self.bond_decoder_m[edge_labels[start, end]])

        if strict:
            try:
                Chem.SanitizeMol(mol)
            except:
                mol = None

        return mol

    def seq2mol(self, seq, strict=False):
        mol = Chem.MolFromSmiles(''.join([self.smiles_decoder_m[e] for e in seq if e != 0]))

        if strict:
            try:
                Chem.SanitizeMol(mol)
            except:
                mol = None

        return mol

    def _generate_train_validation_test(self, n_valid, n_test):

        self.log('Creating train, validation and test sets..')

        n_valid = int(n_valid * len(self))
        n_test = int(n_test * len(self))
        n_train = len(self) - n_valid - n_test

        self.all_idx = np.random.permutation(len(self))
        self.train_idx = self.all_idx[:n_train]
        self.validation_idx = self.all_idx[n_train: n_train + n_valid]
        self.test_idx = self.all_idx[n_train + n_valid:]

        self.train_counter = 0
        self.validation_counter = 0
        self.test_counter = 0

        self.train_count = n_train
        self.validation_count = n_valid
        self.test_count = n_test

        self.log('Created train ({} items), validation ({} items) and test ({} items) sets!'.format(
            n_train, n_valid, n_test))

    def _next_batch(self, counter, count, idx, batch_size):
        if batch_size is not None:
            if counter + batch_size >= count:
                counter = 0
                np.random.shuffle(idx)

            output = [obj[idx[counter:counter + batch_size]]
                      for obj in (self.data, self.smiles, self.data_S, self.data_A, self.data_X,
                                  self.data_D, self.data_F, self.data_Le, self.data_Lv)]

            counter += batch_size
        else:
            output = [obj[idx] for obj in (self.data, self.smiles, self.data_S, self.data_A, self.data_X,
                                           self.data_D, self.data_F, self.data_Le, self.data_Lv)]

        return [counter] + output

    def next_train_batch(self, batch_size=None):
        out = self._next_batch(counter=self.train_counter, count=self.train_count,
                               idx=self.train_idx, batch_size=batch_size)
        self.train_counter = out[0]
        return out[1:]

    def next_validation_batch(self, batch_size=None):
        out = self._next_batch(counter=self.validation_counter, count=self.validation_count,
                               idx=self.validation_idx, batch_size=batch_size)
        self.validation_counter = out[0]
        return out[1:]

    def next_test_batch(self, batch_size=None):
        out = self._next_batch(counter=self.test_counter, count=self.test_count,
                               idx=self.test_idx, batch_size=batch_size)
        self.test_counter = out[0]
        return out[1:]

    @staticmethod
    def log(msg='', date=True):
        print(str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + ' ' + str(msg) if date else str(msg))

    def __len__(self):
        return self.__len

    def plot_samples(self, filename, n_samples, mols_per_row=5, sub_img_size=(300, 300)):
        idx = np.random.choice(self.train_idx, n_samples, False)
        mols = self.data[idx]
        from rdkit.Chem import Draw
        img = Draw.MolsToGridImage(mols, molsPerRow=mols_per_row, subImgSize=sub_img_size)
        img.save(filename)

    def write_sample_smiles(self, filename, n_samples):
        idx = np.random.choice(self.train_idx, n_samples, False)
        smiles = self.smiles[idx]
        with open(filename, 'a') as fh:
            fh.write('\n'.join(smiles))
            fh.write('\n')


class SparseMolecularDatasetWithRewards(SparseMolecularDataset):
    def load(self, filename, metric='validity,qed', conditional_rate=0.3, proportion=1):
        super(SparseMolecularDatasetWithRewards, self).load(filename, proportion)

        if metric != self.metric:  # a new metric required.
            self.metric = metric
            self._genReward(metric)
            self._generate_conditional_train_validation_test(conditional_rate)

        if proportion < 1:
            self.cond_train_idx = np.random.choice(
                self.cond_train_idx, int(len(self.cond_train_idx) * proportion), replace=False)
            self.cond_validation_idx = np.random.choice(
                self.cond_validation_idx, int(len(self.cond_validation_idx) * proportion), replace=False)
            self.cond_test_idx = np.random.choice(
                self.cond_test_idx, int(len(self.cond_test_idx) * proportion), replace=False)

        self.cond_train_count = len(self.cond_train_idx)
        self.cond_validation_count = len(self.cond_validation_idx)
        self.cond_test_count = len(self.cond_test_idx)

        self.cond_len = self.train_count + self.validation_count + self.test_count

    def generate(self, filename, metric='validity,qed', conditional_rate=0.3,
                 add_h=False, filters=lambda x: True, size=None, validation=0.1, test=0.1):
        super(SparseMolecularDatasetWithRewards, self).generate(filename, add_h, filters, size, validation, test)

        self.metric = metric
        self._genReward(metric)

        self._generate_conditional_train_validation_test(conditional_rate)

    def _genReward(self, metric, batch_size=15):
        self.log('Calculating molecule rewards..')

        pr = ProgressBar(60, len(self.data))

        i = 0
        self.data_rwd = []
        while i < len(self.data):
            mols = self.data[i: i + batch_size]
            rwds = reward(mols, metric, self).reshape(-1)
            self.data_rwd.append(rwds)
            i += batch_size
            pr.update(min(i, len(self.data)))
        self.data_rwd = np.concatenate(self.data_rwd, -1)

    def _generate_conditional_train_validation_test(self, conditional_rate):
        train_rewards = self.data_rwd[self.train_idx]
        train_order = np.argsort(train_rewards)  # increasing
        self.cond_train_idx = self.train_idx[
            train_order[int((1 - conditional_rate) * len(self.train_idx)):]]

        validation_rewards = self.data_rwd[self.validation_idx]
        validation_order = np.argsort(validation_rewards)  # increasing
        self.cond_validation_idx = self.validation_idx[
            validation_order[int((1 - conditional_rate) * len(self.validation_idx)):]]

        test_rewards = self.data_rwd[self.test_idx]
        test_order = np.argsort(test_rewards)  # increasing
        self.cond_test_idx = self.test_idx[
            test_order[int((1 - conditional_rate) * len(self.test_idx)):]]

        self.cond_train_counter = 0
        self.cond_validation_counter = 0
        self.cond_test_counter = 0

        self.cond_train_count = len(self.cond_train_idx)
        self.cond_validation_count = len(self.cond_validation_idx)
        self.cond_test_count = len(self.cond_test_idx)

        pass

    def next_cond_train_batch(self, batch_size=None):
        out = self._next_batch(counter=self.cond_train_counter, count=self.cond_train_count,
                               idx=self.cond_train_idx, batch_size=batch_size)
        self.cond_train_counter = out[0]
        return out[1:]

    def next_cond_validation_batch(self, batch_size=None):
        out = self._next_batch(counter=self.cond_validation_counter, count=self.cond_validation_count,
                               idx=self.cond_validation_idx, batch_size=batch_size)
        self.cond_validation_counter = out[0]
        return out[1:]

    def next_cond_test_batch(self, batch_size=None):
        out = self._next_batch(counter=self.test_counter, count=self.test_count,
                               idx=self.test_idx, batch_size=batch_size)
        self.cond_test_counter = out[0]
        return out[1:]
