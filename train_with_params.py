import tensorflow as tf

from utils.sparse_molecular_dataset import SparseMolecularDataset
from utils.trainer import Trainer
from utils.utils import *

from models.gan import *
from models import *

from optimizers.gan import GraphGANOptimizer

DECODER_UNITS = (128, 256, 512)                   # z = Dense(z, dim=units_k)^{(k)}
DISCRIM_UNITS = ((64, 32), 128, (128,))           # (GCN units, Readout units, MLP units)
batch_dim = 24
LA = 0.05
dropout = 0
n_critic = 5
metric = 'validity,qed'  # all = 'np,logp,sas,qed,novelty,dc,unique,diversity,validity'
n_samples = 5000
z_dim = 32
epochs = 64
save_every = 5

data = SparseMolecularDataset()
data.load('data/gdb9_9nodes.sparsedataset')

steps = (len(data) // batch_dim)


def train_fetch_dict(i, steps, epoch, epochs, min_epochs, model, optimizer):
    la = 1 if epoch < epochs / 2 else LA
    a = [optimizer.train_step_G] if i % n_critic == 0 else [optimizer.train_step_D]
    b = [optimizer.train_step_V] if i % n_critic == 0 and LA < 1 and la == 1 else []
    return a + b


def train_feed_dict(i, steps, epoch, epochs, min_epochs, model, optimizer, batch_dim):
    mols, _, _, a, x, _, _, _, _ = data.next_train_batch(batch_dim)
    embeddings = model.sample_z(batch_dim)

    la = 1 if epoch < epochs / 2 else LA

    if LA < 1:  # RL is calculated anyway

        if i % n_critic == 0:
            rewardR = reward(mols)

            n, e = session.run([model.nodes_gumbel_argmax, model.edges_gumbel_argmax],
                               feed_dict={model.training: False, model.embeddings: embeddings})
            n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)
            mols = [data.matrices2mol(n_, e_, strict=True) for n_, e_ in zip(n, e)]

            rewardF = reward(mols)

            feed_dict = {model.edges_labels: a,
                         model.nodes_labels: x,
                         model.embeddings: embeddings,
                         model.rewardR: rewardR,
                         model.rewardF: rewardF,
                         model.training: True,
                         model.dropout_rate: dropout,
                         optimizer.la: la if epoch > 0 else 1.0}

        else:
            feed_dict = {model.edges_labels: a,
                         model.nodes_labels: x,
                         model.embeddings: embeddings,
                         model.training: True,
                         model.dropout_rate: dropout,
                         optimizer.la: la if epoch > 0 else 1.0}
    else:
        feed_dict = {model.edges_labels: a,
                     model.nodes_labels: x,
                     model.embeddings: embeddings,
                     model.training: True,
                     model.dropout_rate: dropout,
                     optimizer.la: 1.0}

    return feed_dict


def eval_fetch_dict(i, epochs, min_epochs, model, optimizer):
    return {'loss D': optimizer.loss_D, 'loss G': optimizer.loss_G,
            'loss RL': optimizer.loss_RL, 'loss V': optimizer.loss_V,
            'la': optimizer.la}


def eval_feed_dict(i, epochs, min_epochs, model, optimizer, batch_dim):
    mols, _, _, a, x, _, _, _, _ = data.next_validation_batch()
    embeddings = model.sample_z(a.shape[0])

    rewardR = reward(mols)

    n, e = session.run([model.nodes_gumbel_argmax, model.edges_gumbel_argmax],
                       feed_dict={model.training: False, model.embeddings: embeddings})
    n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)
    mols = [data.matrices2mol(n_, e_, strict=True) for n_, e_ in zip(n, e)]

    regist_mols(mols, i)

    rewardF = reward(mols)

    feed_dict = {model.edges_labels: a,
                 model.nodes_labels: x,
                 model.embeddings: embeddings,
                 model.rewardR: rewardR,
                 model.rewardF: rewardF,
                 model.training: False}
    return feed_dict


def test_fetch_dict(model, optimizer):
    return {'loss D': optimizer.loss_D, 'loss G': optimizer.loss_G,
            'loss RL': optimizer.loss_RL, 'loss V': optimizer.loss_V,
            'la': optimizer.la}


def test_feed_dict(model, optimizer, batch_dim):
    mols, _, _, a, x, _, _, _, _ = data.next_test_batch()
    embeddings = model.sample_z(a.shape[0])

    rewardR = reward(mols)

    n, e = session.run([model.nodes_gumbel_argmax, model.edges_gumbel_argmax],
                       feed_dict={model.training: False, model.embeddings: embeddings})
    n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)
    mols = [data.matrices2mol(n_, e_, strict=True) for n_, e_ in zip(n, e)]

    rewardF = reward(mols)

    feed_dict = {model.edges_labels: a,
                 model.nodes_labels: x,
                 model.embeddings: embeddings,
                 model.rewardR: rewardR,
                 model.rewardF: rewardF,
                 model.training: False}
    return feed_dict


def regist_mols(mols, epoch):
    smiles = list(map(lambda x: Chem.MolToSmiles(x) if MolecularMetrics.valid_lambda(x) else '', mols))
    valid_unique_smiles = list(set([s for s in smiles if len(s) and '*' not in s and '.' not in s]))
    np.random.shuffle(smiles)
    np.random.shuffle(valid_unique_smiles)

    smiles = smiles[:30]
    valid_unique_smiles = valid_unique_smiles[:30]

    draw_mols = list(map(lambda x: Chem.MolFromSmiles(x) if len(x) else Chem.MolFromSmiles('C'), smiles))
    draw_vu_mols = list(map(lambda x: Chem.MolFromSmiles(x), valid_unique_smiles))

    if len(draw_mols):
        from rdkit.Chem import Draw
        img = Draw.MolsToGridImage(draw_mols, molsPerRow=5, subImgSize=(300, 300))
        img.save(save_dir + "/samples_%03d.png" % epoch)
    if len(draw_vu_mols):
        from rdkit.Chem import Draw
        img = Draw.MolsToGridImage(draw_vu_mols, molsPerRow=5, subImgSize=(300, 300))
        img.save(save_dir + "/val_unique_%03d.png" % epoch)

    with open(save_dir + '/sample.smiles', 'a') as fh:
        fh.write('Epoch %03d:\n' % epoch)
        fh.write('\n'.join(smiles))
        fh.write('#\n')
    with open(save_dir + '/val_unique.smiles', 'a') as fh:
        fh.write('Epoch %03d:\n' % epoch)
        fh.write('\n'.join(valid_unique_smiles))
        fh.write('#\n')


def reward(mols):
    rr = 1.
    for m in ('logp,sas,qed,unique' if metric == 'all' else metric).split(','):

        if m == 'np':
            rr *= MolecularMetrics.natural_product_scores(mols, norm=True)
        elif m == 'logp':
            rr *= MolecularMetrics.water_octanol_partition_coefficient_scores(mols, norm=True)
        elif m == 'sas':
            rr *= MolecularMetrics.synthetic_accessibility_score_scores(mols, norm=True)
        elif m == 'qed':
            rr *= MolecularMetrics.quantitative_estimation_druglikeness_scores(mols, norm=True)
        elif m == 'novelty':
            rr *= MolecularMetrics.novel_scores(mols, data)
        elif m == 'dc':
            rr *= MolecularMetrics.drugcandidate_scores(mols, data)
        elif m == 'unique':
            rr *= MolecularMetrics.unique_scores(mols)
        elif m == 'diversity':
            rr *= MolecularMetrics.diversity_scores(mols, data)
        elif m == 'validity':
            rr *= MolecularMetrics.valid_scores(mols)
        else:
            raise RuntimeError('{} is not defined as a metric'.format(m))

    return rr.reshape(-1, 1)


def _eval_update(i, epochs, min_epochs, model, optimizer, batch_dim, eval_batch):
    mols = samples(data, model, session, model.sample_z(n_samples), sample=True)
    m0, m1 = all_scores(mols, data, norm=True)
    m0 = {k: np.array(v)[np.nonzero(v)].mean() for k, v in m0.items()}
    m0.update(m1)
    return m0


def _test_update(model, optimizer, batch_dim, test_batch):
    mols = samples(data, model, session, model.sample_z(n_samples), sample=True)
    m0, m1 = all_scores(mols, data, norm=True)
    m0 = {k: np.array(v)[np.nonzero(v)].mean() for k, v in m0.items()}
    m0.update(m1)
    return m0


def Argparser():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-n", "--name", type=str, required=True,
                        help='name of the experiment.')
    parser.add_argument('-m', '--model', type=str, help='model name.')
    parser.add_argument('-r', '--replicas', type=int, default=1, help='model name.')
    return parser


MODELS = {  # Model, Discr, use_batch_discr, n_critics
    'molgan': (GraphGANModel, encoder_rgcn, False, 5),
    'molgan+batch': (GraphGANModel, encoder_rgcn, True, 3),
    'pmax': (PacGANModel, encoder_rgcn, False, 3),
    'pe1': (PacElas1Model, encoder_rgcn, False, 5),
    'pe2': (PacElas2Model, encoder_rgcn, False, 5),
    'pem': (PacMeanElas2Model, encoder_rgcn, False, 3),
    'pmerge': (PacStatsMergeModel, encoder_rgcn, False, 3),
    # 'gat': (GraphGANModel, encoder_gat, False),
    # 'flat_gat': (GraphGANModel, encoder_flat_gat, False),
}


if __name__ == '__main__':
    parser = Argparser()
    args = parser.parse_args()

    Model, Discr, batch_discr = MODELS[args.model]

    for i in range(args.replicas):
        
        save_dir = args.name + ('_%02d' % i if args.replicas > 1 else '')
        import os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # model
        model = Model(data.vertexes, data.bond_num_types, data.atom_num_types, z_dim,
                      decoder_units=DECODER_UNITS,
                      discriminator_units=DISCRIM_UNITS,
                      decoder=decoder_adj,
                      discriminator=Discr,
                      soft_gumbel_softmax=False,
                      hard_gumbel_softmax=False,
                      batch_discriminator=batch_discr)

        # optimizer
        optimizer = GraphGANOptimizer(model, learning_rate=1e-3, feature_matching=False)

        # session
        session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        session.run(tf.global_variables_initializer())

        # trainer
        trainer = Trainer(model, optimizer, session)

        print('Parameters: {}'.format(np.sum([np.prod(e.shape) for e in session.run(tf.trainable_variables())])))

        trainer.train(batch_dim=batch_dim, epochs=epochs, steps=steps,
                      train_fetch_dict=train_fetch_dict, train_feed_dict=train_feed_dict,
                      eval_fetch_dict=eval_fetch_dict, eval_feed_dict=eval_feed_dict,
                      test_fetch_dict=test_fetch_dict, test_feed_dict=test_feed_dict,
                      _eval_update=_eval_update, _test_update=_test_update,
                      save_every=save_every, directory=save_dir)

