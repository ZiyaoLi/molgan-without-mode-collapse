import tensorflow as tf

from utils.sparse_molecular_dataset import SparseMolecularDataset
from utils.trainer import Trainer
from utils.utils import *

from models.gan import *
from models import *

from optimizers.gan import GraphGANOptimizer

DECODER_UNITS = (128, 256, 512)                   # z = Dense(z, dim=units_k)^{(k)}
DISCRIM_UNITS = ((64, 32), 128, (128,))           # (GCN units, Readout units, MLP units)
Z_DIM = 32

METRIC = 'validity,qed'  # candidates in 'np,logp,sas,qed,novelty,dc,unique,diversity,validity'
BATCH_DIM = 32
LA = 0.05
DROPOUT = 0
N_CRITIC = 5
N_SAMPLES = 5000
EPOCHS = 1
SAVE_EVERY = None


def train_fetch_dict(i, steps, epoch, epochs, min_epochs, model, optimizer):
    la = 1 if epoch < epochs / 2 else LA
    gan_ops = [optimizer.train_step_G] if i % N_CRITIC == 0 else [optimizer.train_step_D]
    rwd_ops = [optimizer.train_step_V] if i % N_CRITIC == 0 and LA < 1 and la == 1 else []
    return gan_ops + rwd_ops


def train_feed_dict(i, steps, epoch, epochs, min_epochs, model, optimizer, batch_dim):
    molsR, _, _, a, x, _, _, _, _ = data.next_train_batch(batch_dim)
    z_embeds = model.sample_z(batch_dim)

    la = 1 if epoch < epochs / 2 else LA

    if i % N_CRITIC == 0 and LA < 1 and la == 1:  # reward net shall be optimized
        rewardR = reward(molsR, METRIC, data)
        # generate fake mols to train the reward net (as augmentation; determining validity et al)
        n, e = session.run([model.nodes_gumbel_argmax, model.edges_gumbel_argmax],
                           feed_dict={model.training: False, model.embeddings: z_embeds})
        n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)
        molsF = [data.matrices2mol(n_, e_, strict=True) for n_, e_ in zip(n, e)]
        rewardF = reward(molsF, METRIC, data)

        feed_dict = {model.edges_labels: a,
                     model.nodes_labels: x,
                     model.embeddings: z_embeds,
                     model.rewardR: rewardR,
                     model.rewardF: rewardF,
                     model.training: True,
                     model.dropout_rate: DROPOUT,
                     optimizer.la: la if epoch > 0 else 1.0}

    else:
        feed_dict = {model.edges_labels: a,
                     model.nodes_labels: x,
                     model.embeddings: z_embeds,
                     model.training: True,
                     model.dropout_rate: DROPOUT,
                     optimizer.la: 1.0}

    return feed_dict


def eval_fetch_dict(i, epochs, min_epochs, model, optimizer):
    return {'loss D': optimizer.loss_D, 'loss G': optimizer.loss_G,
            'loss RL': optimizer.loss_RL, 'loss V': optimizer.loss_V,
            'la': optimizer.la}


def eval_feed_dict(i, epochs, min_epochs, model, optimizer, batch_dim):
    mols, _, _, a, x, _, _, _, _ = data.next_validation_batch()
    embeddings = model.sample_z(a.shape[0])

    rewardR = reward(mols, METRIC, data)

    n, e = session.run([model.nodes_gumbel_argmax, model.edges_gumbel_argmax],
                       feed_dict={model.training: False, model.embeddings: embeddings})
    n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)
    mols = [data.matrices2mol(n_, e_, strict=True) for n_, e_ in zip(n, e)]

    record_mols_and_smiles(mols, i)

    rewardF = reward(mols, METRIC, data)

    feed_dict = {model.edges_labels: a,
                 model.nodes_labels: x,
                 model.embeddings: embeddings,
                 model.rewardR: rewardR,
                 model.rewardF: rewardF,
                 model.training: False}
    return feed_dict


def _eval_update(i, epochs, min_epochs, model, optimizer, batch_dim, eval_batch):
    mols = samples(data, model, session, model.sample_z(N_SAMPLES), sample=True)
    m0, m1 = all_scores(mols, data, norm=True)
    m0 = {k: np.array(v)[np.nonzero(v)].mean() for k, v in m0.items()}
    m0.update(m1)
    return m0


test_fetch_dict = eval_fetch_dict
test_feed_dict = eval_feed_dict
_test_update = _eval_update


def record_mols_and_smiles(mols, epoch, draw=True):
    smiles = list(map(lambda x: Chem.MolToSmiles(x) if MolecularMetrics.valid_lambda(x) else '', mols))
    valid_unique_smiles = list(set([s for s in smiles if len(s) and '*' not in s and '.' not in s]))
    np.random.shuffle(smiles)
    np.random.shuffle(valid_unique_smiles)

    smiles = smiles[:30]
    valid_unique_smiles = valid_unique_smiles[:30]

    with open(save_dir + '/sample.smiles', 'a') as fh:
        fh.write('START Epoch %03d:\n' % epoch)
        fh.write('\n'.join(smiles))
        fh.write('END Epoch %03d:\n' % epoch)
    with open(save_dir + '/val_unique.smiles', 'a') as fh:
        fh.write('START Epoch %03d:\n' % epoch)
        fh.write('\n'.join(valid_unique_smiles))
        fh.write('END Epoch %03d:\n' % epoch)

    draw_mols = list(map(lambda x: Chem.MolFromSmiles(x) if len(x) else Chem.MolFromSmiles('C'), smiles))
    draw_vu_mols = list(map(lambda x: Chem.MolFromSmiles(x), valid_unique_smiles))

    if draw:
        from rdkit.Chem import Draw
        if draw and len(draw_mols):
            img = Draw.MolsToGridImage(draw_mols, molsPerRow=5, subImgSize=(300, 300))
            img.save(save_dir + "/samples_%03d.png" % epoch)
        if len(draw_vu_mols):
            img = Draw.MolsToGridImage(draw_vu_mols, molsPerRow=5, subImgSize=(300, 300))
            img.save(save_dir + "/val_unique_%03d.png" % epoch)


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

    Model, Discr, batch_discr, N_CRITIC = MODELS[args.model]

    for i in range(args.replicas):

        tf.reset_default_graph()  # rebuild comp. graph across replicas

        data = SparseMolecularDataset()
        data.load('data/gdb9_9nodes.sparsedataset')

        steps = (len(data) // BATCH_DIM)

        save_dir = args.name + ('_%02d' % i if args.replicas > 1 else '')
        import os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # model
        model = Model(data.vertexes, data.bond_num_types, data.atom_num_types, Z_DIM,
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

        trainer.train(batch_dim=BATCH_DIM, epochs=EPOCHS, steps=steps,
                      train_fetch_dict=train_fetch_dict, train_feed_dict=train_feed_dict,
                      eval_fetch_dict=eval_fetch_dict, eval_feed_dict=eval_feed_dict,
                      test_fetch_dict=test_fetch_dict, test_feed_dict=test_feed_dict,
                      _eval_update=_eval_update, _test_update=_test_update,
                      save_every=SAVE_EVERY, directory=save_dir)

        session.close()

        del model, optimizer, trainer, data


