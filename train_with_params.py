import tensorflow as tf

from utils.sparse_molecular_dataset import SparseMolecularDataset
from utils.trainer import Trainer
from utils.utils import *

from models.gan import *
from models import *

from optimizers.gan import GraphGANOptimizer

from example import train_feed_dict, train_fetch_dict, \
    eval_feed_dict, eval_fetch_dict, test_feed_dict, test_fetch_dict, \
    reward, _eval_update, _test_update

DECODER_UNITS = (128, 256, 512)                   # z = Dense(z, dim=units_k)^{(k)}
DISCRIM_UNITS = ((64, 32), 128, (128,))           # (GCN units, Readout units, MLP units)
batch_dim = 32
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


def Argparser():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-n", "--name", type=str, required=True,
                        help='name of the experiment.')
    parser.add_argument('-m', '--model', type=str, help='model name.')
    return parser


MODELS = {  # Model, Discr, use_batch_discr
    'molgan': (GraphGANModel, encoder_rgcn, False),
    'pacgan': (PacGANModel, encoder_rgcn, False),
    'pacstats': (PacStatsGANModel, encoder_rgcn, False),
    'pacxstats': (PacXStatsGANModel, encoder_rgcn, False),
    'molgan+batch': (GraphGANModel, encoder_rgcn, True),
    'gat': (GraphGANModel, encoder_gat, False),
    'flat_gat': (GraphGANModel, encoder_flat_gat, False)
}


if __name__ == '__main__':
    parser = Argparser()
    args = parser.parse_args()

    Model, Discr, batch_discr = MODELS[args.model]

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
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    # trainer
    trainer = Trainer(model, optimizer, session)

    print('Parameters: {}'.format(np.sum([np.prod(e.shape) for e in session.run(tf.trainable_variables())])))

    trainer.train(batch_dim=batch_dim, epochs=epochs, steps=steps,
                  train_fetch_dict=train_fetch_dict, train_feed_dict=train_feed_dict,
                  eval_fetch_dict=eval_fetch_dict, eval_feed_dict=eval_feed_dict,
                  test_fetch_dict=test_fetch_dict, test_feed_dict=test_feed_dict,
                  _eval_update=_eval_update, _test_update=_test_update,
                  save_every=save_every, directory=args.name)
