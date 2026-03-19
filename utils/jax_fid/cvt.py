from .utils import download

import pickle
from functools import partial
from collections import defaultdict
import jax
from flax import core

class ddd(dict):
    def __getitem__(self, key):
        if key not in self:
            self[key] = ddd()
        return super().__getitem__(key)

def load_bbasiconv2d(op, ob, p):
    op['BatchNorm_0']['bias'] = p['bn']['bias']
    op['BatchNorm_0']['scale'] = p['bn']['scale']
    op['Conv_0']['kernel'] = p['conv']['kernel']
    # op['Conv_0']['bias'] = p['conv']['bias']
    
    ob['BatchNorm_0']['mean'] = p['bn']['mean']
    ob['BatchNorm_0']['var'] = p['bn']['var']
    
def load_inceptionA(op, ob, p):
    load_bbasiconv2d(op['BasicConv2d_0'], ob['BasicConv2d_0'], p['branch1x1'])
    load_bbasiconv2d(op['BasicConv2d_1'], ob['BasicConv2d_1'], p['branch5x5_1'])
    load_bbasiconv2d(op['BasicConv2d_2'], ob['BasicConv2d_2'], p['branch5x5_2'])
    load_bbasiconv2d(op['BasicConv2d_3'], ob['BasicConv2d_3'], p['branch3x3dbl_1'])
    load_bbasiconv2d(op['BasicConv2d_4'], ob['BasicConv2d_4'], p['branch3x3dbl_2'])
    load_bbasiconv2d(op['BasicConv2d_5'], ob['BasicConv2d_5'], p['branch3x3dbl_3'])
    load_bbasiconv2d(op['BasicConv2d_6'], ob['BasicConv2d_6'], p['branch_pool'])
    
def load_inceptionB(op, ob, p):
    load_bbasiconv2d(op['BasicConv2d_0'], ob['BasicConv2d_0'], p['branch3x3'])
    load_bbasiconv2d(op['BasicConv2d_1'], ob['BasicConv2d_1'], p['branch3x3dbl_1'])
    load_bbasiconv2d(op['BasicConv2d_2'], ob['BasicConv2d_2'], p['branch3x3dbl_2'])
    load_bbasiconv2d(op['BasicConv2d_3'], ob['BasicConv2d_3'], p['branch3x3dbl_3'])
    
def load_inceptionC(op, ob, p):
    load_bbasiconv2d(op['BasicConv2d_0'], ob['BasicConv2d_0'], p['branch1x1'])
    load_bbasiconv2d(op['BasicConv2d_1'], ob['BasicConv2d_1'], p['branch7x7_1'])
    load_bbasiconv2d(op['BasicConv2d_2'], ob['BasicConv2d_2'], p['branch7x7_2'])
    load_bbasiconv2d(op['BasicConv2d_3'], ob['BasicConv2d_3'], p['branch7x7_3'])
    load_bbasiconv2d(op['BasicConv2d_4'], ob['BasicConv2d_4'], p['branch7x7dbl_1'])
    load_bbasiconv2d(op['BasicConv2d_5'], ob['BasicConv2d_5'], p['branch7x7dbl_2'])
    load_bbasiconv2d(op['BasicConv2d_6'], ob['BasicConv2d_6'], p['branch7x7dbl_3'])
    load_bbasiconv2d(op['BasicConv2d_7'], ob['BasicConv2d_7'], p['branch7x7dbl_4'])
    load_bbasiconv2d(op['BasicConv2d_8'], ob['BasicConv2d_8'], p['branch7x7dbl_5'])
    load_bbasiconv2d(op['BasicConv2d_9'], ob['BasicConv2d_9'], p['branch_pool'])

def load_inceptionD(op, ob, p):
    load_bbasiconv2d(op['BasicConv2d_0'], ob['BasicConv2d_0'], p['branch3x3_1'])
    load_bbasiconv2d(op['BasicConv2d_1'], ob['BasicConv2d_1'], p['branch3x3_2'])
    load_bbasiconv2d(op['BasicConv2d_2'], ob['BasicConv2d_2'], p['branch7x7x3_1'])
    load_bbasiconv2d(op['BasicConv2d_3'], ob['BasicConv2d_3'], p['branch7x7x3_2'])
    load_bbasiconv2d(op['BasicConv2d_4'], ob['BasicConv2d_4'], p['branch7x7x3_3'])
    load_bbasiconv2d(op['BasicConv2d_5'], ob['BasicConv2d_5'], p['branch7x7x3_4'])
    
def load_inceptionE(op, ob, p):
    load_bbasiconv2d(op['BasicConv2d_0'], ob['BasicConv2d_0'], p['branch1x1'])
    load_bbasiconv2d(op['BasicConv2d_1'], ob['BasicConv2d_1'], p['branch3x3_1'])
    load_bbasiconv2d(op['BasicConv2d_2'], ob['BasicConv2d_2'], p['branch3x3_2a'])
    load_bbasiconv2d(op['BasicConv2d_3'], ob['BasicConv2d_3'], p['branch3x3_2b'])
    load_bbasiconv2d(op['BasicConv2d_4'], ob['BasicConv2d_4'], p['branch3x3dbl_1'])
    load_bbasiconv2d(op['BasicConv2d_5'], ob['BasicConv2d_5'], p['branch3x3dbl_2'])
    load_bbasiconv2d(op['BasicConv2d_6'], ob['BasicConv2d_6'], p['branch3x3dbl_3a'])
    load_bbasiconv2d(op['BasicConv2d_7'], ob['BasicConv2d_7'], p['branch3x3dbl_3b'])
    load_bbasiconv2d(op['BasicConv2d_8'], ob['BasicConv2d_8'], p['branch_pool'])
    
def load_all():
    params = download('https://dl.dropboxusercontent.com/s/xt6zvlvt22dcwck/inception_v3_weights_fid.pickle', 'ce58f6044b0bf244c4e3185158b7fece')
    out = {'params': ddd(), 'batch_stats': ddd()}
    
    load_bbasiconv2d(out['params']['BasicConv2d_0'], out['batch_stats']['BasicConv2d_0'], params['Conv2d_1a_3x3'])
    load_bbasiconv2d(out['params']['BasicConv2d_1'], out['batch_stats']['BasicConv2d_1'], params['Conv2d_2a_3x3'])
    load_bbasiconv2d(out['params']['BasicConv2d_2'], out['batch_stats']['BasicConv2d_2'], params['Conv2d_2b_3x3'])
    load_bbasiconv2d(out['params']['BasicConv2d_3'], out['batch_stats']['BasicConv2d_3'], params['Conv2d_3b_1x1'])
    load_bbasiconv2d(out['params']['BasicConv2d_4'], out['batch_stats']['BasicConv2d_4'], params['Conv2d_4a_3x3'])

    load_inceptionA(out['params']['InceptionA_0'], out['batch_stats']['InceptionA_0'], params['Mixed_5b'])
    load_inceptionA(out['params']['InceptionA_1'], out['batch_stats']['InceptionA_1'], params['Mixed_5c'])
    load_inceptionA(out['params']['InceptionA_2'], out['batch_stats']['InceptionA_2'], params['Mixed_5d'])

    load_inceptionB(out['params']['InceptionB_0'], out['batch_stats']['InceptionB_0'], params['Mixed_6a'])

    load_inceptionC(out['params']['InceptionC_0'], out['batch_stats']['InceptionC_0'], params['Mixed_6b'])
    load_inceptionC(out['params']['InceptionC_1'], out['batch_stats']['InceptionC_1'], params['Mixed_6c'])
    load_inceptionC(out['params']['InceptionC_2'], out['batch_stats']['InceptionC_2'], params['Mixed_6d'])
    load_inceptionC(out['params']['InceptionC_3'], out['batch_stats']['InceptionC_3'], params['Mixed_6e'])

    load_inceptionD(out['params']['InceptionD_0'], out['batch_stats']['InceptionD_0'], params['Mixed_7a'])

    load_inceptionE(out['params']['InceptionE_0'], out['batch_stats']['InceptionE_0'], params['Mixed_7b'])
    load_inceptionE(out['params']['InceptionE_1'], out['batch_stats']['InceptionE_1'], params['Mixed_7c'])
    
    out['params']['Dense_0']['Dense_0']['kernel'] = params['fc']['kernel']
    out['params']['Dense_0']['Dense_0']['bias'] = params['fc']['bias']
    
    # to dict
    out = core.freeze(out)

    n_treeleaves_expected = 472
    n_params_expected = 23885392
    n_treeleaves_got = sum(1 for _ in jax.tree_util.tree_leaves(out))
    n_params_got = sum(p.size for p in jax.tree_util.tree_leaves(out))
    assert n_treeleaves_got == n_treeleaves_expected, f'Expected {n_treeleaves_expected} tree leaves, got {n_treeleaves_got}'
    assert n_params_got == n_params_expected, f'Expected {n_params_expected} params, got {n_params_got}'
    return out

if __name__ == '__main__':
    oo = load_all()
    print('Inception v3 params for FID loaded successfully.')

    td = jax.tree_structure(
        core.freeze(pickle.load(open('debug.pkl', 'rb')))
    )
    assert jax.tree_structure(oo) == td, (jax.tree_structure(oo), td)
    print('test passed!')