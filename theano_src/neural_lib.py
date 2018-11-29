import os
import theano.tensor as T
import theano
import time
from theano import config
import numpy as np
import collections
#theano.config.compute_test_value = 'off'
#theano.config.profile=True
#theano.config.profile_memory=True


''' This is the major file that defines neural classes (individual components in neural architectures) and several helper functions to facilitate configuring the neural classes.
    The basic classes include: 
        Embedding
        Entity_attention
        (Bi)LSTM
        (Bi)GraphLSTM (and several variants)
        TargetHidden 
        LogitRegression'''


np.random.seed(1)
def name_tv(*params):
    """
    Helper function to generate names
    Join the params as string using '_'
    and also add a unique id, since every node in a theano
    graph should have a unique name
    """
    if not hasattr(name_tv, "uid"):
        name_tv.uid = 0
    name_tv.uid += 1
    tmp = "_".join(params)
    return "_".join(['tparam', tmp, str(name_tv.uid)])


def np_floatX(data):
     return np.asarray(data, dtype=config.floatX)

def tparams_make_name(*params):
    tmp = make_name(*params)
    return "_".join(['tparam', tmp])

def make_name(*params):
    """
    Join the params as string using '_'
    and also add a unique id, since every node in a theano
    graph should have a unique name
    """
    return "_".join(params)

def reverse(tensor):
    rev, _ = theano.scan(lambda itm: itm,
            sequences=tensor,
            go_backwards=True,
            strict=True,
            name='reverse_rand%d'%np.random.randint(1000))
    return rev


def read_matrix_from_file(fn, dic):
    '''
    Assume that the file contains words in first column,
    and embeddings in the rest and that dic maps words to indices.
    '''
    _data = open(fn).read().strip().split('\n')
    _data = [e.strip().split() for e in _data]
    dim = len(_data[0]) - 1
    data = {}
    # NOTE: The norm of onesided_uniform rv is sqrt(n)/sqrt(3)
    # Since the expected value of X^2 = 1/3 where X ~ U[0, 1]
    # => sum(X_i^2) = dim/3
    # => norm       = sqrt(dim/3)
    # => norm/dim   = sqrt(1/3dim)
    multiplier = np.sqrt(1.0/(3*dim))
    for e in _data:
        r = np.array([float(_e) for _e in e[1:]])
        data[e[0]] = (r/np.linalg.norm(r)) * multiplier
    M = ArrayInit(ArrayInit.onesided_uniform, multiplier=1.0/dim).initialize(len(data), dim)
    for word, idx in dic.iteritems():
        if word in data:
            M[idx] = data[word]
    return M

''' Dropout. Can be used in different places.'''
def _dropout_from_layer(rng, layer, p):
    """p is the probablity of dropping a unit
    """
    srng = theano.tensor.shared_randomstreams.RandomStreams(
            rng.randint(999999))
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask = srng.binomial(n=1, p=1-p, size=layer.shape)
    # The cast is important because
    output = layer * T.cast(mask, theano.config.floatX)
    return output

''' The class for initializing parameters matrixs.'''
class ArrayInit(object):
    normal = 'normal'
    onesided_uniform = 'onesided_uniform'
    twosided_uniform = 'twosided_uniform'
    ortho = 'ortho'
    zero = 'zero'
    unit = 'unit'
    ones = 'ones'
    fromfile = 'fromfile'
    def __init__(self, option,
            multiplier=0.01,
            matrix=None,
            word2idx=None):
        self.option = option
        self.multiplier = multiplier
        self.matrix_filename = None
        self.matrix = self._matrix_reader(matrix, word2idx)
        if self.matrix is not None:
            self.multiplier = 1
        return

    def _matrix_reader(self, matrix, word2idx):
        if type(matrix) is str:
            self.matrix_filename = matrix
            assert os.path.exists(matrix), "File %s not found"%matrix
            matrix = read_matrix_from_file(matrix, word2idx)
            return matrix
        else:
            return None

    def initialize(self, *xy, **kwargs):
        if self.option == ArrayInit.normal:
            M = np.random.randn(*xy)
        elif self.option == ArrayInit.onesided_uniform:
            M = np.random.rand(*xy)
        elif self.option == ArrayInit.twosided_uniform:
            M = np.random.uniform(-1.0, 1.0, xy)
        elif self.option == ArrayInit.ortho:
            f = lambda dim: np.linalg.svd(np.random.randn(dim, dim))[0]
            if int(xy[1]/xy[0]) < 1 and xy[1]%xy[0] != 0:
                raise ValueError(str(xy))
            M = np.concatenate(tuple(f(xy[0]) for _ in range(int(xy[1]/xy[0]))),
                    axis=1)
            assert M.shape == xy
        elif self.option == ArrayInit.zero:
            M = np.zeros(xy)
        elif self.option in [ArrayInit.unit, ArrayInit.ones]:
            M = np.ones(xy)
        elif self.option == ArrayInit.fromfile:
            assert isinstance(self.matrix, np.ndarray)
            M = self.matrix
        else:
            raise NotImplementedError
        #self.multiplier = (kwargs['multiplier']
        multiplier = (kwargs['multiplier']
            if ('multiplier' in kwargs
                    and kwargs['multiplier'] is not None)
                else self.multiplier)
        #return (M*self.multiplier).astype(config.floatX)
        return (M*multiplier).astype(config.floatX)

    def __repr__(self):
        mults = ', multiplier=%s'%((('%.3f'%self.multiplier)
            if type(self.multiplier) is float
            else str(self.multiplier)))
        mats = ((', matrix="%s"'%self.matrix_filename)
                if self.matrix_filename is not None
                else '')
        return "ArrayInit(ArrayInit.%s%s%s)"%(self.option, mults, mats)


class SerializableLambda(object):
    def __init__(self, s):
        self.s = s
        self.f = eval(s)
        return

    def __repr__(self):
        return "SerializableLambda('%s')"%self.s

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)


class StackConfig(collections.MutableMapping):
    """A dictionary like object that would automatically recognize
    keys that end with the following pattern and return appropriate keys.
    _out_dim  :
    _initializer :
    The actions to take are stored in a list for easy composition.
    """
    actions = [
            (lambda key: key.endswith('_out_dim')       , lambda x: x),
            (lambda key: key.endswith('_T_initializer') , ArrayInit(ArrayInit.onesided_uniform)),
            (lambda key: key.endswith('_U_initializer') , ArrayInit(ArrayInit.ortho, multiplier=1)),
            (lambda key: key.endswith('_W_initializer') , ArrayInit(ArrayInit.twosided_uniform, multiplier=1)),
            (lambda key: key.endswith('_N_initializer') , ArrayInit(ArrayInit.normal)),
            (lambda key: key.endswith('_b_initializer') , ArrayInit(ArrayInit.zero)),
            (lambda key: key.endswith('_p_initializer') , ArrayInit(ArrayInit.twosided_uniform, multiplier=1)),
            (lambda key: key.endswith('_c_initializer') , ArrayInit(ArrayInit.twosided_uniform, multiplier=1)),
            (lambda key: key.endswith('_reg_weight')    , 0),
            (lambda key: key.endswith('_viterbi')     , False),
            (lambda key: key.endswith('_begin')         , 1),
            (lambda key: key.endswith('_end')           , -1),
            #(lambda key: key.endswith('_activation_fn') , lambda x: x + theano.tensor.abs_(x)),
            #(lambda key: key.endswith('_v_initializer') , ArrayInit(ArrayInit.ones, multiplier=NotImplemented)),
            ]
    def __init__(self, dictionary):
        self.store = collections.OrderedDict()
        self.store.update(dictionary)

    def __getitem__(self, key):
        if key in self.store:
            return self.store[key]
        for (predicate, retval) in self.actions:
            if predicate(key):
                return retval
        raise KeyError(key)

    def __setitem__(self, key, value):
        self.store[key] = value

    def __delitem__(self, key):
        del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def reset(self):
        for k in self.store:
            if k.startswith('tparam_'):
                del self.store[k]
        return


class Chip(object):
    """ The abstract class for neural chips.
    A Chip object requires name and a param dictionary
    that contains param[name+'_'+out_dim] (This can be a function that depends on the input_dim)
    Other than that it must also contain appropriate initializers for all the parameters.

    The params dictionary is updated to contain 'tparam_<param name>_uid'
    """
    def __init__(self, name, params=None):
        """ I set the output dimension of every node in the parameters.
            The input dimension would be set when prepend is called.
            (Since prepend method receives the previous chip)
        """
        self.name = name
        if params is not None:
            print 'current chip:', name, 'out dimension:', self.kn('out_dim')
            self.out_dim = params[self.kn('out_dim')]
            print 'init chip:', self.name, 'out dim:', self.out_dim
            self.params = params
        return

    def prepend(self, previous_chip):
        """ Note that my input_dim of self = output_dim of previous_chip
        Also we keep track of absolute_input (the first input) to the layer
        """
        #if hasattr(previous_chip, 'kn') and previous_chip.kn('win') in previous_chip.params:
        #    print 'window size:', previous_chip.params[previous_chip.kn('win')]
        self.in_dim = previous_chip.out_dim * previous_chip.params[previous_chip.kn('win')] if (hasattr(previous_chip, 'kn') and previous_chip.kn('win') in previous_chip.params) else previous_chip.out_dim
        #print 'previous_chip out_dim:', previous_chip.out_dim, 'previous_chip window:', previous_chip.params[previous_chip.kn('win')]
        if hasattr(self.out_dim, '__call__'):
            self.out_dim = self.out_dim(self.in_dim)
        print 'in prepend, chip', self.name, 'in dim =', self.in_dim, 'out dim =', self.out_dim
        self.parameters = []
        return self

    def compute(self, input_tv):
        """ Note that input_tv = previous_chip.output_tv
        This method returns a dictionary of internal weight params
        and This method sets self.output_tv
        """
        raise NotImplementedError

    def regularizable_variables(self):
        """ If a value stored in the dictionary has the attribute
        is_regularizable then that value is regularizable
        """
        return [k for k in self.params
                if hasattr(self.params[k], 'is_regularizable')
                and self.params[k].is_regularizable]

    def kn(self, thing):
        if len(thing) == 1: # It is probably ['U', 'W', 'b', 'T', 'N'] or some such Matrix
            keyname_suffix = '_initializer'
        else:
            keyname_suffix = ''
        return self.name + '_' + thing + keyname_suffix

    def _declare_mat(self, name, *dim, **kwargs):
        multiplier = (kwargs['multiplier']
                if 'multiplier' in kwargs
                else None)
        var = theano.shared(
                self.params[self.kn(name)].initialize(*dim, multiplier=multiplier),
                name=tparams_make_name(self.name, name)
                )
        if 'is_regularizable' not in kwargs:
            var.is_regularizable = True # Default
        else:
            var.is_regularizable = kwargs['is_regularizable']
        return var

    def needed_key(self):
        return self._needed_key_impl()

    def _needed_key_impl(self, *things):
        return [self.kn(e) for e in ['out_dim'] + list(things)]

class Start(object):
    """ A start object which has all the necessary attributes that
    any chip object that would call it would need.
    """
    def __init__(self, out_dim, output_tv):
        self.out_dim = out_dim
        self.output_tv = output_tv

# Note: should make changes here to pass pre_defined embeddings as parameters.
class Embedding(Chip):
    def prepend(self, previous_chip):
        self = super(Embedding, self).prepend(previous_chip)
        if 'emb_matrix' in self.params:
            print 'pre_trained embedding!!'
            self.T_ = self.params['emb_matrix']
            print self.T_, type(self.T_)
        else:
            self.T_ = self._declare_mat('T', self.params['voc_size'], self.out_dim)
            self.params['emb_dim'] = self.out_dim
        self.parameters = [self.T_]
        return self

    """ An embedding converts  one-hot-vectors to dense vectors.
    We never take dot products with one-hot-vectors.
    This requires a T_initializer
    """
    def compute(self, input_tv):
        print input_tv, type(input_tv)
        n_timesteps = input_tv.shape[0]
        window_size = 1
        if input_tv.ndim == 2:
            window_size = input_tv.shape[1] 
        elif input_tv.ndim == 3:
            batch_size = input_tv.shape[1]
            window_size = input_tv.shape[2]
        print 'input_tv dimension:', input_tv.ndim
        print 'window size = ', window_size
        self.params[self.kn('win')] = window_size
        if input_tv.ndim < 3:
            self.output_tv = self.T_[input_tv.flatten()].reshape([n_timesteps, window_size * self.out_dim], ndim=2)
        else:
            self.output_tv = self.T_[input_tv.flatten()].reshape([n_timesteps, batch_size, window_size * self.out_dim], ndim=3)
        if self.params.get(self.kn('dropout_rate'), 0.0) != 0.0:
            print 'DROP OUT!!! at circuite', self.name, 'Drop out rate: ', self.params[self.kn('dropout_rate')]
            self.output_tv = _dropout_from_layer(self.params['rng'], self.output_tv, self.params[self.kn('dropout_rate')])
        # Note: when we import the pre-defined emb_matrix, we do not add it to internal parameters because we already defined it as a regularizable parameter.
        #if 'emb_matrix' in self.params:
        #    return tuple() 
        #else:

    def needed_key(self):
        return self._needed_key_impl('T')


# compute attention according to the similarity between the token with the entities
class Entity_attention(Chip):
    ''' input_tv shape: (sent_len, batch_size, tv_dim)'''
    def get_att_weights(self, input_tv, i, entity_idxs):
        if input_tv.ndim == 3:
            ''' entities_tv shape: (batch_size, tv_dim)'''
            entity_tv = T.sum(input_tv * entity_idxs[:, :, None], axis=0)
            self.entity_tvs = T.set_subtensor(self.entity_tvs[i], entity_tv)
            ''' attention_weights shape: (sent_len, batch_size)'''
            #input_tv = input_tv.dimshuffle(1,0,2)
            #attention_weights = T.nnet.softmax(T.batched_dot(input_tv/input_tv.norm(2, axis=2, keepdims=True), entity_tv/entity_tv.norm(2, axis=1, keepdims=True))).T
            attention_weights = T.nnet.softmax(T.batched_dot(input_tv.dimshuffle(1,0,2), entity_tv)).T
            #attention_score = T.exp(T.sum(input_tv * entity_tv[None, :, :], axis=2))
            #attention_weights = attention_score / attention_score.sum(axis=0) 
            return attention_weights
        else:
            entity_tv = T.sum(input_tv * entity_idxs.dimshuffle(0, 'x'), axis=0)
            attention_weights = T.nnet.softmax(T.dot(input_tv, entity_tv))
            return attention_weights

    def compute(self, input_tv, entities_tv):
        print 'in Entity_attention layer, input dimension:', input_tv.ndim
        if input_tv.ndim == 3:
            self.attention_weights = T.zeros_like(input_tv[:, :, 0])
            self.att_wt_arry = T.zeros_like(input_tv[:,:,:3]).dimshuffle(2,0,1)
            self.entity_tvs = T.zeros_like(input_tv[:3, :, :])
        else:
            self.attention_weights = T.zeros_like(input_tv[:, 0]) 
            self.att_wt_arry = T.zeros_like(input_tv[:,:3]).T
        for i, enidx in enumerate(entities_tv):
            temp_weight = self.get_att_weights(input_tv, i, enidx)
            self.att_wt_arry = T.set_subtensor(self.att_wt_arry[i], temp_weight)
            self.attention_weights += temp_weight
        self.attention_weights = self.attention_weights / len(entities_tv)
        print 'attention weight dimensions:', self.attention_weights.ndim
        if input_tv.ndim == 3:
            self.output_tv = input_tv * self.attention_weights[:, :, None]
        else:
            self.output_tv = input_tv * self.attention_weights[:, None]


class Activation(Chip):
    """ This requires a (activation)_fn parameter
    """
    def compute(self, input_tv):
        self.output_tv = self.params[self.kn('fn')](input_tv)

    def needed_key(self):
        return self._needed_key_impl('fn')

class TargetHidden(Chip):
    def prepend(self, previous_chip, entity_num):
        self.previous_chip = previous_chip
        super(TargetHidden, self).prepend(previous_chip)
        self.out_dim = entity_num*self.in_dim
        return self

    ''' If input_tv.ndim == 2: shape == (maxlen, in_dim),
           output_tv.shape == (num_entity * in_dim)
        If input_tv.ndim == 3: shape == (maxlen, n_samples, in_dim)
           output_tv.shape == (n_samples, num_entity * in_dim)
    '''
    def compute(self, input_tv, entities_tv):
        if input_tv.ndim == 3:
            self.output_tv = T.sum(input_tv * entities_tv[0][:, :, None], axis=0)
            for enidx in entities_tv[1:]:
                self.output_tv= T.concatenate([self.output_tv, T.sum(input_tv * enidx[:, :, None], axis=0)], axis=1)

        else:
            self.output_tv = T.sum(input_tv * entities_tv[0].dimshuffle(0, 'x'), axis=0)
            #print 'in TargetHidden, variable types:', input_tv.dtype, entities_tv[0].dtype
            for enidx in entities_tv[1:]:
                self.output_tv = T.concatenate([self.output_tv, T.sum(input_tv * enidx.dimshuffle(0, 'x'), axis=0)])
    

class LogitRegression(Chip):
    def prepend(self, previous_chip):
        self.previous_chip = previous_chip
        super(LogitRegression, self).prepend(previous_chip)
        self.W = self._declare_mat('W', self.in_dim, self.out_dim)
        self.b = self._declare_mat('b', self.out_dim) 
        self.parameters = [self.W, self.b]
        return self

    def compute(self, input_tv):
        if input_tv.ndim == 3:
            self.output_tv = T.nnet.softmax(T.dot(input_tv.max(axis=0), self.W) + self.b)
            self.gold_y = T.ivector(make_name(self.name, 'gold_y')).astype('int32')
            self.score = -T.mean(T.log(self.output_tv)[T.arange(self.gold_y.shape[0]), self.gold_y]) 
        elif input_tv.ndim == 2:
            self.output_tv = T.nnet.softmax(T.dot(input_tv, self.W) + self.b)
            self.gold_y = T.ivector(make_name(self.name, 'gold_y')).astype('int32')
            self.score = -T.mean(T.log(self.output_tv)[T.arange(self.gold_y.shape[0]), self.gold_y]) 
        else:
            assert input_tv.ndim == 1
            self.output_tv = T.nnet.softmax(T.dot(input_tv, self.W) + self.b).dimshuffle(1,)
            #print 'in LogitRegression, variable types:', input_tv.dtype, self.output_tv.dtype
            self.gold_y = T.iscalar(make_name(self.name, 'gold_y')).astype('int32')
            self.score = -T.log(self.output_tv[self.gold_y])


    def needed_key(self):
        return self._needed_key_impl('W', 'b')


class Linear(Chip):
    def prepend(self, previous_chip):
        self = super(Linear, self).prepend(previous_chip)
        self.N = self._declare_mat('N', self.in_dim, self.out_dim)
        self.parameters = [self.N]
        return self

    """ A Linear Chip is a matrix Multiplication
    It requires a U_initializer
    """
    def compute(self, input_tv):
        self.output_tv = T.dot(input_tv, self.N)

    def needed_key(self):
        return self._needed_key_impl('N')

class Bias(Chip):
    def prepend(self, previous_chip):
        self = super(Bias, self).prepend(previous_chip)
        self.b = self._declare_mat('b', self.out_dim) #, is_regularizable=False)
        self.parameters = [self.b]
        return self

    """ A Bias Chip adds a vector to the input
    It requires a b_initializer
    """
    def compute(self, input_tv):
        self.output_tv = input_tv + self.b

    def needed_key(self):
        return self._needed_key_impl('b')

class BiasedLinear(Chip):
    def __init__(self, name, params=None):
        super(BiasedLinear, self).__init__(name, params)
        self.params[self.name+"_linear_out_dim"] = params[self.kn('out_dim')]
        self.params[self.name+"_bias_out_dim"] = params[self.kn('out_dim')]
        self.Linear = Linear(name+'_linear', self.params)
        self.Bias = Bias(name+'_bias', self.params)

    def prepend(self, previous_chip):
        self.Bias.prepend(self.Linear.prepend(previous_chip))
        self.in_dim = self.Linear.in_dim
        self.parameters = self.Linear.parameters + self.Bias.parameters
        return self
    """ Composition of Linear and Bias
    It requires a U_initializer and a b_initializer
    """
    def compute(self, input_tv):
        self.Linear.compute(input_tv)
        self.Bias.compute(self.Linear.output_tv)
        self.output_tv = self.Bias.output_tv

    def needed_key(self):
        return self.Linear.needed_key() + self.Bias.needed_key()


class Convolutional_NN(Chip):
    def prepend(self, previous_chip):
        self = super(Convolutional_NN, self).prepend(previous_chip)
        self.W = self._declare_mat('W', self.in_dim, self.out_dim)
        self.b = self._declare_mat('b', self.out_dim) #, is_regularizable = False)
        self.parameters = [self.W, self.b]
        return self

    """ This requires W, U and b initializer
    """
    def compute(self, input_tv):
        self.output_tv = T.dot(input_tv, self.W).astype(config.floatX) + self.b

    def needed_key(self):
        return self._needed_key_impl('W', 'b')


class LSTM(Chip):
    def prepend(self, previous_chip):
        self = super(LSTM, self).prepend(previous_chip)
        print 'lstm in dim:', self.in_dim, 'out dim:', self.out_dim
        self.go_backwards = self.params[self.kn('go_backwards')]
        self.W = self._declare_mat('W', self.in_dim, 4*self.out_dim)
        self.U = self._declare_mat('U', self.out_dim, 4*self.out_dim)
        self.b = self._declare_mat('b', 4*self.out_dim) #, is_regularizable = False)
        #self.p = self._declare_mat('p', 3*self.out_dim)
        self.parameters = [self.W, self.U, self.b] #, self.p]
        return self

    """ This requires W, U and b initializer
    """
    def compute(self, input_tv, mask=None):
        n_steps = input_tv.shape[0]
        if input_tv.ndim == 3:
            n_samples = input_tv.shape[1]
        else:
            n_samples = 1

        def __slice(matrix, row_idx, stride):
            if matrix.ndim == 3:
                return matrix[:, :, row_idx * stride:(row_idx + 1) * stride]
            elif matrix.ndim == 2:
                return matrix[:, row_idx * stride:(row_idx + 1) * stride]
            else:
                return matrix[row_idx*stride: (row_idx+1)*stride]

        def __step(x_, h_prev, c_prev, U): #, p):
            """
            x = Transformed and Bias incremented Input  (This is basically a matrix)
                We do the precomputation for efficiency.
            h_prev = previous output of the LSTM (Left output of this function)
            c_prev = previous cell value of the LSTM (Right output of this function)

            This is the vanilla version of the LSTM without peephole connections
            See: Section 2, "LSTM: A Search Space Odyssey", Klaus et. al, ArXiv(2015)
            http://arxiv.org/pdf/1503.04069v1.pdf for details.
            """
            preact = T.dot(h_prev, U) + x_
            i = T.nnet.sigmoid(__slice(preact, 0, self.out_dim) )#+ __slice(p, 0, self.out_dim)*c_prev) # Input gate
            f = T.nnet.sigmoid(__slice(preact, 1, self.out_dim) )#+ __slice(p, 1, self.out_dim)*c_prev) # Forget gate
            z = T.tanh(__slice(preact, 3, self.out_dim)) # block input
            c = f * c_prev + i * z # cell state
            o = T.nnet.sigmoid(__slice(preact, 2, self.out_dim) )#+ __slice(p, 2, self.out_dim) * c) # output gate
            h = o * T.tanh(c)  # block output
            return h, c

        def __step_batch(x_, m_, h_prev, c_prev, U): #, p):
            # Shape: (4*out_dim, num_sample) + (num_sample, 4*out_dim).T
            preact = T.dot(h_prev, U) + x_
            i = T.nnet.sigmoid(__slice(preact, 0, self.out_dim) )#+ __slice(p, 0, self.out_dim)*c_prev) # Input gate
            f = T.nnet.sigmoid(__slice(preact, 1, self.out_dim) )#+ __slice(p, 1, self.out_dim)*c_prev) # Forget gate
            z = T.tanh(__slice(preact, 3, self.out_dim)) # block input
            c = f * c_prev + i * z # cell state
            o = T.nnet.sigmoid(__slice(preact, 2, self.out_dim) )#+ __slice(p, 2, self.out_dim) * c) # output gate
            h = o * T.tanh(c)  # block output
            c = m_[:, None] * c + (1. - m_)[:, None] * c_prev
            h = m_[:, None] * h + (1. - m_)[:, None] * h_prev
            return h, c

        x_in = T.dot(input_tv, self.W).astype(config.floatX) + self.b
        seq_in = [x_in]
        lstm_step = __step
        h_init = T.alloc(np_floatX(0.), self.out_dim)
        c_init = T.alloc(np_floatX(0.), self.out_dim)
        if mask is not None:
            seq_in.append(mask)
            lstm_step = __step_batch
            h_init = T.alloc(np_floatX(0.), n_samples, self.out_dim)
            c_init = T.alloc(np_floatX(0.), n_samples, self.out_dim)
        print 'lstm step:', lstm_step
        rval, _ = theano.scan(lstm_step,
                              sequences=seq_in,
                              outputs_info=[h_init, c_init],
                              non_sequences=[self.U], #, self.p],
                              go_backwards=self.go_backwards,
                              name=name_tv(self.name, 'LSTM_layer'),
                              n_steps=n_steps,
                              strict=True,
        )
        self.output_tv = reverse(rval[0]) if self.go_backwards else rval[0]
        if self.params.get(self.kn('dropout_rate'), 0.0) != 0.0:
            print 'DROP OUT!!! at circuite', self.name, 'Drop out rate: ', self.params[self.kn('dropout_rate')]
            self.output_tv = _dropout_from_layer(self.params['rng'], self.output_tv, self.params[self.kn('dropout_rate')])

    def needed_key(self):
        return self._needed_key_impl('W', 'U', 'b', 'go_backwards')


class GraphLSTM(Chip):
    def prepend(self, previous_chip):
        self = super(GraphLSTM, self).prepend(previous_chip)
        print 'graph lstm in dim:', self.in_dim, 'out dim:', self.out_dim
        self.go_backwards = self.params[self.kn('go_backwards')]
        self.W = self._declare_mat('W', self.in_dim, 4*self.out_dim)
        self.U = self._declare_mat('U', self.out_dim, 4*self.out_dim)
        self.b = self._declare_mat('b', 4*self.out_dim) #, is_regularizable = False)
        #self.p = self._declare_mat('p', 3*self.out_dim)
        self.parameters = [self.W, self.U, self.b] #, self.p)
        return self

    # Shapes: x_= (4*out_dim,), child_h, child_c = (out_dim, sent_len), child_exists = (sent_len)
    def recursive_unit(self, x_, child_h, child_c, child_exists, U):
        h_tilde = T.sum(child_h, axis=-1) / child_exists.sum() #T.cast(, theano.config.floatX) 
        preact = T.dot(h_tilde, U) + x_
        i = T.nnet.sigmoid(self.slice(preact, 0, self.out_dim) ) #+ self.slice(self.p, 0, self.out_dim)*c_prev) # Input gate
        #f = T.nnet.sigmoid(self.slice(preact, 1, self.out_dim) )#+ self.slice(self.p, 1, self.out_dim)*c_prev) # Forget gate
        f = T.nnet.sigmoid(self.slice(x_, 1, self.out_dim).dimshuffle('x', 0) + T.dot(child_h.T, self.slice(U, 1, self.out_dim)) )
        z = T.tanh(self.slice(preact, 3, self.out_dim)) # block input
        c = (f.T * child_c).sum(axis=-1) / child_exists.sum() + i * z # cell state
        o = T.nnet.sigmoid(self.slice(preact, 2, self.out_dim) )#+ self.slice(self.p, 2, self.out_dim) * c) # output gate
        h = o * T.tanh(c)  # block output
        return h, c


    # Shapes: x_= (batch_size, 4*out_dim,), child_h, child_c = (batch_size, sent_len, out_dim), 
    # child_exists = (batch_size, sent_len)
    def recursive_unit_batch(self, x_, child_h, child_c, child_exists, U):
        h_tilde = T.sum(child_h, axis=1) / child_exists[:, None] #T.cast(, theano.config.floatX) 
        preact = T.dot(h_tilde, U) + x_
        i = T.nnet.sigmoid(self.slice(preact, 0, self.out_dim) ) #+ self.slice(self.p, 0, self.out_dim)*c_prev) # Input gate
        #f = T.nnet.sigmoid(self.slice(preact, 1, self.out_dim) )#+ self.slice(self.p, 1, self.out_dim)*c_prev) # Forget gate
        f = T.nnet.sigmoid(self.slice(x_, 1, self.out_dim)[:, None, :] + T.dot(child_h, self.slice(U, 1, self.out_dim)) ) #/ child_exists[:, :, None] 
        z = T.tanh(self.slice(preact, 3, self.out_dim)) # block input
        c = (f * child_c).sum(axis=1) / child_exists[:, None] + i * z # cell state
        o = T.nnet.sigmoid(self.slice(preact, 2, self.out_dim) )#+ self.slice(self.p, 2, self.out_dim) * c) # output gate
        h = o * T.tanh(c)  # block output
        return h, c

    def slice(self, matrix, row_idx, stride):
        if matrix.ndim == 3:
            return matrix[:, :, row_idx * stride:(row_idx + 1) * stride]
        elif matrix.ndim == 2:
            return matrix[:, row_idx * stride:(row_idx + 1) * stride]
        elif matrix.ndim == 1:
            return matrix[row_idx*stride: (row_idx+1)*stride]
        else:
            raise NotImplementedError


    """ This requires W, U and b initializer
    """
    def compute(self, input_tv, mask=None):
        n_steps = input_tv.shape[0]
        if input_tv.ndim == 3:
            n_samples = input_tv.shape[1]
        else:
            n_samples = 1


        def __step(x_, m_, t, node_h, node_c, U):
            child_h = node_h * m_
            child_c = node_c * m_
            if self.go_backwards:
                valid_mask = m_[t:]
            else:
                valid_mask = m_[:t+1]
            curr_h, curr_c = self.recursive_unit(x_, child_h, child_c, valid_mask, U)
            node_h = T.set_subtensor(node_h[:,t], curr_h)
            node_c = T.set_subtensor(node_c[:,t], curr_c)
            return node_h, node_c

        # Shapes: x_: (batch_size, 4*out_dim), children_mask: (batch_size, sent_len)
        # t: (1,), node_h: (batch_size, sent_len, out_dim), U: (out_dim, 4*out_dim)
        def __step_batch(x_, children_mask, t, node_h, node_c, U):
            child_h = node_h * children_mask[:, :, None]
            child_c = node_c * children_mask[:, :, None]
            if self.go_backwards:
                valid_mask = children_mask[:, t+1:].sum(axis=1)
            else:
                valid_mask = children_mask[:, :t].sum(axis=1)
            convert_mask = T.zeros_like(valid_mask)
            diff_mask = T.cast(T.eq(valid_mask, convert_mask), dtype=theano.config.floatX)
            valid_mask += diff_mask
            curr_h, curr_c = self.recursive_unit_batch(x_, child_h, child_c, valid_mask, U)
            node_h = T.set_subtensor(node_h[:, t, :], curr_h)
            node_c = T.set_subtensor(node_c[:, t, :], curr_c)
            return node_h, node_c

        x_in = T.dot(input_tv, self.W).astype(config.floatX) + self.b
        seq_in = [x_in, mask, T.arange(n_steps)]
        if input_tv.ndim == 3:
            lstm_step = __step_batch
            h_init = T.alloc(np_floatX(0.), n_samples, n_steps, self.out_dim)
            c_init = T.alloc(np_floatX(0.), n_samples, n_steps, self.out_dim)
        else:
            lstm_step = __step
            h_init = T.alloc(np_floatX(0.), self.out_dim, n_steps)
            c_init = T.alloc(np_floatX(0.), self.out_dim, n_steps)
        print 'lstm step:', lstm_step
        rval, _ = theano.scan(lstm_step,
                              sequences=seq_in,
                              outputs_info=[h_init, c_init],
                              non_sequences=[self.U], #, self.p],
                              go_backwards=self.go_backwards,
                              name=name_tv(self.name, 'GraphLSTM_layer'),
                              n_steps=n_steps,
                              strict=True,
        )
        if input_tv.ndim == 3:
            self.output_tv = rval[0][-1].dimshuffle(1,0,2)
            #self.output_tv = reverse(rval[0][-1].dimshuffle(1,0,2)) if self.go_backwards else rval[0][-1].dimshuffle(1,0,2)
        else:
            self.output_tv = rval[0][-1].T
            #self.output_tv = reverse(rval[0][-1].T) if self.go_backwards else rval[0][-1].T
        #print 'in GraphLSTM, variable types:', input_tv.dtype, self.output_tv.dtype
        if self.params.get(self.kn('dropout_rate'), 0.0) != 0.0:
            print 'DROP OUT!!! at circuite', self.name, 'Drop out rate: ', self.params[self.kn('dropout_rate')]
            self.output_tv = _dropout_from_layer(self.params['rng'], self.output_tv, self.params[self.kn('dropout_rate')])

    def needed_key(self):
        return self._needed_key_impl('W', 'U', 'b', 'go_backwards')

# A special BiasedLinear layer that takes advatages of the fact that the input is a one-hot vector.
class Onehot_Linear(Chip): 
    def prepend(self, previous_chip):
        self = super(Onehot_Linear, self).prepend(previous_chip)
        self.T = self._declare_mat('T', self.in_dim, self.out_dim)
        #self.b = self._declare_mat('b', self.out_dim)
        self.parameters = [self.T]
        return self
    """ Composition of Linear and Bias
    It requires a U_initializer and a b_initializer
    """
    def compute(self, input_tv):
        linear_trans = self.T[input_tv.flatten()].reshape((input_tv.shape[0], input_tv.shape[1], self.out_dim))
        self.output_tv = linear_trans #+ self.b

    def needed_key(self):
        return self._needed_key_impl('T') #, 'b') 

# A special autoencoder that takes advantages of the input being a one-hot vector
class AutoEncoder(Chip):
    def __init__(self, name, params=None):
        super(AutoEncoder, self).__init__(name, params)
        self.params[self.name+"_encoder_out_dim"] = params[self.kn('hidden_dim')]
        self.params[self.name+"_decoder_out_dim"] = params[self.kn('out_dim')]
        self.Encoder = Onehot_Linear(name+'_encoder', self.params)
        self.Decoder = BiasedLinear(name+'_decoder', self.params)

    def prepend(self, previous_chip):
        self.Decoder.prepend(self.Encoder.prepend(previous_chip))
        self.in_dim = self.Encoder.in_dim
        self.parameters = self.Encoder.parameters + self.Decoder.parameters
        return self

    # input_tv shape: (batch_size, sent_len, 1), the last dimension indicate which type it is, the value < num_arc_type
    def compute(self, input_tv):
        self.Encoder.compute(input_tv)
        self.Decoder.compute( T.nnet.sigmoid(self.Encoder.output_tv) )
        self.output_tv = T.nnet.sigmoid(self.Decoder.output_tv)

    def needed_key(self):
        return self.Encoder.needed_key() + self.Decoder.needed_key()


class GraphLSTM_WtdEmbMult(Chip):
    def prepend(self, previous_chip):
        self = super(GraphLSTM_WtdEmbMult, self).prepend(previous_chip)
        print 'graph lstm in dim:', self.in_dim, 'out dim:', self.out_dim
        self.go_backwards = self.params[self.kn('go_backwards')]
        self.W = self._declare_mat('W', self.in_dim, 4*self.out_dim)
        num_arc_types = self.params[self.kn('type_dim')]
        temp_U = np.zeros((num_arc_types, self.out_dim, 4*self.out_dim)).astype(theano.config.floatX)
        for i in range(num_arc_types):
            temp_U[i] = ArrayInit(ArrayInit.ortho).initialize(self.out_dim, 4*self.out_dim, multiplier=1)
        self.U = theano.shared(temp_U, name=tparams_make_name(self.name, 'U'))
        self.U.is_regularizable = True
        self.U = theano.shared(temp_U, name=tparams_make_name(self.name, 'U'))
        self.U.is_regularizable = True
        self.b = self._declare_mat('b', 4*self.out_dim) #, is_regularizable = False)
        #self.p = self._declare_mat('p', 3*self.out_dim)
        #type_matrix = T.eye(self.params[self.kn('arc_types')]+1, dtype=theano.config.floatX)
        self.T = self._declare_mat('T', self.params[self.kn('arc_types')], num_arc_types)
        self.parameters = [self.W, self.U, self.T, self.b] #, self.p)
        return self

    # Shapes: x_ = (4*out_dim,), child_h = (out_dim, n_steps, arc_types), child_c = (out_dim, n_steps), child_exists = (n_steps, arc_types)
    def recursive_unit(self, x_, child_h, child_c, child_exists, U):
        h_tilde = T.sum(child_h, axis=1) / child_exists.sum() #T.cast(, theano.config.floatX) 
        preact = T.tensordot(h_tilde.T, U, [[0,1],[0,1]]) + x_
        i = T.nnet.sigmoid(self.slice(preact, 0, self.out_dim) ) #+ self.slice(self.p, 0, self.out_dim)*c_prev) # Input gate
        #f = T.nnet.sigmoid(self.slice(preact, 1, self.out_dim) )#+ self.slice(self.p, 1, self.out_dim)*c_prev) # Forget gate
        f = T.nnet.sigmoid(self.slice(x_, 1, self.out_dim).dimshuffle('x', 0) + T.tensordot(child_h, self.slice(U, 1, self.out_dim), [[0, 2],[1, 0]]) )
        z = T.tanh(self.slice(preact, 3, self.out_dim)) # block input
        c = (f.T * child_c).sum(axis=1) / child_exists.sum() + i * z # cell state
        o = T.nnet.sigmoid(self.slice(preact, 2, self.out_dim) )#+ self.slice(self.p, 2, self.out_dim) * c) # output gate
        h = o * T.tanh(c)  # block output
        return h, c
    
    # Shapes: x_= (batch_size, 4*out_dim,), child_h = (batch_size, sent_len,out_dim+arc_types+1), 
    # child_c = (batch_size, sent_len, out_dim), child_exists = (batch_size, sent_len)
    def recursive_unit_batch(self, x_, child_h, child_c, child_exists, U):
        # Result shape = (batch_size, arc_types, out_dim)
        h_tilde = T.sum(child_h, axis=1) / child_exists[:, None, None] #T.cast(, theano.config.floatX) 
        preact = T.tensordot(h_tilde, U, [[1,2],[0,1]]) + x_
        i = T.nnet.sigmoid(self.slice(preact, 0, self.out_dim) ) #+ self.slice(self.p, 0, self.out_dim)*c_prev) # Input gate
        #f = T.nnet.sigmoid(self.slice(preact, 1, self.out_dim) )#+ self.slice(self.p, 1, self.out_dim)*c_prev) # Forget gate
        f = T.nnet.sigmoid(self.slice(x_, 1, self.out_dim)[:, None, :] + T.tensordot(child_h, self.slice(U, 1, self.out_dim), [[2,3],[0,1]]) ) #/ child_exists[:, :, None] 
        z = T.tanh(self.slice(preact, 3, self.out_dim)) # block input
        c = (f * child_c).sum(axis=1) / child_exists[:, None] + i * z # cell state
        o = T.nnet.sigmoid(self.slice(preact, 2, self.out_dim) )#+ self.slice(self.p, 2, self.out_dim) * c) # output gate
        h = o * T.tanh(c)  # block output
        return h, c

    def slice(self, matrix, row_idx, stride):
        if matrix.ndim == 3:
            return matrix[:, :, row_idx * stride:(row_idx + 1) * stride]
        elif matrix.ndim == 2:
            return matrix[:, row_idx * stride:(row_idx + 1) * stride]
        else:
            return matrix[row_idx*stride: (row_idx+1)*stride]


    """ This requires W, U and b initializer
    """
    def compute(self, input_tv, mask=None):
        n_steps = input_tv.shape[0]
        if input_tv.ndim == 3:
            n_samples = input_tv.shape[1]
        else:
            n_samples = 1

        # Shapes: m_ = (n_steps, arc_type), node_h = (out_dim, n_steps)
        def __step(x_, m_, t, node_h, node_c, U):
            child_h = node_h[:,:,None] * m_
            child_c = node_c * m_.sum(axis=1)
            if self.go_backwards:
                valid_mask = m_[t+1:, :]
            else:
                valid_mask = m_[:t, :]
            curr_h, curr_c = self.recursive_unit(x_, child_h, child_c, valid_mask, U)
            node_h = T.set_subtensor(node_h[:,t], curr_h)
            node_c = T.set_subtensor(node_c[:,t], curr_c)
            return node_h, node_c

        # Shapes: x_: (batch_size, 4*out_dim), m_: (batch_size, sent_len, 2)
        # t: (1,), node_h: (batch_size, sent_len, out_dim), U: (out_dim, 4*out_dim)
        def __step_batch(x_, m_, t, node_h, node_c, U, type_matrix):
            child_mask = m_[:, :, 0]
            dep_type = T.cast(m_[:, :, 1], dtype='int32')
            arc_types = type_matrix[dep_type.flatten()].reshape((node_h.shape[0], node_h.shape[1], self.params[self.kn('type_dim')]))
            child_h = node_h[:, :, :, None] * arc_types[:, :, None, :] * child_mask[:, :, None, None]
            #child_h = T.concatenate((node_h, arc_types), axis=2) * child_mask[:, :, None]
            child_c = node_c * child_mask[:, :, None]
            if self.go_backwards:
                valid_mask = child_mask[:, t+1:].sum(axis=1)
            else:
                valid_mask = child_mask[:, :t].sum(axis=1)
            convert_mask = T.zeros_like(valid_mask)
            diff_mask = T.cast(T.eq(valid_mask, convert_mask), dtype=theano.config.floatX)
            valid_mask += diff_mask
            curr_h, curr_c = self.recursive_unit_batch(x_, child_h, child_c, valid_mask, U)
            node_h = T.set_subtensor(node_h[:, t, :], curr_h)
            node_c = T.set_subtensor(node_c[:, t, :], curr_c)
            return node_h, node_c
	################################################################


        x_in = T.dot(input_tv, self.W).astype(config.floatX) + self.b
        seq_in = [x_in, mask, T.arange(n_steps)]
        if input_tv.ndim == 3:
            lstm_step = __step_batch
            h_init = T.alloc(np_floatX(0.), n_samples, n_steps, self.out_dim)
            c_init = T.alloc(np_floatX(0.), n_samples, n_steps, self.out_dim)
        else:
            lstm_step = __step
            h_init = T.alloc(np_floatX(0.), self.out_dim, n_steps)
            c_init = T.alloc(np_floatX(0.), self.out_dim, n_steps)
        print 'lstm step:', lstm_step
        rval, _ = theano.scan(lstm_step,
                              sequences=seq_in,
                              outputs_info=[h_init, c_init],
                              non_sequences=[self.U, self.T], #A], #, self.p],
                              go_backwards=self.go_backwards,
                              name=name_tv(self.name, 'WeightedAddGraphLSTM_layer'),
                              n_steps=n_steps,
                              strict=True,
        )
        if input_tv.ndim == 3:
            self.output_tv = rval[0][-1].dimshuffle(1,0,2)
            #self.output_tv = reverse(rval[0][-1].dimshuffle(1,0,2)) if self.go_backwards else rval[0][-1].dimshuffle(1,0,2)
        else:
            self.output_tv = rval[0][-1].T
            #self.output_tv = reverse(rval[0][-1].T) if self.go_backwards else rval[0][-1].T
        #print 'in GraphLSTM, variable types:', input_tv.dtype, self.output_tv.dtype
        if self.params.get(self.kn('dropout_rate'), 0.0) != 0.0:
            print 'DROP OUT!!! at circuite', self.name, 'Drop out rate: ', self.params[self.kn('dropout_rate')]
            self.output_tv = _dropout_from_layer(self.params['rng'], self.output_tv, self.params[self.kn('dropout_rate')])

    def needed_key(self):
        return self._needed_key_impl('W', 'U', 'b', 'T', 'go_backwards', 'arc_types')


class GraphLSTM_WtdAdd(Chip):
    def prepend(self, previous_chip):
        self = super(GraphLSTM_WtdAdd, self).prepend(previous_chip)
        print 'graph lstm in dim:', self.in_dim, 'out dim:', self.out_dim
        self.go_backwards = self.params[self.kn('go_backwards')]
        self.W = self._declare_mat('W', self.in_dim, 4*self.out_dim)
        num_arc_types = self.params[self.kn('type_dim')]
        temp_U = np.zeros((num_arc_types+self.out_dim, 4*self.out_dim)).astype(theano.config.floatX)
        self.U = theano.shared(temp_U, name=tparams_make_name(self.name, 'U'))
        self.U.is_regularizable = True
        self.b = self._declare_mat('b', 4*self.out_dim) #, is_regularizable = False)
        #self.p = self._declare_mat('p', 3*self.out_dim)
        #type_matrix = T.eye(self.params[self.kn('arc_types')]+1, dtype=theano.config.floatX)
        self.T = self._declare_mat('T', self.params[self.kn('arc_types')], num_arc_types)
        #self.TA = T.concatenate([T.zeros([1, self.params[self.kn('type_dim')]], dtype=theano.config.floatX), self.T], axis=0)
        #self.T = T.set_subtensor(self.T[0], 0.)
        self.parameters = [self.W, self.U, self.T, self.b] #, self.p)
        return self

    # Shapes: x_ = (4*out_dim,), child_h = (out_dim, n_steps, arc_types), child_c = (out_dim, n_steps), child_exists = (n_steps, arc_types)
    def recursive_unit(self, x_, child_h, child_c, child_exists, U):
        h_tilde = T.sum(child_h, axis=1) / child_exists.sum() #T.cast(, theano.config.floatX) 
        preact = T.tensordot(h_tilde.T, U, [[0,1],[0,1]]) + x_
        i = T.nnet.sigmoid(self.slice(preact, 0, self.out_dim) ) #+ self.slice(self.p, 0, self.out_dim)*c_prev) # Input gate
        #f = T.nnet.sigmoid(self.slice(preact, 1, self.out_dim) )#+ self.slice(self.p, 1, self.out_dim)*c_prev) # Forget gate
        f = T.nnet.sigmoid(self.slice(x_, 1, self.out_dim).dimshuffle('x', 0) + T.tensordot(child_h, self.slice(U, 1, self.out_dim), [[0, 2],[1, 0]]) )
        z = T.tanh(self.slice(preact, 3, self.out_dim)) # block input
        c = (f.T * child_c).sum(axis=1) / child_exists.sum() + i * z # cell state
        o = T.nnet.sigmoid(self.slice(preact, 2, self.out_dim) )#+ self.slice(self.p, 2, self.out_dim) * c) # output gate
        h = o * T.tanh(c)  # block output
        return h, c
    
    # Shapes: x_= (batch_size, 4*out_dim,), child_h = (batch_size, sent_len,out_dim+arc_types+1), 
    # child_c = (batch_size, sent_len, out_dim), child_exists = (batch_size, sent_len)
    def recursive_unit_batch(self, x_, child_h, child_c, child_exists, U):
        # Result shape = (batch_size, arc_types, out_dim)
        h_tilde = T.sum(child_h, axis=1) / child_exists[:, None] #T.cast(, theano.config.floatX) 
        preact = T.dot(h_tilde, U) + x_
        i = T.nnet.sigmoid(self.slice(preact, 0, self.out_dim) ) #+ self.slice(self.p, 0, self.out_dim)*c_prev) # Input gate
        #f = T.nnet.sigmoid(self.slice(preact, 1, self.out_dim) )#+ self.slice(self.p, 1, self.out_dim)*c_prev) # Forget gate
        #f = T.nnet.sigmoid(self.slice(x_, 1, self.out_dim)[:, None, :] + T.tensordot(child_h, self.slice(U, 1, self.out_dim), [[2,3],[0,1]]) ) #/ child_exists[:, :, None] 
        f = T.nnet.sigmoid(self.slice(x_, 1, self.out_dim)[:, None, :] + T.dot(child_h, self.slice(U, 1, self.out_dim)) )
        z = T.tanh(self.slice(preact, 3, self.out_dim)) # block input
        c = (f * child_c).sum(axis=1) / child_exists[:, None] + i * z # cell state
        o = T.nnet.sigmoid(self.slice(preact, 2, self.out_dim) )#+ self.slice(self.p, 2, self.out_dim) * c) # output gate
        h = o * T.tanh(c)  # block output
        return h, c

    def slice(self, matrix, row_idx, stride):
        if matrix.ndim == 3:
            return matrix[:, :, row_idx * stride:(row_idx + 1) * stride]
        elif matrix.ndim == 2:
            return matrix[:, row_idx * stride:(row_idx + 1) * stride]
        else:
            return matrix[row_idx*stride: (row_idx+1)*stride]


    """ This requires W, U and b initializer
    """
    def compute(self, input_tv, mask=None):
        n_steps = input_tv.shape[0]
        if input_tv.ndim == 3:
            n_samples = input_tv.shape[1]
        else:
            n_samples = 1

        # Shapes: m_ = (n_steps, arc_type), node_h = (out_dim, n_steps)
        def __step(x_, m_, t, node_h, node_c, U):
            child_h = node_h[:,:,None] * m_
            child_c = node_c * m_.sum(axis=1)
            if self.go_backwards:
                valid_mask = m_[t+1:, :]
            else:
                valid_mask = m_[:t, :]
            curr_h, curr_c = self.recursive_unit(x_, child_h, child_c, valid_mask, U)
            node_h = T.set_subtensor(node_h[:,t], curr_h)
            node_c = T.set_subtensor(node_c[:,t], curr_c)
            return node_h, node_c

        # Shapes: x_: (batch_size, 4*out_dim), m_: (batch_size, sent_len, 2)
        # t: (1,), node_h: (batch_size, sent_len, out_dim), U: (out_dim, 4*out_dim)
        def __step_batch(x_, m_, t, node_h, node_c, U, type_matrix):
            child_mask = m_[:, :, 0]
            dep_type = T.cast(m_[:, :, 1], dtype='int32')
            arc_types = type_matrix[dep_type.flatten()].reshape((node_h.shape[0], node_h.shape[1], self.params[self.kn('type_dim')]))
            child_h = T.concatenate((node_h, arc_types), axis=2) * child_mask[:, :, None]
            child_c = node_c * child_mask[:, :, None]
            if self.go_backwards:
                valid_mask = child_mask[:, t+1:].sum(axis=1)
            else:
                valid_mask = child_mask[:, :t].sum(axis=1)
            convert_mask = T.zeros_like(valid_mask)
            diff_mask = T.cast(T.eq(valid_mask, convert_mask), dtype=theano.config.floatX)
            valid_mask += diff_mask
            curr_h, curr_c = self.recursive_unit_batch(x_, child_h, child_c, valid_mask, U)
            node_h = T.set_subtensor(node_h[:, t, :], curr_h)
            node_c = T.set_subtensor(node_c[:, t, :], curr_c)
            return node_h, node_c

        x_in = T.dot(input_tv, self.W).astype(config.floatX) + self.b
        seq_in = [x_in, mask, T.arange(n_steps)]
        if input_tv.ndim == 3:
            lstm_step = __step_batch
            h_init = T.alloc(np_floatX(0.), n_samples, n_steps, self.out_dim)
            c_init = T.alloc(np_floatX(0.), n_samples, n_steps, self.out_dim)
        else:
            lstm_step = __step
            h_init = T.alloc(np_floatX(0.), self.out_dim, n_steps)
            c_init = T.alloc(np_floatX(0.), self.out_dim, n_steps)
        print 'lstm step:', lstm_step
        rval, _ = theano.scan(lstm_step,
                              sequences=seq_in,
                              outputs_info=[h_init, c_init],
                              non_sequences=[self.U, self.T], #A], #, self.p],
                              go_backwards=self.go_backwards,
                              name=name_tv(self.name, 'WeightedAddGraphLSTM_layer'),
                              n_steps=n_steps,
                              strict=True,
        )
        if input_tv.ndim == 3:
            self.output_tv = rval[0][-1].dimshuffle(1,0,2)
            #self.output_tv = reverse(rval[0][-1].dimshuffle(1,0,2)) if self.go_backwards else rval[0][-1].dimshuffle(1,0,2)
        else:
            self.output_tv = rval[0][-1].T
            #self.output_tv = reverse(rval[0][-1].T) if self.go_backwards else rval[0][-1].T
        #print 'in GraphLSTM, variable types:', input_tv.dtype, self.output_tv.dtype
        if self.params.get(self.kn('dropout_rate'), 0.0) != 0.0:
            print 'DROP OUT!!! at circuite', self.name, 'Drop out rate: ', self.params[self.kn('dropout_rate')]
            self.output_tv = _dropout_from_layer(self.params['rng'], self.output_tv, self.params[self.kn('dropout_rate')])

    def needed_key(self):
        return self._needed_key_impl('W', 'U', 'b', 'T', 'go_backwards', 'arc_types')


class GraphLSTM_Wtd(Chip):
    def prepend(self, previous_chip):
        self = super(GraphLSTM_Wtd, self).prepend(previous_chip)
        print 'graph lstm in dim:', self.in_dim, 'out dim:', self.out_dim
        self.go_backwards = self.params[self.kn('go_backwards')]
        self.W = self._declare_mat('W', self.in_dim, 4*self.out_dim)
        num_arc_types = self.params[self.kn('arc_types')]
        temp_U = np.zeros((num_arc_types, self.out_dim, 4*self.out_dim)).astype(theano.config.floatX)
        for i in range(num_arc_types):
            temp_U[i] = ArrayInit(ArrayInit.ortho).initialize(self.out_dim, 4*self.out_dim, multiplier=1)
        self.U = theano.shared(temp_U, name=tparams_make_name(self.name, 'U'))
        self.U.is_regularizable = True
        self.b = self._declare_mat('b', 4*self.out_dim) #, is_regularizable = False)
        #self.p = self._declare_mat('p', 3*self.out_dim)
        self.parameters = [self.W, self.U, self.b] #, self.p)
        return self

    # Shapes: x_ = (4*out_dim,), child_h = (out_dim, n_steps, arc_types), child_c = (out_dim, n_steps), child_exists = (n_steps, arc_types)
    def recursive_unit(self, x_, child_h, child_c, child_exists, U):
        h_tilde = T.sum(child_h, axis=1) / child_exists.sum() #T.cast(, theano.config.floatX) 
        preact = T.tensordot(h_tilde.T, U, [[0,1],[0,1]]) + x_
        i = T.nnet.sigmoid(self.slice(preact, 0, self.out_dim) ) #+ self.slice(self.p, 0, self.out_dim)*c_prev) # Input gate
        #f = T.nnet.sigmoid(self.slice(preact, 1, self.out_dim) )#+ self.slice(self.p, 1, self.out_dim)*c_prev) # Forget gate
        f = T.nnet.sigmoid(self.slice(x_, 1, self.out_dim).dimshuffle('x', 0) + T.tensordot(child_h, self.slice(U, 1, self.out_dim), [[0, 2],[1, 0]]) )
        z = T.tanh(self.slice(preact, 3, self.out_dim)) # block input
        c = (f.T * child_c).sum(axis=1) / child_exists.sum() + i * z # cell state
        o = T.nnet.sigmoid(self.slice(preact, 2, self.out_dim) )#+ self.slice(self.p, 2, self.out_dim) * c) # output gate
        h = o * T.tanh(c)  # block output
        return h, c
    
    # Shapes: x_= (batch_size, 4*out_dim,), child_h = (batch_size, sent_len, arc_types, out_dim), 
    # child_c = (batch_size, sent_len, out_dim), child_exists = (batch_size, sent_len, arc_types)
    def recursive_unit_batch(self, x_, child_h, child_c, child_exists, U):
        # Result shape = (batch_size, arc_types, out_dim)
        h_tilde = T.sum(child_h, axis=1) / child_exists[:, None, None] #T.cast(, theano.config.floatX) 
        preact = T.tensordot(h_tilde, U, [[1,2],[0,1]]) + x_
        i = T.nnet.sigmoid(self.slice(preact, 0, self.out_dim) ) #+ self.slice(self.p, 0, self.out_dim)*c_prev) # Input gate
        #f = T.nnet.sigmoid(self.slice(preact, 1, self.out_dim) )#+ self.slice(self.p, 1, self.out_dim)*c_prev) # Forget gate
        f = T.nnet.sigmoid(self.slice(x_, 1, self.out_dim)[:, None, :] + T.tensordot(child_h, self.slice(U, 1, self.out_dim), [[2,3],[0,1]]) ) #/ child_exists[:, :, None] 
        z = T.tanh(self.slice(preact, 3, self.out_dim)) # block input
        c = (f * child_c).sum(axis=1) / child_exists[:, None] + i * z # cell state
        o = T.nnet.sigmoid(self.slice(preact, 2, self.out_dim) )#+ self.slice(self.p, 2, self.out_dim) * c) # output gate
        h = o * T.tanh(c)  # block output
        return h, c

    def slice(self, matrix, row_idx, stride):
        if matrix.ndim == 3:
            return matrix[:, :, row_idx * stride:(row_idx + 1) * stride]
        elif matrix.ndim == 2:
            return matrix[:, row_idx * stride:(row_idx + 1) * stride]
        else:
            return matrix[row_idx*stride: (row_idx+1)*stride]


    """ This requires W, U and b initializer
    """
    def compute(self, input_tv, mask=None):
        n_steps = input_tv.shape[0]
        if input_tv.ndim == 3:
            n_samples = input_tv.shape[1]
        else:
            n_samples = 1

        # Shapes: m_ = (n_steps, arc_type), node_h = (out_dim, n_steps)
        def __step(x_, m_, t, node_h, node_c, U):
            child_h = node_h[:,:,None] * m_
            child_c = node_c * m_.sum(axis=1)
            if self.go_backwards:
                valid_mask = m_[t+1:, :]
            else:
                valid_mask = m_[:t, :]
            curr_h, curr_c = self.recursive_unit(x_, child_h, child_c, valid_mask, U)
            node_h = T.set_subtensor(node_h[:,t], curr_h)
            node_c = T.set_subtensor(node_c[:,t], curr_c)
            return node_h, node_c

        # Shapes: x_: (batch_size, 4*out_dim), m_: (batch_size, sent_len, arc_type)
        # t: (1,), node_h: (batch_size, sent_len, out_dim), U: (out_dim, 4*out_dim)
        def __step_batch(x_, m_, t, node_h, node_c, U):
            child_h = node_h[:, :, None, :] * m_[:, :, :, None]
            child_c = node_c * m_.sum(axis=2)[:, :, None]
            if self.go_backwards:
                valid_mask = m_[:, t+1:, :].sum(axis=(1,2))
            else:
                valid_mask = m_[:, :t, :].sum(axis=(1,2))
            convert_mask = T.zeros_like(valid_mask)
            diff_mask = T.cast(T.eq(valid_mask, convert_mask), dtype=theano.config.floatX)
            valid_mask += diff_mask
            curr_h, curr_c = self.recursive_unit_batch(x_, child_h, child_c, valid_mask, U)
            node_h = T.set_subtensor(node_h[:, t, :], curr_h)
            node_c = T.set_subtensor(node_c[:, t, :], curr_c)
            return node_h, node_c

        x_in = T.dot(input_tv, self.W).astype(config.floatX) + self.b
        seq_in = [x_in, mask, T.arange(n_steps)]
        if input_tv.ndim == 3:
            lstm_step = __step_batch
            h_init = T.alloc(np_floatX(0.), n_samples, n_steps, self.out_dim)
            c_init = T.alloc(np_floatX(0.), n_samples, n_steps, self.out_dim)
        else:
            lstm_step = __step
            h_init = T.alloc(np_floatX(0.), self.out_dim, n_steps)
            c_init = T.alloc(np_floatX(0.), self.out_dim, n_steps)
        print 'lstm step:', lstm_step
        rval, _ = theano.scan(lstm_step,
                              sequences=seq_in,
                              outputs_info=[h_init, c_init],
                              non_sequences=[self.U], #, self.p],
                              go_backwards=self.go_backwards,
                              name=name_tv(self.name, 'WeightedGraphLSTM_layer'),
                              n_steps=n_steps,
                              strict=True,
        )
        if input_tv.ndim == 3:
            self.output_tv = rval[0][-1].dimshuffle(1,0,2)
            #self.output_tv = reverse(rval[0][-1].dimshuffle(1,0,2)) if self.go_backwards else rval[0][-1].dimshuffle(1,0,2)
        else:
            self.output_tv = rval[0][-1].T
            #self.output_tv = reverse(rval[0][-1].T) if self.go_backwards else rval[0][-1].T
        #print 'in GraphLSTM, variable types:', input_tv.dtype, self.output_tv.dtype
        if self.params.get(self.kn('dropout_rate'), 0.0) != 0.0:
            print 'DROP OUT!!! at circuite', self.name, 'Drop out rate: ', self.params[self.kn('dropout_rate')]
            self.output_tv = _dropout_from_layer(self.params['rng'], self.output_tv, self.params[self.kn('dropout_rate')])

    def needed_key(self):
        return self._needed_key_impl('W', 'U', 'b', 'go_backwards', 'arc_types')


class BiLSTM(Chip):
    def __init__(self, name, params=None):
        super(BiLSTM, self).__init__(name, params)
        print 'print bilstm parameters:', self.params
        self.params[self.name+"_forward_go_backwards"] = False
        self.params[self.name+"_backward_go_backwards"] = True
        self.params[self.name+"_forward_out_dim"] = params[self.kn('out_dim')]
        self.params[self.name+"_backward_out_dim"] = params[self.kn('out_dim')]
        self.forward_chip = LSTM(self.name+"_forward", self.params) #
        self.backward_chip = LSTM(self.name+"_backward", self.params) #

    def prepend(self, previous_chip):
        self.forward_chip.prepend(previous_chip)
        self.backward_chip.prepend(previous_chip)
        self.in_dim = self.forward_chip.in_dim
        self.out_dim = self.forward_chip.out_dim + self.backward_chip.out_dim
        self.parameters = self.forward_chip.parameters + self.backward_chip.parameters
        return self

    """ This requires W, U and b initializer
    """
    def compute(self, input_tv, mask=None):
        # Before creating the sub LSTM's set the out_dim to half
        # Basically this setting would be used by the sub LSTMs
        self.forward_chip.compute(input_tv, mask)
        self.backward_chip.compute(input_tv, mask)
        # Without mini-batch, the output shape is (sent_len, hidden_dim)
        # With mini-batch, the shape is (sent_len, num_sample, hidden_dim)
        self.output_tv = T.concatenate([self.forward_chip.output_tv, self.backward_chip.output_tv], axis=-1)
        if self.params.get(self.kn('dropout_rate'), 0.0) != 0.0:
            print 'DROP OUT!!! at circuite', self.name, 'Drop out rate: ', self.params[self.kn('dropout_rate')]
            self.output_tv = _dropout_from_layer(self.params['rng'], self.output_tv, self.params[self.kn('dropout_rate')])
        #self.out_dim = self.forward_chip.out_dim + self.backward_chip.out_dim

    def needed_key(self):
        return self.forward_chip.needed_key() + self.backward_chip.needed_key()


class BiGraphLSTM(BiLSTM):
    def __init__(self, name, params=None):
        super(BiGraphLSTM, self).__init__(name, params)
        print 'print bi-graph-lstm parameters:', self.params
        self.params[self.name+"_forward_go_backwards"] = False
        self.params[self.name+"_backward_go_backwards"] = True
        self.params[self.name+"_forward_out_dim"] = params[self.kn('out_dim')]
        self.params[self.name+"_backward_out_dim"] = params[self.kn('out_dim')]
        self.forward_chip = GraphLSTM(self.name+"_forward", self.params) #
        self.backward_chip = GraphLSTM(self.name+"_backward", self.params) #
        #self.params[self.kn('win')] = 2


class BiGraphLSTM_Wtd(BiLSTM):
    def __init__(self, name, params=None):
        super(BiGraphLSTM_Wtd, self).__init__(name, params)
        print 'print bi-weighted-graph-lstm parameters:', self.params
        self.params[self.name+"_forward_go_backwards"] = False
        self.params[self.name+"_backward_go_backwards"] = True
        self.params[self.name+"_forward_out_dim"] = params[self.kn('out_dim')]
        self.params[self.name+"_backward_out_dim"] = params[self.kn('out_dim')]
        self.params[self.name+"_forward_arc_types"] = params[self.kn('arc_types')]
        self.params[self.name+"_backward_arc_types"] = params[self.kn('arc_types')]
        self.forward_chip = GraphLSTM_Wtd(self.name+"_forward", self.params) #
        self.backward_chip = GraphLSTM_Wtd(self.name+"_backward", self.params) #
        #self.params[self.kn('win')] = 2


class BiGraphLSTM_WtdAdd(BiLSTM):
    def __init__(self, name, params=None):
        super(BiGraphLSTM_WtdAdd, self).__init__(name, params)
        print 'print bi-weighted-graph-lstm parameters:', self.params
        self.params[self.name+"_forward_go_backwards"] = False
        self.params[self.name+"_backward_go_backwards"] = True
        self.params[self.name+"_forward_out_dim"] = params[self.kn('out_dim')]
        self.params[self.name+"_backward_out_dim"] = params[self.kn('out_dim')]
        self.params[self.name+"_forward_arc_types"] = params[self.kn('arc_types')]
        self.params[self.name+"_backward_arc_types"] = params[self.kn('arc_types')]
        self.params[self.name+"_forward_type_dim"] = params[self.kn('type_dim')]
        self.params[self.name+"_backward_type_dim"] = params[self.kn('type_dim')]
        self.forward_chip = GraphLSTM_WtdAdd(self.name+"_forward", self.params) #
        self.backward_chip = GraphLSTM_WtdAdd(self.name+"_backward", self.params) #
        #self.params[self.kn('win')] = 2


class BiGraphLSTM_WtdEmbMult(BiLSTM):
    def __init__(self, name, params=None):
        super(BiGraphLSTM_WtdEmbMult, self).__init__(name, params)
        print 'print bi-weighted-graph-lstm parameters:', self.params
        self.params[self.name+"_forward_go_backwards"] = False
        self.params[self.name+"_backward_go_backwards"] = True
        self.params[self.name+"_forward_out_dim"] = params[self.kn('out_dim')]
        self.params[self.name+"_backward_out_dim"] = params[self.kn('out_dim')]
        self.params[self.name+"_forward_arc_types"] = params[self.kn('arc_types')]
        self.params[self.name+"_backward_arc_types"] = params[self.kn('arc_types')]
        self.params[self.name+"_forward_type_dim"] = params[self.kn('type_dim')]
        self.params[self.name+"_backward_type_dim"] = params[self.kn('type_dim')]
        self.forward_chip = GraphLSTM_WtdEmbMult(self.name+"_forward", self.params) 
        self.backward_chip = GraphLSTM_WtdEmbMult(self.name+"_backward", self.params) #
        #self.params[self.kn('win')] = 2



class L2Reg(Chip):
    """ This supposes that the previous chip would have a score attribute.
    And it literally only changes the score attribute by adding the regularization term
    on top of it.
    """
    def prepend(self, previous_chip):
        self.previous_chip = previous_chip
        super(L2Reg, self).prepend(previous_chip)
        return self

    def compute(self, input_tv):
        L2 = T.sum(T.stack([T.sum(self.params[k]*self.params[k])
            for k
            in self.regularizable_variables()]))
        L2.name = tparams_make_name(self.name, 'L2')
        self.score = self.previous_chip.score + self.params[self.kn('reg_weight')] * L2

    def __getattr__(self, item):
        """ Inherit all the attributes of the previous chip.
        At present I can only see this functionality being useful
        for the case of the Slice and Regularization chip. Maybe we would move
        this down in case it is found necessary later on, but there is
        chance of abuse.
        """
        try:
            return getattr(self.previous_chip, item)
        except KeyError:
            raise AttributeError(item)

    def needed_key(self):
        return self._needed_key_impl('reg_weight')

