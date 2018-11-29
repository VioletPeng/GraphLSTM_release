#!/usr/bin/python

import argparse, os, random, subprocess, sys, time
import theano, numpy
import theano.tensor as T
from copy import copy
from neural_lib import StackConfig, ArrayInit
from neural_architectures import CNNRelation, LSTMRelation, LSTMRelation_multitask, GraphLSTMRelation, WeightedGraphLSTMRelation, WeightedAddGraphLSTMRelation, WeightedGraphLSTMRelation_multitask, WeightedAddGraphLSTMRelation_multitask
from train_util import dict_from_argparse, shuffle, create_relation_circuit, convert_id_to_word, add_arg, add_arg_to_L, conv_data_graph, create_multitask_relation_circuit 
from train_util import sgd, adadelta, rmsprop, read_matrix_from_gzip, read_matrix_from_file, read_matrix_and_idmap_from_file, batch_run_func, get_minibatches_idx, save_parameters, load_params
from data_process import load_data, load_data_cv, prepare_data 
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh


''' Convert the entity index: from indexing to dense-vector(but many zero entries) multiplication.'''
def conv_idxs(idxs, length):
    new_idxs = [numpy.zeros(length).astype(theano.config.floatX) for i in range(len(idxs))]
    for i, idx in enumerate(idxs):
        new_idxs[i][idx] = 1.0
    return new_idxs

''' For prediction, both batch version and non-batch version'''
def predict(_args, lex_test, idxs_test, f_classify, groundtruth_test, batchsize=1, graph=False, dep=None, weighted=False, print_prediction=False, prediction_file=None):
    ''' On the test set predict the labels using f_classify.
    Compare those labels against groundtruth.

    It returns a dictionary 'results' that contains
    f1 : F1 or Accuracy
    p : Precision
    r : Recall
    '''
    predictions_test = []
    if print_prediction:
        assert prediction_file is not None
        pred_file = open(prediction_file, 'w')
    if batchsize > 1:
        nb_idxs = get_minibatches_idx(len(lex_test), batchsize, shuffle=False)
        for i, tr_idxs in enumerate(nb_idxs):
            words = [lex_test[ii] for ii in tr_idxs]
            eidxs = [idxs_test[ii] for ii in tr_idxs]
            #labels = [groundtruth_test[ii] for ii in tr_idxs]
            orig_eidxs = eidxs
            if graph:
                assert dep is not None
                masks = [dep[ii] for ii in tr_idxs]
            else:
                masks = None
            x, masks, eidxs = prepare_data(words, eidxs, masks, maxlen=200)
            if weighted or not graph:
                pred_all = f_classify(x, masks, *eidxs)
                predictions_test.extend(list(numpy.argmax(pred_all, axis=1))) #[0]))
            else:
                pred_all = f_classify(x, masks.sum(axis=-1), *eidxs)
                predictions_test.extend(list(numpy.argmax(pred_all, axis=1)))
            if print_prediction:
                for idx, p in zip(tr_idxs, pred_all): 
                    pred_file.write(str(idx) + '\t' + str(p[1]) + '\n')
    else:
        for i, (word, idxs) in enumerate(zip(lex_test, idxs_test)):
            idxs = conv_idxs(idxs, len(word))
            if graph:
                assert dep is not None
                if weighted:
                    predictions_test.append(f_classify(word, dep[i], *idxs))  #.sum(axis=-1) 
                else:
                    predictions_test.append(f_classify(word, dep[i].sum(axis=-1), *idxs))  #.sum(axis=-1) 
            else:
                predictions_test.append(f_classify(word, *idxs))
    print 'in predict,', len(predictions_test), len(groundtruth_test)
    if print_prediction:
        pred_file.close()
    #results = eval_logitReg_F1(predictions_test, groundtruth_test) 
    results = eval_logitReg_accuracy(predictions_test, groundtruth_test)
    return results, predictions_test

def eval_logitReg_accuracy(predictions, goldens):
    assert len(predictions) == len(goldens)
    correct = 0.0
    for p, g in zip(predictions, goldens):
        if p == g:
            correct += 1.0
    return correct/len(predictions)

def eval_logitReg_F1(predictions, goldens):
    assert len(predictions) == len(goldens)
    tp, fp, fn = 0.0, 0.0, 0.0
    for p, g in zip(predictions, goldens):
        if p == 1:
            if g == 1:
                tp += 1.0
            else:
                fp += 1.0
        else:
            if g == 1:
                fn += 1.0
    prec = tp / (tp + fp) if (tp != 0 or fp != 0) else 0
    recall = tp / (tp + fn) if (tp != 0 or fn != 0) else 0 
    #print 'precision:', prec, 'recall:', recall
    F1 = 2*prec*recall / (prec + recall) if (prec != 0 or recall != 0) else 0
    return prec, recall, F1

''' For training on single-task setting, both batch and non-batch version'''
def train_single(train_lex, train_idxs, train_y, _args, f_cost, f_update, epoch_id, learning_rate, nsentences, batchsize=1, dep=None, weighted=False):
    ''' This function is called from the main method. and it is primarily responsible for updating the
    parameters. Because of the way that create_relation_circuit works that creates f_cost, f_update etc. this function
    needs to be flexible and can't be put in a lib.
    Look at lstm_dependency_parsing_simplification.py for more pointers.
    '''
    # None-batched version
    def train_instance(words, idxs, label, learning_rate, f_cost, f_update):
        ' Since function is called only for side effects, it is likely useless anywhere else'
        if words.shape[0] < 2:
            return 0.0
        inputs = idxs + [words, label]
        iter_cost = f_cost(*inputs) #words, id1, id2, labels)
        f_update(learning_rate)
        return iter_cost

    # Mini-batch version
    def train_batch(words, masks, idxs, label, learning_rate, f_cost, f_update):
        if words.shape[0] < 2:
            return 0.0
        inputs = idxs + [words, masks, label]
        iter_cost = f_cost(*inputs) #words, id1, id2, labels)
        f_update(learning_rate)
        return iter_cost

    ## main body of train
    if dep:
        shuffle([train_lex, train_idxs, train_y, dep], _args.seed)
    else:
        shuffle([train_lex, train_idxs, train_y], _args.seed)
    if nsentences < len(train_lex):
        train_lex = train_lex[:nsentences]
        train_idxs = train_idxs[:nsentences]
        train_y = train_y[:nsentences]
    tic = time.time()
    aggregate_cost = 0.0
    temp_cost_arr = [0.0] * 2
    
    # make the judge on whether use mini-batch or not.
    # No mini-batch
    if batchsize == 1:
        for i, (words, idxs, label) in enumerate(zip(train_lex, train_idxs, train_y)):
            if len(words) < 2:
                continue
            #assert len(words) == len(labels) #+ 2
            idxs = conv_idxs(idxs, len(words))
            if _args.graph:
                assert dep is not None
                if weighted:
                    aggregate_cost += train_batch(words, dep[i], idxs, label, learning_rate, f_cost, f_update)
                else:
                    aggregate_cost += train_batch(words, dep[i].sum(axis=-1), idxs, label, learning_rate, f_cost, f_update)
            else:
                aggregate_cost += train_instance(words, idxs, label, learning_rate, f_cost, f_update)
            if _args.verbose == 2 and i % 10 == 0:
                print '[learning] epoch %i >> %2.2f%%' % (epoch_id, (i + 1) * 100. / nsentences),
                print 'completed in %.2f (sec). << avg loss: %.2f <<\r' % (time.time() - tic, aggregate_cost/(i+1)),
                sys.stdout.flush()
    # Mini-batch
    else:
        nb_idxs = get_minibatches_idx(len(train_lex), batchsize, shuffle=False)
        nbatches = len(nb_idxs)
        for i, tr_idxs in enumerate(nb_idxs):
            words = [train_lex[ii] for ii in tr_idxs]
            eidxs = [train_idxs[ii] for ii in tr_idxs]
            labels = [train_y[ii] for ii in tr_idxs]
            orig_eidxs = eidxs
            if _args.graph:
                assert dep is not None
                masks = [dep[ii] for ii in tr_idxs]
            else:
                masks = None
            x, masks, eidxs = prepare_data(words, eidxs, masks, maxlen=200)
            
            #print 'mask shape:', masks.shape
            if weighted or dep is None:
                iter_cost = train_batch(x, masks, eidxs, labels, learning_rate, f_cost, f_update)
                aggregate_cost += iter_cost#[0]
            else:
                aggregate_cost += train_batch(x, masks.sum(axis=-1), eidxs, labels, learning_rate, f_cost, f_update)
            if _args.verbose == 2 :
                print '[learning] epoch %i >> %2.2f%%' % (epoch_id, (i + 1) * 100. / nbatches),
                print 'completed in %.2f (sec). << avg loss: %.2f <<\r' % (time.time() - tic, aggregate_cost/(i+1)),
                #print 'completed in %.2f (sec). << avg loss: %.2f <<%%' % (time.time() - tic, aggregate_cost/(i+1)),
                #print 'average cost for each part: (%.2f, %.2f) <<\r' %(temp_cost_arr[0]/(i+1), temp_cost_arr[1]/(i+1)),
                sys.stdout.flush()
    if _args.verbose == 2:
        print '\n>> Epoch completed in %.2f (sec) <<' % (time.time() - tic), 'training cost: %.2f' % (aggregate_cost)


''' For training on multi-task setting, both batch and non-batch version'''
def train_alternative(_args, f_costs_and_updates, epoch_id, learning_rate_arr, nsentences_arr, words_arr, label_arr, idx_arr, dep_mask_arr, batch_size):
    num_tasks = len(f_costs_and_updates)
    print 'num_tasks:', num_tasks
    for i in range(num_tasks):
        f_cost, f_update = f_costs_and_updates[i]
        nsent = nsentences_arr[i]
        if nsent < len(words_arr[i]):
            if epoch_id == 0:
                if dep_mask_arr[0] is not None:
                    shuffle([words_arr[i], idx_arr[i], label_arr[i], dep_mask_arr[i]], _args.seed)
                else:
                    shuffle([words_arr[i], idx_arr[i], label_arr[i]], _args.seed)
            if _args.graph:
                train_single(words_arr[i][epoch_id*nsent:(epoch_id+1)*nsent], idx_arr[i][epoch_id*nsent:(epoch_id+1)*nsent], label_arr[i][epoch_id*nsent:(epoch_id+1)*nsent], _args, f_cost, f_update, epoch_id, learning_rate_arr[i], nsentences_arr[i], batch_size, dep_mask_arr[i][epoch_id*nsent:(epoch_id+1)*nsent], _args.weighted)
            else:
                train_single(words_arr[i][epoch_id*nsent:(epoch_id+1)*nsent], idx_arr[i][epoch_id*nsent:(epoch_id+1)*nsent], label_arr[i][epoch_id*nsent:(epoch_id+1)*nsent], _args, f_cost, f_update, epoch_id, learning_rate_arr[i], nsentences_arr[i], batch_size, None, _args.weighted)
        else:
            if _args.graph:
                train_single(words_arr[i], idx_arr[i], label_arr[i], _args, f_cost, f_update, epoch_id, learning_rate_arr[i], nsentences_arr[i], batch_size, dep_mask_arr[i], _args.weighted)
            else:
                train_single(words_arr[i], idx_arr[i], label_arr[i], _args, f_cost, f_update, epoch_id, learning_rate_arr[i], nsentences_arr[i], batch_size, None, _args.weighted)

''' Initialize some parameters for training and prediction'''
def prepare_corpus(_args):
    numpy.random.seed(_args.seed)
    random.seed(_args.seed)
    if _args.win_r or _args.win_l:
        _args.wemb1_win = _args.win_r - _args.win_l + 1
    word2idx = _args.dicts['words2idx']
    _args.label2idx = _args.dicts['labels2idx']
    #_args.idx2word = dict((k, v) for v, k in word2idx.iteritems())
    _args.nsentences = len(_args.train_set[1])
    _args.voc_size = len(word2idx)
    _args.y_dim = len(_args.label2idx)
    if 'arcs2idx' in _args.dicts:
        _args.lstm_arc_types = len(_args.dicts['arcs2idx'])
        print 'lstm arc types =', _args.lstm_arc_types
    #del _args.dicts
    #_args.groundtruth_valid = convert_id_to_word(_args.valid_set[-1], _args.idx2label)
    #_args.groundtruth_test = convert_id_to_word(_args.test_set[-1], _args.idx2label)
    _args.logistic_regression_out_dim = len(_args.label2idx)
    eval_args(_args)
    _args.lstm_go_backwards = True #False
    try:
        print 'Circuit:', _args.circuit.__name__
        print 'Chkpt1', len(_args.label2idx), _args.nsentences, _args.train_set[1][0], _args.voc_size, _args.train_set[2][0], _args.valid_set[1][0], _args.valid_set[2][0]
        print 'Chkpt2', _args.wemb1_T_initializer.matrix.shape, _args.wemb1_T_initializer.matrix[0]
    except AttributeError:
        pass

''' Compile the architecture.'''
def compile_circuit(_args):
    ### build circuits. ###
    (_args.f_cost, _args.f_update, _args.f_classify, cargs) = create_relation_circuit(_args, StackConfig)
    _args.train_func = train_single
    print "Finished Compiling"
    return cargs

def convert_args(_args, prefix):
    from types import StringType
    for a in _args.__dict__:  #TOPO_PARAM + TRAIN_PARAM:
        try:
            if type(a) is StringType and a.startswith(prefix):
                _args.__dict__[a[len(prefix)+1:]] = _args.__dict__[a]
                del _args.__dict__[a]
        except:
            pass
      
def eval_args(_args):
    for a in _args.__dict__:  #TOPO_PARAM + TRAIN_PARAM:
        try:
            _args.__dict__[a] = eval(_args.__dict__[a])
        except:
            pass

def run_wild_prediction(_args):
    best_f1 = -numpy.inf
    param = dict(clr = _args.lr, ce = 0, be = 0, epoch_id = -1)
    cargs = compile_circuit(_args)
    while param['epoch_id']+1 < _args.nepochs:
        param['epoch_id'] += 1
        run_training(_args, param)
    train_lex, train_y, train_idxs, train_dep = _args.train_set
    valid_lex, valid_y, valid_idxs, valid_dep = _args.valid_set
    res_train, _ = predict(_args, train_lex, train_idxs, _args.f_classify, train_y, _args.batch_size, _args.graph, train_dep, _args.weighted)
    res_valid, _ = predict(_args, valid_lex, valid_idxs, _args.f_classify, valid_y, _args.batch_size, _args.graph, valid_dep, _args.weighted, _args.print_prediction, _args.prediction_file)
    if _args.verbose:
        print('TEST: epoch', param['epoch_id'],
                'train performances'   , res_train,
                'valid performances'   , res_valid)
    print('Training accuracy', res_train,
          )


def run_training(_args, param):
    train_lex, train_y, train_idxs, train_dep = _args.train_set
    _args.train_func(train_lex, train_idxs, train_y, _args, _args.f_cost, _args.f_update, param['epoch_id'], param['clr'], _args.nsentences, _args.batch_size, train_dep, _args.weighted)
   

def run_epochs(_args, test_data=True):
    best_f1 = -numpy.inf
    param = dict(clr = _args.lr, ce = 0, be = 0, epoch_id = -1)
    cargs = compile_circuit(_args)
    while param['epoch_id']+1 < _args.nepochs:
        param['epoch_id'] += 1
        run_training(_args, param)
        train_lex, train_y, train_idxs, train_dep = _args.train_set
        valid_lex, valid_y, valid_idxs, valid_dep = _args.valid_set
        if test_data:
            test_lex, test_y, test_idxs, test_dep = _args.test_set
        res_train, _ = predict(_args, train_lex, train_idxs, _args.f_classify, train_y, _args.batch_size, _args.graph, train_dep, _args.weighted)
        res_valid, _ = predict(_args, valid_lex, valid_idxs, _args.f_classify, valid_y, _args.batch_size, _args.graph, valid_dep, _args.weighted)
        if _args.verbose:
            print('TEST: epoch', param['epoch_id'],
                  'train performances'   , res_train,
                  'valid performances'   , res_valid)
        # If this update created a 'new best' model then save it.
        if type(res_valid) is tuple:
            curr_f1 = res_valid[-1]
        else:
            curr_f1 = res_valid
        if curr_f1 > best_f1:
            best_f1 = curr_f1 
            param['be']  = param['epoch_id']
            param['last_decay'] = param['be']
            param['vf1'] = res_valid 
            param['best_classifier'] = _args.f_classify
            if test_data:
                res_test, _ = predict(_args, test_lex, test_idxs, _args.f_classify, test_y, _args.batch_size, _args.graph, test_dep, _args.weighted, _args.print_prediction, _args.prediction_file)
                # get the prediction, convert and write to concrete.
                param['tf1'] = res_test
                print '\nEpoch:%d'%param['be'], 'Test accuracy:', res_test, '\n'        
                ############## Test load parameters!!!! ########
                #cargs = {}
                #print "loading parameters!"
                #load_params(_args.parameters_file, cargs)
                #f_classify = cargs['f_classify']
                #res_test, _ = predict(_args, test_lex, test_idxs, f_classify, test_y, _args.batch_size, _args.graph, test_dep, _args.weighted)
                #print 'Load parameter test accuracy:', res_test, '\n'        
                ############## End Test ##############
        if _args.decay and (param['epoch_id'] - param['last_decay']) >= _args.decay_epochs:
            print 'learning rate decay at epoch', param['epoch_id'], '! Previous best epoch number:', param['be']
            param['last_decay'] = param['epoch_id']
            param['clr'] *= 0.5
        # If learning rate goes down to minimum then break.
        if param['clr'] < _args.minimum_lr:
            print "\nLearning rate became too small, breaking out of training"
            break
    print('BEST RESULT: epoch', param['be'],
          'valid accuracy', param['vf1'],
          )
    if test_data:
        print('best test accuracy', param['tf1'],
          )


def run_multi_task(_args, cargs, num_domains, num_tasks, mode='alternative', test_data=False):
    param = dict(epoch_id = -1)
    for i in range(num_domains*num_tasks):
        param['vf1'+str(i)] = -numpy.inf
    while param['epoch_id']+1 < _args.nepochs:
        param['epoch_id'] += 1
        # Four sets of training in _args.train, each contains feature, lex and label, so the resulted train_data grouped feature, lex and labels together.
        train_data = [list(group) for group in zip(*_args.trainSet)]
        if mode == 'alternative':
            train_alternative(_args, _args.f_costs_and_updates, param['epoch_id'], _args.lr_arr, _args.nsentences_arr, train_data[0], train_data[1], train_data[2], train_data[3], _args.batch_size)
        else:
            raise NotImplementedError 
        for i, dev in enumerate(_args.devSet):
            train_lex, train_y, train_idxs, train_dep = _args.trainSet[i]
            valid_lex, valid_y, valid_idxs, valid_dep = dev
            if i == 0:
                train_lex, train_y, train_idxs = train_lex[:_args.nsentences_arr[i]], train_y[:_args.nsentences_arr[i]], train_idxs[:_args.nsentences_arr[i]]
                sample_idx = random.sample(range(len(valid_y)), 3000)
                valid_lex = [valid_lex[idx] for idx in sample_idx]
                valid_idxs = [valid_idxs[idx] for idx in sample_idx]
                valid_y = [valid_y[idx] for idx in sample_idx]
                if _args.graph:
                    train_dep = train_dep[:_args.nsentences_arr[i]]
                    valid_dep = [valid_dep[idx] for idx in sample_idx]
            res_train, _ = predict(_args, train_lex, train_idxs, _args.f_classifies[i], train_y, _args.batch_size, _args.graph, train_dep, _args.weighted)
            res_valid, _ = predict(_args, valid_lex, valid_idxs, _args.f_classifies[i], valid_y, _args.batch_size, _args.graph, valid_dep, _args.weighted, _args.print_prediction, _args.prediction_files[i])
            if _args.verbose:
                print('TEST: epoch', param['epoch_id'], 'task:', i,
                        'train F1'   , res_train,
                        'valid F1'   , res_valid)
            if res_valid > param['vf1'+str(i)]: 
                param['be'+str(i)]  = param['epoch_id']
                param['last_decay'+str(i)] = param['be'+str(i)]
                param['vf1'+str(i)] = res_valid 
                #cargs['f_classify_'+str(i)] = _args.f_classifies[i]
                #save_parameters(_args.parameters_file, cargs)
                if test_data:
                    if i == 0:
                        tws, tls, tis, tds = valid_lex, valid_y, valid_idxs, valid_dep 
                    else:
                        tws, tls, tis, tds = _args.testSet[i]
                    pred_res, _ = predict(_args, tws, tis, _args.f_classifies[i], tls, _args.batch_size, _args.graph, tds, _args.weighted)
                    # also store test performance here.
                    param['tf1'+str(i)] = pred_res
                    param['best_classifier'+str(i)] = _args.f_classifies[i]
                    if _args.verbose:
                        print('test F1'   , pred_res)
            if _args.decay and (param['epoch_id'] - param['last_decay'+str(i)]) >= args.decay_epochs:
                print 'learning rate decay at epoch', param['epoch_id'], ', for dataset', i, '! Previous best epoch number:', param['be'+str(i)], 'current learning rates:', _args.lr_arr
                param['last_decay'+str(i)] = param['epoch_id']
                _args.lr_arr[i] *= 0.5
            # If learning rate goes down to minimum then break.
            if _args.lr_arr[i] < args.minimum_lr: 
                print "\nNER Learning rate became too small, breaking out of training"
                break
    for i in range(num_domains*num_tasks):
        print 'DATASET', i, 'BEST RESULT: epoch', param['be'+str(i)], 'valid F1', param['vf1'+str(i)],
        if test_data:
            print 'best test F1', param['tf1'+str(i)]

        
def prepare_params_shareLSTM(_args):
    numpy.random.seed(_args.seed)
    random.seed(_args.seed)
    # Do not load data if we want to debug topology
    _args.voc_size = len(_args.global_word_map)
    #_args.idx2word = dict((k, v) for v, k in _args.global_word_map.iteritems())
    print 'sentence size array:', _args.nsentences_arr
    #_args.nsentences = min(nsentences_arr) 
    _args.wemb1_win = _args.win_r - _args.win_l + 1
    _args.lstm_go_backwards = False
    #del _args.global_word_map
    eval_args(_args)
    print 'Chkpt1: datasets label sizes:',
    print [len(label_dict) for label_dict in _args.idx2label_dicts],
    print 'training datasets size:',
    print [len(ds[0]) for ds in _args.trainSet],
    print 'vocabulary size:', _args.voc_size 


def combine_word_dicts(dict1, dict2):
    print 'the size of the two dictionaries are:', len(dict1), len(dict2)
    combine_dict = dict1.copy()
    for k, v in dict2.items():
        if k not in combine_dict:
            combine_dict[k] = len(combine_dict)
    print 'the size of the combined dictionary is:', len(combine_dict)
    return combine_dict 


def load_all_data_multitask(_args):
    # load 3 corpora
    _args.loaddata = load_data_cv
    dataSets = []
    dataset_map = dict()
    lr_arr = []
    _args.num_entity_d0 = 2
    arc_type_dict = dict()
    _args.prediction_files = [_args.drug_gene_prediction_file, _args.drug_var_prediction_file, _args.triple_prediction_file]
    dataSets.append(_args.loaddata(_args.drug_gene_dir, _args.total_fold, _args.dev_fold, _args.test_fold, arc_type_dict, _args.num_entity_d0, dep=_args.graph, content_fname=_args.content_file, dep_fname=_args.dependent_file, add=_args.add))
    dataset_map['drug_gene'] = len(dataset_map)
    lr_arr.append(_args.dg_lr)
    _args.num_entity_d1 = 2
    dataSets.append(_args.loaddata(_args.drug_variant_dir, _args.total_fold, _args.dev_fold, _args.test_fold, arc_type_dict, _args.num_entity_d1, dep=_args.graph, content_fname=_args.content_file, dep_fname=_args.dependent_file, add=_args.add))
    dataset_map['drug_variant'] = len(dataset_map)
    lr_arr.append(_args.dv_lr)
    _args.num_entity_d2 = 3
    dataSets.append(_args.loaddata(_args.drug_gene_variant_dir, _args.total_fold, _args.dev_fold, _args.test_fold, arc_type_dict, _args.num_entity_d2, dep=_args.graph, content_fname=_args.content_file, dep_fname=_args.dependent_file, add=_args.add))
    dataset_map['drug_gene_variant'] = len(dataset_map)
    lr_arr.append(_args.dgv_lr)
    # load embedding
    _args.global_word_map = dict()
    for ds in dataSets:
        _args.global_word_map = combine_word_dicts(_args.global_word_map, ds[-1]['words2idx']) 
    if _args.emb_dir != 'RANDOM':
        print 'started loading embeddings from file', _args.emb_dir        
        M_emb, _ = read_matrix_from_file(_args.emb_dir, _args.global_word_map) 
        print 'global map size:', len(M_emb), len(_args.global_word_map)
        ## load pretrained embeddings
        _args.emb_matrix = theano.shared(M_emb, name='emb_matrix')
        _args.emb_dim = len(M_emb[0])
        _args.wemb1_out_dim = _args.emb_dim
        if _args.fine_tuning :
            print 'fine tuning!!!!!'
            _args.emb_matrix.is_regularizable = True
    print 'loading data dataset map:', dataset_map
    return dataSets, lr_arr, dataset_map

### convert the old word idx to the new one ###
def convert_word_idx(corpus_word, idx2word_old, word2idx_new):
    if type(corpus_word[0]) is int:
        return [word2idx_new[idx2word_old[idx]] for idx in corpus_word]
    else:
        return [convert_word_idx(line, idx2word_old, word2idx_new) for line in corpus_word]


def data_prep_shareLSTM(_args):
    _args.rng = numpy.random.RandomState(_args.seed)
    dataSets, _args.lr_arr, dataset_map = load_all_data_multitask(_args)
    if 'arcs2idx' in dataSets[0][-1]:
        _args.lstm_arc_types = len(dataSets[0][-1]['arcs2idx'])
        print 'lstm arc types =', _args.lstm_arc_types
    ## re-map words in the news cws dataset
    idx2word_dicts = [dict((k, v) for v, k in ds[-1]['words2idx'].iteritems()) for ds in dataSets]
    _args.idx2label_dicts = [dict((k, v) for v, k in ds[-1]['labels2idx'].iteritems()) for ds in dataSets]
    for i, ds in enumerate(dataSets):
        # ds structure: train_set, valid_set, test_set, dicts
        print len(ds[0]), len(ds[0][0][-1]), ds[0][1][-1], ds[0][2][-1]
        print len(ds[1]), len(ds[1][0][-1]), ds[1][1][-1], ds[1][2][-1]
        print len(ds[2]), len(ds[2][0][-1]), ds[2][1][-1], ds[2][2][-1]
        ds[0][0], ds[1][0], ds[2][0] = batch_run_func((ds[0][0], ds[1][0], ds[2][0]), convert_word_idx, idx2word_dicts[i], _args.global_word_map)
        ## convert word, feature and label for array to numpy array
        ds[0], ds[1], ds[2] = batch_run_func((ds[0], ds[1], ds[2]), conv_data_graph, _args.win_l, _args.win_r)
        check_input(ds[0][:3], len(_args.global_word_map))
        check_input(ds[1][:3], len(_args.global_word_map))
    '''Probably want to move part of the below code to the common part.'''
    _args.trainSet = [ds[0] for ds in dataSets]
    _args.devSet = [ds[1] for ds in dataSets]
    _args.testSet = [ds[2] for ds in dataSets]
    _args.nsentences_arr = [len(ds[0]) for ds in _args.trainSet]
    if _args.sample_coef != 0:
        _args.nsentences_arr[0] = int(_args.sample_coef * _args.nsentences_arr[-1])


def check_input(dataset, voc_size):
    for i, (x, y, idx) in enumerate(zip(*dataset)):
        sent_len = len(x)
        for ii in idx:
            try:
            	assert numpy.all(numpy.array(ii) < sent_len) and numpy.all(ii > -1)
            except:
                print 'abnormal index:', ii, 'at instance', i, 'sentence length:', sent_len
        try:
            assert (y == 0 or y == 1)
        except:
            print 'abnormal label:', y
        try:
            assert numpy.all(numpy.array(x) < voc_size)
        except:
            print 'abnormal input:', x

def run_wild_test(_args):
    _args.rng = numpy.random.RandomState(_args.seed)
    _args.loaddata = load_data
    if 'Graph' in _args.circuit:
        _args.graph = True
    if 'Add' in _args.circuit:
        _args.add = True
    if 'Weighted' in _args.circuit:
        _args.weighted = True
    _args.train_set, _args.valid_set, _args.test_set, _args.dicts = _args.loaddata(_args.train_path, _args.valid_path, num_entities=_args.num_entity, dep=_args.graph, train_dep=_args.train_graph, valid_dep=_args.valid_graph, add=_args.add)
    # convert the data from array to numpy arrays
    _args.train_set, _args.valid_set, _args.test_set = batch_run_func((_args.train_set, _args.valid_set, _args.test_set), conv_data_graph, _args.win_l, _args.win_r)
    print 'word dict size:', len(_args.dicts['words2idx'])
    print 'checking training data!'
    check_input(_args.train_set[:3], len(_args.dicts['words2idx']))
    print 'checking test data!'
    check_input(_args.valid_set[:3], len(_args.dicts['words2idx']))
    print 'finish check inputs!!!'
    word2idx = _args.dicts['words2idx']
    prepare_corpus(_args)
    if _args.emb_dir != 'RANDOM':
        print 'started loading embeddings from file', _args.emb_dir        
        M_emb, _ = read_matrix_from_file(_args.emb_dir, word2idx) 
        #M_emb, _ = read_matrix_from_gzip(_args.emb_dir, word2idx) 
        print 'global map size:', len(M_emb) #, count, 'of them are initialized from glove'
        emb_var = theano.shared(M_emb, name='emb_matrix')
        _args.emb_matrix = emb_var
        _args.emb_dim = len(M_emb[0])
        _args.wemb1_out_dim = _args.emb_dim
        if _args.fine_tuning :
            print 'fine tuning!!!!!'
            _args.emb_matrix.is_regularizable = True
    run_wild_prediction(_args) 


def run_single_corpus(_args):
    _args.rng = numpy.random.RandomState(_args.seed)
    _args.loaddata = load_data_cv
    if 'Graph' in _args.circuit:
        _args.graph = True
    if 'Add' in _args.circuit:
        _args.add = True
    if 'Weighted' in _args.circuit:
        _args.weighted = True
    # For the GENIA experiment
    #_args.train_set, _args.valid_set, _args.test_set, _args.dicts = _args.loaddata(_args.data_dir, _args.data_dir, num_entities=_args.num_entity, dep=_args.graph, content_fname=_args.content_file, dep_fname=_args.dependent_file, add=_args.add)
    # For the n-ary experiments
    _args.train_set, _args.valid_set, _args.test_set, _args.dicts = _args.loaddata(_args.data_dir, _args.total_fold, _args.dev_fold, _args.test_fold, num_entities=_args.num_entity, dep=_args.graph, content_fname=_args.content_file, dep_fname=_args.dependent_file, add=_args.add)
    # convert the data from array to numpy arrays
    _args.train_set, _args.valid_set, _args.test_set = batch_run_func((_args.train_set, _args.valid_set, _args.test_set), conv_data_graph, _args.win_l, _args.win_r)
    print 'word dict size:', len(_args.dicts['words2idx'])
    print 'checking training data!'
    check_input(_args.train_set[:3], len(_args.dicts['words2idx']))
    print 'checking test data!'
    check_input(_args.valid_set[:3], len(_args.dicts['words2idx']))
    print 'finish check inputs!!!'
    word2idx = _args.dicts['words2idx']
    prepare_corpus(_args)
    #for k, v in word2idx.iteritems():
    #    print k, v
    if _args.emb_dir != 'RANDOM':
        print 'started loading embeddings from file', _args.emb_dir        
        M_emb, _ = read_matrix_from_file(_args.emb_dir, word2idx) 
        #M_emb, _ = read_matrix_from_gzip(_args.emb_dir, word2idx) 
        print 'global map size:', len(M_emb) #, count, 'of them are initialized from glove'
        emb_var = theano.shared(M_emb, name='emb_matrix')
        _args.emb_matrix = emb_var
        _args.emb_dim = len(M_emb[0])
        _args.wemb1_out_dim = _args.emb_dim
        if _args.fine_tuning :
            print 'fine tuning!!!!!'
            _args.emb_matrix.is_regularizable = True
    run_epochs(_args) 


def run_corpora_multitask(_args):
    if 'Graph' in _args.circuit:
        _args.graph = True
    if 'Add' in _args.circuit:
        _args.add = True
    if 'Weighted' in _args.circuit:
        _args.weighted = True
    data_prep_shareLSTM(_args)
    prepare_params_shareLSTM(_args)
    _args.f_costs_and_updates, _args.f_classifies, cargs = create_multitask_relation_circuit(_args, StackConfig, len(_args.trainSet))
    print "Finished Compiling"
    run_multi_task(_args, cargs, 1, len(_args.trainSet), mode=_args.train_mode) #, test_data=True)


def create_arg_parser(args=None):
    _arg_parser = argparse.ArgumentParser(description='LSTM')
    add_arg.arg_parser = _arg_parser
    add_arg('--setting'        , 'run_single_corpus', help='Choosing between running single corpus or multi-task')
    ## File IO
    # For single task
    add_arg('--data_dir'        , '.')
    # For multitask
    add_arg('--drug_gene_dir'        , '.')
    add_arg('--drug_variant_dir'        , '.')
    add_arg('--drug_gene_variant_dir'         , '.')
    # End for multitask
    # For wild prediction
    add_arg('--train_path'         , '.')
    add_arg('--valid_path'         , '.')
    add_arg('--train_graph'         , '.')
    add_arg('--valid_graph'         , '.')
    add_arg('--content_file'         , 'sentences')
    add_arg('--dependent_file'         , 'graph_arcs')
    add_arg('--parameters_file'         , 'best_parameters')
    add_arg('--prediction_file'         , 'prediction')
    add_arg('--drug_gene_prediction_file'         , '.')
    add_arg('--drug_var_prediction_file'         , '.')
    add_arg('--triple_prediction_file'         , '.')
    add_arg('--num_entity'        , 2)
    add_arg('--total_fold'        , 10)
    add_arg('--dev_fold'        , 0)
    add_arg('--test_fold'        , 1)
    add_arg('--circuit'         , 'LSTMRelation')
    add_arg('--emb_dir'       , '../treelstm/data', help='The initial embedding file name for cws')
    add_arg('--wemb1_dropout_rate'    , 0.0, help='Dropout rate for the input embedding layer')
    add_arg('--lstm_dropout_rate'    , 0.0, help='Dropout rate for the lstm output embedding layer')
    add_arg('--representation'      , 'charpos', help='Use which representation')
    add_arg('--fine_tuning'   , True)
    add_arg('--feature_thresh'     , 0)
    add_arg('--graph'    , False)
    add_arg('--weighted'         , False)
    add_arg('--add'         , False)
    add_arg('--print_prediction'    , True)
    ## Task
    add_arg('--task'    , 'news_cws')
    add_arg('--oovthresh'     , 0    , help="The minimum count (upto and including) OOV threshold for NER") # Maybe 1 ?
    ## Training
    add_arg_to_L(TRAIN_PARAM, '--cost_coef'           , 0.0)
    add_arg_to_L(TRAIN_PARAM, '--sample_coef'           , 0.0)
    add_arg_to_L(TRAIN_PARAM, '--batch_size'           , 1)
    add_arg_to_L(TRAIN_PARAM, '--train_mode'           , 'alternative')
    add_arg_to_L(TRAIN_PARAM, '--lr'           , 0.01)
    add_arg_to_L(TRAIN_PARAM, '--dg_lr'           , 0.005)
    add_arg_to_L(TRAIN_PARAM, '--dv_lr'           , 0.005)
    add_arg_to_L(TRAIN_PARAM, '--dgv_lr'           , 0.005)
    add_arg_to_L(TRAIN_PARAM, '--nepochs'      , 30)
    add_arg_to_L(TRAIN_PARAM, '--optimizer'    , 'sgd', help='sgd or adadelta')
    add_arg_to_L(TRAIN_PARAM, '--seed'         , 1) #int(random.getrandbits(10)))
    add_arg_to_L(TRAIN_PARAM, '--decay'        , True,  help='whether learning rate decay')
    add_arg_to_L(TRAIN_PARAM, '--decay_epochs' , 5)
    add_arg_to_L(TRAIN_PARAM, '--minimum_lr'   , 1e-5)
    ## Topology
    add_arg_to_L(TOPO_PARAM, '--emission_trans_out_dim',    -1)
    add_arg_to_L(TOPO_PARAM, '--crf_viterbi',   False)
    add_arg_to_L(TOPO_PARAM, '--lstm_win_size',               5)
    add_arg_to_L(TOPO_PARAM, '--wemb1_out_dim',               300)
    add_arg_to_L(TOPO_PARAM, '--lstm_out_dim',                150)
    add_arg_to_L(TOPO_PARAM, '--CNN_out_dim',                 500)
    add_arg_to_L(TOPO_PARAM, '--lstm_type_dim',                50)
    add_arg_to_L(TOPO_PARAM, '--MLP_hidden_out_dim',                1000)
    add_arg_to_L(TOPO_PARAM, '--MLP_activation_fn',                'tanh')
    add_arg_to_L(TOPO_PARAM, '--L2Reg_reg_weight',             0.0)
    add_arg_to_L(TOPO_PARAM, '--win_l',                          0)
    add_arg_to_L(TOPO_PARAM, '--win_r',                          0)
    ## DEBUG
    add_arg('--verbose'      , 2)

    return _arg_parser
    
    
if __name__ == "__main__":
    #######################################################################################
    ## PARSE ARGUMENTS, BUILD CIRCUIT, TRAIN, TEST
    #######################################################################################
    TOPO_PARAM = []
    TRAIN_PARAM = []
    _arg_parser = create_arg_parser()
    args = _arg_parser.parse_args()
    #run_wild_test(args)
    #run_corpora_multitask(args)
    eval(args.setting)(args)
