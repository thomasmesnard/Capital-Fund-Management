import logging
import numpy
import sys
import importlib
import csv

from contextlib import closing

import theano
from theano import tensor
from theano.tensor.shared_randomstreams import RandomStreams

from fuel.datasets import Dataset, IndexableDataset
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme

from blocks.serialization import load_parameter_values, secure_dump, BRICK_DELIMITER
from blocks.extensions import Printing, SimpleExtension, FinishAfter
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.extensions.saveload import Checkpoint, Load
from blocks.graph import ComputationGraph
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.algorithms import GradientDescent, StepRule, CompositeRule

try:
    from blocks.extras.extensions.plot import Plot
    plot_avail = True
except ImportError:
    plot_avail = False

import datastream
from paramsaveload import SaveLoadParams

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

sys.setrecursionlimit(500000)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print >> sys.stderr, 'Usage: %s config' % sys.argv[0]
        sys.exit(1)
    model_name = sys.argv[1]
    config = importlib.import_module('%s' % model_name)

    # Build datastream
    _,train_stream, valid_stream = datastream.setup_datastream('training_input.csv',
        'challenge_output_data_training_file_prediction_of_transaction_volumes_in_financial_markets.csv',
        config.batch_size,config.proportion_train)

    # Build model
    m = config.Model()

    # Train the model
    dump_path = 'model_data/%s' % (model_name)

    # Define the model
    model = Model(m.sgd_cost)

    algorithm = GradientDescent(cost=m.sgd_cost,
                                step_rule=config.step_rule,
                                parameters=model.parameters)


    extensions = [
            TrainingDataMonitoring(
                [v for p in m.monitor_vars for v in p],
                prefix='train', every_n_batches=config.print_freq),
            DataStreamMonitoring(
                [v for l in m.monitor_vars for v in l],
                valid_stream,
                prefix='valid',
                every_n_batches=config.valid_freq),
            Printing(every_n_batches=config.print_freq, after_epoch=False),
    ]
    extensions.append(FinishAfter(after_n_epochs=200)) #after_n_batches
    if plot_avail:
        plot_channels = [['valid_' + v.name for v in p]for p in m.monitor_vars]+[['train_' + v.name for v in p] for p in m.monitor_vars] 
        extensions.append(
            Plot(document='CFM_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s'
            %(config.name,config.couches,config.hidden_dim,
              config.activation_function_name,config.batch_size,config.w_noise_std,
              config.i_dropout, config.algo,config.learning_rate_value,
              config.momentum_value,config.decay_rate_value,config.StepClipping_value),
                 channels=plot_channels,
                 every_n_batches=config.print_freq,
                 after_epoch=False)
        )

    if config.save_freq is not None and dump_path is not None:
        extensions.append(
            SaveLoadParams(path=dump_path+'CFM_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s.pkl'
            %(config.name,config.couches,config.hidden_dim,
              config.activation_function_name,config.batch_size,config.w_noise_std,
              config.i_dropout, config.algo,config.learning_rate_value,
              config.momentum_value,config.decay_rate_value,config.StepClipping_value),
                           model=model,
                           before_training=True,
                           after_training=True,
                           after_epoch=False,
                           every_n_batches=config.save_freq)
        )

    main_loop = MainLoop(
        model=model,
        data_stream=train_stream,
        algorithm=algorithm,
        extensions=extensions
    )

    main_loop.run()
    #main_loop.profile.report()


    test = numpy.genfromtxt('testing_input.csv',skip_header=1,delimiter=',',filling_values=0)[:,:,None]
    test_input = test[:,3:,:].astype(numpy.float32)
    train_input_mean = 1470614.1
    train_input_std = 3256577.0
    train_input_mean_reshape = (train_input_mean*numpy.ones(test_input.shape[1])).astype(numpy.float32)
    train_input_std_reshape = (train_input_std*numpy.ones(test_input.shape[1])).astype(numpy.float32)
    test_input = (test_input - train_input_mean_reshape[None,:,None])/train_input_std_reshape[None,:,None]
    
    test_product = test[:,2,0]
    ref = {int(k):i for i, k in enumerate(open('listid.txt'))}
    test_product = numpy.array([ref[test_product[i]] for i in range(len(test_product))])
    
    test_id = test[:,0]   
     
    ds_test = IndexableDataset({'id':test_id,'input':test_input,'product':test_product}) 
    scheme_test = SequentialScheme(batch_size=10000,examples= ds_test.num_examples)
    stream_test = DataStream(ds_test,iteration_scheme = scheme_test)

    pred_test = ComputationGraph([m.pred]).get_theano_function()
    with open('results_deep_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s.csv'
            %(config.name,config.couches,config.hidden_dim,
              config.activation_function_name,config.batch_size,config.w_noise_std,
              config.i_dropout, config.algo,config.learning_rate_value,
              config.momentum_value,config.decay_rate_value,config.StepClipping_value)
            , 'wb') as csvfile:
        print "Writing results on test set..."
        csv_func = csv.writer(csvfile, delimiter=',')
        csv_func.writerow(['ID','TARGET'])
        for d in stream_test.get_epoch_iterator(as_dict=True):
            print d['id'][0]
            output,  =  pred_test(**{x: d[x] for x in ['input','product']})
            for i in range(output.shape[0]):
                csv_func.writerow([int(test_id[i]),float(output[i])])