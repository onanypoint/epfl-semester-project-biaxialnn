from tqdm import tqdm, tqdm_notebook
import _pickle as pickle
import configparser
import numpy as np
import signal
import sys
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

quantization = config.getint('DEFAULT', 'measure_quantization')
seq_len = config.getint('DEFAULT', 'seq_len')  *  quantization   

def calculate_loss(model, y_true, y_predict):
    """Calculate loss
    
    Note
    ----
    We expand the first dimension to match the number of dimension of 
    the training samples. We have to do this because theano not react well 
    to ellipses notations.
    
    Parameters
    ----------
    model : utils.model.Model
        The model in use
    y_true : array_like
    y_predict : array_like
    
    Returns
    -------
    float
        The loss value
    """
    y_true = np.expand_dims(y_true, 0)
    y_predict = np.expand_dims(y_predict, 0)
    return model.loss_func(y_true, y_predict).eval()

def validate(model, pieces, repeat = 3):
    """Compute validation loss
    
    Parameters
    ----------
    model : utils.model.Model
        The model to use
    pieces : dict
        Dictrionary containing statematrixes as values
    repeat : {int}, optional
        Average over (the default is 3)
    
    Returns
    -------
    float
        The validation error average over repeat times.
    """
    sub_val = []
    for i in range(repeat):
        xIpt, xOpt = map(np.array, model.data_manager.get_piece_segment(pieces))
        seed_length = int(len(xIpt)/2)
        val = calculate_loss(model, xOpt[seed_length:seed_length+16], model.generate_fun(16, 1, xIpt[:seed_length]))
        sub_val.append(val) 
    return np.mean(sub_val)

def generate_sample(model, pieces, directory, name):
    """Generate a sample and save it to disk

    Note
    ----
    Only use the first note just as in the original model
    
    Parameters
    ----------
    model : utils.model.Model
        The model to use
    pieces : dict
        Dictrionary containing statematrixes as values
    directory : str
        path to parent folder
    name : str
        specific name of the file, will be append after prefix
    """
    xIpt, xOpt = map(np.array, model.data_manager.get_piece_segment(pieces))
    seed_i, seed_o = (xIpt[0], xOpt[0])
    generated_sample = model.generate_fun(seq_len, 1, np.expand_dims(seed_i, axis=0))
    statematrix = np.concatenate((np.expand_dims(seed_o, 0), generated_sample), axis=0)
    s = model.data_manager.s.statematrix_to_stream(statematrix)
    np.save(directory + 'samples/sample_{}.npy'.format(name), statematrix)
    s.write('musicxml', directory + 'samples/sample_{}.xml'.format(name))
    
    pickle.dump(model.learned_config, open(directory + 'weights/params_{}.p'.format(name), 'wb'))

def train_piece(model, pieces, epochs, directory , start=0, validation_split=0.1):
    """Train neural networks
    
    This method is used to train the biaxial neural network. This method will
    train for epochs time and will dump to disk the loss (both error and 
    validation) to disk. 

    Note
    ----
    This method can be stopped gracefully.
    
    Parameters
    ----------
    model : utils.model.Model
        The model to use
    pieces : dict
        Dictrionary containing statematrixes as values
    epochs : int
        Number of epoch to train for
    directory : str
        path to parent folder
    start : int, optional
       at which epochs to start training (the default is 0)
    validation_split : float, optional
        Percentage of validation data to use(the default is 0.1)
    
    Returns
    -------
    (array_like, array_like)
        The loss arrays
    """
    stopflag = [False]
    
    split = int(len(pieces)*(1-validation_split))
    train_pieces = {k: pieces[k] for k in list(pieces.keys())[:split]}
    val_pieces   = {k: pieces[k] for k in list(pieces.keys())[split:]}

    errors = []
    validations = []
    lowest_error = sys.maxsize

    def signal_handler(signame, sf):
        stopflag[0] = True
    
    old_handler = signal.signal(signal.SIGINT, signal_handler)
    
    pbar = tqdm(total=epochs) 
    for i in range(start,start+epochs):
        
        if stopflag[0]:
            break

        error = model.update_fun(*model.data_manager.get_piece_batch(train_pieces))

        if error < lowest_error: 
            lowest_error = error
            pickle.dump(model.learned_config, open(directory + 'weights/params_lowest.p', 'wb'))

        if i % 200 == 0:
            validation_loss = validate(model, val_pieces)
            validations.append(validation_loss)
            errors.append(float(error))
            print("epoch {:5d},  error {:10.4f},  validation {:10.4f}".format(i, float(error), float(validation_loss)))

        if i % 1000 == 0 or (i % 200 == 0 and i < 1000):
            print("epoch {:5d},  generating".format(i))
            generate_sample(model, pieces, directory, str(i))

        pbar.update(1)

    pbar.close()

    signal.signal(signal.SIGINT, old_handler)
    
    print("Finish Training")
    pickle.dump(errors, open(directory + 'loss_training.p', 'wb'))
    pickle.dump(errors, open(directory + 'loss_validation.p', 'wb'))

    return errors, validations