from tqdm import tqdm
import _pickle as pickle
import configparser
import numpy as np
import signal
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

quantization = config.getint('DEFAULT', 'measure_quantization')
seq_len = config.getint('DEFAULT', 'seq_len')  *  quantization   

def calculate_loss(model, y_true, y_predict):
    y_true = np.expand_dims(y_true, 0)
    y_predict = np.expand_dims(y_predict, 0)
    return model.loss_func(y_true, y_predict).eval()

def validate(model, pieces, repeat = 3):
    sub_val = []
    for i in range(repeat):
        xIpt, xOpt = map(np.array, model.data_manager.get_piece_segment(pieces))
        seed_length = int(len(xIpt)/2)
        val = calculate_loss(model, xOpt[seed_length:seed_length+16], model.generate_fun(16, 1, xIpt[:seed_length]))
        sub_val.append(val) 
    return np.mean(sub_val)

def generate_sample(model, pieces, directory, name):
    xIpt, xOpt = map(np.array, model.data_manager.get_piece_segment(pieces))
    seed_i, seed_o = (xIpt[0], xOpt[0])
    generated_sample = model.generate_fun(seq_len, 1, np.expand_dims(seed_i, axis=0))
    statematrix = np.concatenate((np.expand_dims(seed_o, 0), generated_sample), axis=0)
    s = model.data_manager.s.statematrix_to_stream(statematrix)
    np.save(directory + 'samples/sample_{}.npy'.format(name), statematrix)
    s.write('musicxml', directory + 'samples/sample_{}.xml'.format(name))
    
    pickle.dump(model.learned_config, open(directory + 'weights/params_{}.p'.format(name), 'wb'))

def train_piece(model, pieces, epochs, directory , start=0, validation_split=0.1):
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
        pbar.update(1)

        if stopflag[0]:
            break

        error = model.update_fun(*model.data_manager.get_piece_batch(train_pieces))

        if error < lowest_error: 
            lowest_error = error
            pickle.dump(errors, open(directory + 'weights/params_lowest.p', 'wb'))

        if i % 200 == 0:
            validation_loss = validate(model, val_pieces)
            validations.append(validation_loss)
            errors.append(float(error))
            print("epoch {:5d},  error {:10.4f},  validation {:10.4f}".format(i, float(error), float(validation_loss)))

        if i % 1000 == 0 or (i % 200 == 0 and i < 1000):
            print("epoch {:5d},  generating".format(i))
            generate_sample(model, pieces, directory, str(i))

    pbar.close()

    signal.signal(signal.SIGINT, old_handler)
    
    print("Finish Training")
    pickle.dump(errors, open(directory + 'loss_training.p', 'wb'))
    pickle.dump(errors, open(directory + 'loss_validation.p', 'wb'))

    return errors, validations