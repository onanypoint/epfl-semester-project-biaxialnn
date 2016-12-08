import random
import numpy as np
import configparser
import os
from tqdm import tqdm

config = configparser.ConfigParser()
config.read('config.ini')

quantization = config.getint('DEFAULT', 'measure_quantization')
batch_size = config.getint('DEFAULT', 'batch_size')     
seq_len = config.getint('DEFAULT', 'seq_len')  *  quantization       
division_len  = config.getint('DEFAULT', 'division_len') 

class DataManager(object):
    def __init__(self, feature_builder, state_matrix_builder):        
        self.f = feature_builder
        self.s = state_matrix_builder

    def pieces_to_statematrix(self, corpus, count = None):
        pieces = {}

        if count:
            count = min(len(corpus), int(count))
            corpus = random.sample(set(corpus), int(count))

        pbar = tqdm(total=len(corpus))
        for i, score_path in enumerate(corpus):
            pbar.update(1)
            try:
                stream = m21.converter.parse(score_path)
                processed = self.s.preprocess_stream(stream)
                outMatrix = self.s.stream_to_statematrix(processed)
            except Exception as e:
                print("Error", score_path)
                continue
                
            if len(outMatrix) < seq_len:
                continue

            name = os.path.basename(score_path)
            pieces[name] = outMatrix
        
        pbar.close()

        return pieces

    def get_piece_segment(self, pieces):
        # The loop 'assures' the method to return a sequence.
        for i in range(100):
            try:
                piece_output = random.choice(list(pieces.values()))
                start = random.randrange(0,len(piece_output)-seq_len,division_len)
                if verbose: print("Range is {} {} {} -> {}".format(0,len(piece_output)-seq_len,division_len, start))
                break
            except:
               pass

        if i == 100: raise IllegalArgumentError(ValueError)

        seg_out = piece_output[start:start+seq_len]
        seg_in = self.f.note_state_matrix_to_input_form(seg_out)

        return seg_in, seg_out

    def get_piece_batch(self, pieces):
        i,o = zip(*[self.get_piece_segment(pieces) for _ in range(batch_size)])
        return np.array(i), np.array(o)