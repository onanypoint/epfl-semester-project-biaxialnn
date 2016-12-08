import random
import numpy as np
import configparser
import os
import music21 as m21
from tqdm import tqdm

config = configparser.ConfigParser()
config.read('config.ini')

quantization = config.getint('DEFAULT', 'measure_quantization')
batch_size = config.getint('DEFAULT', 'batch_size')     
seq_len = config.getint('DEFAULT', 'seq_len')  *  quantization       
division_len  = config.getint('DEFAULT', 'division_len') 

class DataManager(object):
    """Wrapper around a Feature builder and a StateMatrix Builder
    
    Enable easy access to both the Feature builder and the state matrix 
    builder. Usefull when defining a full pipeline from statematrix generation
    to model creatin to generation. It simplify the interaction with the sizes
    of both feature vectors and output vectors.
    
    Parameters
    ----------
    quantization : int
        In how many "beat" a measure is divided
    batch_size : int
        How many sequences a training batch will be composed
    seq_len : int
        Length of a sequence
    division_len : int
        Interval minimum between each sequence
    """

    def __init__(self, feature_builder, state_matrix_builder):        
        self.f = feature_builder
        self.s = state_matrix_builder

    def pieces_to_statematrix(self, corpus, count = None):
        """Process scores to statematrix
        
        Use to process a scores (in xml, mid, etc) to a statematrix format. 
        Will use the basename of the scores (name of the file) as key in 
        the dictionary.
        
        Parameters
        ----------
        corpus : array_like
            List of the scores path
        count : int, optional
            Number of scores to transform (the default is None)
        
        Returns
        -------
        dict
            A dictionary of 'file name' : statematrix
        """
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
                print("Error", score_path, e)
                continue
                
            if len(outMatrix) < seq_len:
                continue

            name = os.path.basename(score_path)
            pieces[name] = outMatrix
        
        pbar.close()

        return pieces

    def get_piece_segment(self, pieces):
        """Return input and ouput representation
        
        Method used to get randomly a sequence (in statematrix form) of 
        seq_len length. It will loop until a sequence long enough can be 
        found. If after 100 tries still no success will through an error.
        
        Parameters
        ----------
        pieces : dict (name:statematrix)
            The dictionary of score in the statematrix form
        verbose : bool, optional
            Print selected range to screen. (the default is False)
        
        Returns
        -------
        seg_in: numpy.array
            Input representation (based on feature builder)
        seg_out: numpy.array
            Ouput representation (based on statematrix given by the dict)

        Raises
        ------
        IllegalArgumentError
            Because no valid sequence could be found

        """

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
        """Return input and ouput representation in batch form.
        
        Return the input and output representation based on this data manager
        in a batched form. It is used during the training phase. It enable 
        the "concatenation" of multiple sequences into a single training sample.
        
        Parameters
        ----------
        pieces : dict (name:statematrix)
             The dictionary of score in the statematrix form
        
        Returns
        -------
        tuple(numpy.array, numpy.array)
            The input representation (based on the `FeatureBuilder`) and 
            the ouput representation (based on the `StateMatrixBuilder`)

        Examples
        --------
        We will use a simple feature builder that return a feature vector
        of length 80 and a statematrix builder which output is 2 "wide". Also
        the batch will be of 3.

        >>> pieces = {'foo': [[[0,0],[1,1],[1,0] ... ] ... ], 'bar': ...}
        >>> len(pieces)
        10
        >>> pieces['foo'].shape
        (180, 87, 2)
        >>> i,o = get_piece_batch(pieces)
        >>> i.shape
        (3, 180, 87, 80)
        >>> o.shape
        (3, 180, 87, 2)

        """
        i,o = zip(*[self.get_piece_segment(pieces) for _ in range(batch_size)])
        return np.array(i), np.array(o)