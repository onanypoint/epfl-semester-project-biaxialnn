import theano, theano.tensor as T
import numpy as np
import theano_lstm
from theano_lstm import LSTM, StackedCells, Layer, create_optimization_updates, MultiDropout

def has_hidden(layer):
    """Whetever the layer has a trainable initial hidden state.
    
    Parameters
    ----------
    layer : theano_lstm.Layer
    
    Returns
    -------
    boolean
        Whetever the layer as hidden state
    """
    return hasattr(layer, 'initial_hidden_state')

def matrixify(vector, n):
    """Transform a vector or a matrix into a matrix

    Repeate the vector n times in the 0 axis.
    
    Parameters
    ----------
    vector : array_like
        Vector (or matrix) to be repreated
    n : int
        Number of repetition
    
    Returns
    -------
    theano.tensor
        A theano tensor corresponding to the n times repeated vector

    Example
    -------

    >>>> vect1.shape
    (10,)
    >>>> matrixify(vect1, 5).shape.eval()
    (5,10)
    >>>> vect2.shape
    (10,15)
    >>>> matrixify(vect2, 5).shape.eval()
    (5,10,15)

    """
    return T.repeat(T.shape_padleft(vector), n, axis=0)

def initial_state(layer, dimensions = None):
    """Initialise hidden state
    
    Used to initialize any layer with reccurrent relation with initial hidden
    state if needed. None is returned if the network is going to return 
    something nd thus there is no need to send anything to the next step of the 
    recurrence.
    
    Parameters
    ----------
    layer : theano_lstm.Layer
        The layer of interest
    dimensions : int, optional
        The dimension of the needed output (the default is None). It is used 
        during the creation of the hidden state for the scan function.
    
    Returns
    -------
    theano.tensor
       The initial hidden state of the layer if no dimension are given, 
       a matrixified version of the initial hidden state otherwise or None 
       if layer has no hidden layer.
    """
    if dimensions is None:
        return layer.initial_hidden_state if has_hidden(layer) else None
    else:
        return matrixify(layer.initial_hidden_state, dimensions) if has_hidden(layer) else None

def initial_state_with_taps(layer, dimensions = None):
    """Optionally wrap tensor variable into a dict with taps=[-1]
    
    Used during creation of a scan function. Indeed the function need to know
    in which order to process the input. More information on the "taps" 
    parameter can be found here:

    - http://deeplearning.net/software/theano/library/scan.html
    
    Parameters
    ----------
    layer : theano_lstm.layer
    dimensions : int, optional
        The dimension of the needed output (the default is None). It is used 
        during the creation of the hidden state for the scan function.
    
    Returns
    -------
    dict or None
        A dictionary with the relevant state and taps
    """
    state = initial_state(layer, dimensions)
    if state is not None:
        return dict(initial=state, taps=[-1])
    else:
        return None

class PassthroughLayer(Layer):
    """Empty layer

    Used to get the final output of LSTM layers.

    Note
    ----
    Should be added to a theano_lstm.StackedCells after a recurrent layer.
    
    """
    
    def __init__(self):
        self.is_recursive = False
    
    def create_variables(self):
        pass
    
    def activate(self, x):
        return x
    
    @property
    def params(self):
        return []
    
    @params.setter
    def params(self, param_list):
        pass       

def get_last_layer(result):
    """Get last element (layer)
    
    Used to return the output of the last layer of theano_lstm.StackedCells
    if it has multiple layer and the output directly if it is a simple layer.
    
    Parameters
    ----------
    result : array_like
    
    Returns
    -------
    array_like
        final output of the stackecells or the given layer
    """
    if isinstance(result, list):
        return result[-1]
    else:
        return result

def ensure_list(result):
    """Make sure result is a list
        
    Parameters
    ----------
    result : object
    
    Returns
    -------
    array_like
        The input list or the object wrapped in a list
    """
    if isinstance(result, list):
        return result
    else:
        return [result]

class OutputFormToInputFormOp(theano.Op):
    """Theano operation to go from output space to feature space
    
    More information can be found here:

    - http://deeplearning.net/software/theano/extending/extending_theano.html
    
    Attributes
    ----------
    __props__ : tuple
        Properties attribute
    """

    __props__ = ()

    def __init__(self, data_manager, *args):
        self.d = data_manager

    def make_node(self, state, time):
        """Creates an Apply node representing the application of the op on 
        the inputs provided.
        
        Parameters
        ----------
        state : array_like
            The state to transform into feature space
        time : int
            The current time being processed
        
        Returns
        -------
        theano.Apply
            [description]
        """
        state = T.as_tensor_variable(state)
        time = T.as_tensor_variable(time)
        return theano.Apply(self, [state, time], [T.bmatrix()])
    
    def perform(self, node, inputs_storage, output_storage):
        """Peform the transformation from output to feature space.
        
        Defines the Python implementation of the op. It is in charge of doing 
        the processing to go from output space (statematrix) to feature space.
        
        Parameters
        ----------
        node : 
            Reference to an Apply node which was previously obtained via 
            the Op‘s make_node() method.
        inputs_storage : array_like
            A list of references to data which can be operated on using 
            non-symbolic statements
        output_storage : array_like
            A list of storage cells where the output is to be stored
        """
        state, time = inputs_storage
        output_storage[0][0] = np.array(self.d.f.note_state_single_to_input_form(state, time), dtype='int8')

class Model(object):

    def __init__(self, data_manager, t_layer_sizes, p_layer_sizes, dropout=0):
        print('{:25}'.format("Initializing Model"), end='', flush=True)
        self.t_layer_sizes = t_layer_sizes
        self.p_layer_sizes = p_layer_sizes
        self.dropout = dropout

        self.data_manager = data_manager
        self.t_input_size = self.data_manager.f.feature_count
        self.output_size = self.data_manager.s.information_count

        self.time_model = StackedCells(self.t_input_size, celltype=LSTM, layers = t_layer_sizes)
        self.time_model.layers.append(PassthroughLayer())

        p_input_size = t_layer_sizes[-1] + self.output_size
        self.pitch_model = StackedCells( p_input_size, celltype=LSTM, layers = p_layer_sizes)
        self.pitch_model.layers.append(Layer(p_layer_sizes[-1], self.output_size, activation = T.nnet.sigmoid))

        self.conservativity = T.fscalar()
        self.srng = T.shared_randomstreams.RandomStreams(np.random.randint(0, 1024))

        self.epsilon = np.spacing(np.float32(1.0))

        print("Done")

    @property
    def params(self):
        return self.time_model.params + self.pitch_model.params
    
    @params.setter
    def params(self, param_list):
        ntimeparams = len(self.time_model.params)
        self.time_model.params = param_list[:ntimeparams]
        self.pitch_model.params = param_list[ntimeparams:]

    @property
    def learned_config(self):
        return [self.time_model.params, self.pitch_model.params, [l.initial_hidden_state for mod in (self.time_model, self.pitch_model) for l in mod.layers if has_hidden(l)]]

    @learned_config.setter
    def learned_config(self, learned_list):
        self.time_model.params = learned_list[0]
        self.pitch_model.params = learned_list[1]
        for l, val in zip((l for mod in (self.time_model, self.pitch_model) for l in mod.layers if has_hidden(l)), learned_list[2]):
            l.initial_hidden_state.set_value(val.get_value())
    
    def setup(self):
        self.setup_train()
        self.setup_generate()

    def loss_func(self, y_true, y_predict):
        active_notes = T.shape_padright(y_true[:,:,:,0])
        mask = T.concatenate([T.ones_like(active_notes), active_notes, T.repeat(T.ones_like(active_notes), self.output_size-2, -1)], axis=-1)
        loglikelihoods = mask * T.log( 2*y_predict*y_true - y_predict - y_true + 1 + self.epsilon )
        return T.neg(T.sum(loglikelihoods))

    def setup_train(self):
        print('{:25}'.format("Setup Train"), end='', flush=True)
       
        self.input_mat = T.btensor4()
        self.output_mat = T.btensor4()

        def step_time(in_data, *other):
            other = list(other)
            split = -len(self.t_layer_sizes) if self.dropout else len(other)
            hiddens = other[:split]
            masks = [None] + other[split:] if self.dropout else []
            new_states = self.time_model.forward(in_data, prev_hiddens=hiddens, dropout=masks)
            return new_states

        def step_note(in_data, *other):
            other = list(other)
            split = -len(self.p_layer_sizes) if self.dropout else len(other)
            hiddens = other[:split]
            masks = [None] + other[split:] if self.dropout else []
            new_states = self.pitch_model.forward(in_data, prev_hiddens=hiddens, dropout=masks)
            return new_states

        def get_dropout(layers, num_time_parallel = 1):
            if self.dropout > 0:
                return theano_lstm.MultiDropout( [(num_time_parallel, shape) for shape in layers], self.dropout)
            else:
                return []

        # TIME PASS
        input_slice = self.input_mat[:,0:-1]
        n_batch, n_time, n_note, n_ipn = input_slice.shape
        time_inputs = input_slice.transpose((1,0,2,3)).reshape((n_time,n_batch*n_note,n_ipn))
        
        time_masks = get_dropout(self.t_layer_sizes, time_inputs.shape[1])
        time_outputs_info = [initial_state_with_taps(layer, time_inputs.shape[1]) for layer in self.time_model.layers]
        time_result, _ = theano.scan(fn=step_time, sequences=[time_inputs], non_sequences=time_masks, outputs_info=time_outputs_info)
        self.time_thoughts = time_result

        last_layer = get_last_layer(time_result)
        n_hidden = last_layer.shape[2]
        time_final = get_last_layer(time_result).reshape((n_time,n_batch,n_note,n_hidden)).transpose((2,1,0,3)).reshape((n_note,n_batch*n_time,n_hidden))
        
        # PITCH PASS
        start_note_values = T.alloc(np.array(0,dtype=np.int8), 1, time_final.shape[1], self.output_size )
        correct_choices = self.output_mat[:,1:,0:-1,:].transpose((2,0,1,3)).reshape((n_note-1, n_batch*n_time, self.output_size))
        note_choices_inputs = T.concatenate([start_note_values, correct_choices], axis=0)
        
        note_inputs = T.concatenate([time_final, note_choices_inputs] ,axis=2 )
        
        note_masks = get_dropout(self.p_layer_sizes, note_inputs.shape[1])
        note_outputs_info = [initial_state_with_taps(layer, note_inputs.shape[1]) for layer in self.pitch_model.layers]
        note_result, _ = theano.scan(fn=step_note, sequences=[note_inputs], non_sequences=note_masks, outputs_info=note_outputs_info)
        

        self.note_thoughts = note_result

        note_final = get_last_layer(note_result).reshape((n_note,n_batch,n_time, self.output_size)).transpose(1,2,0,3)
        
        self.cost = self.loss_func(self.output_mat[:,1:], note_final)

        updates, _, _, _, _ = create_optimization_updates(self.cost, self.params, method="adadelta")
        self.update_fun = theano.function(
            inputs=[self.input_mat, self.output_mat],
            outputs=self.cost,
            updates=updates,
            allow_input_downcast=True)


        print("Done")

   
    def _predict_step_note(self, in_data_from_time, *states):
            hiddens = list(states[:-1])
            in_data_from_prev = states[-1]
            in_data = T.concatenate([in_data_from_time, in_data_from_prev])
            
            if self.dropout > 0:
                masks = [1 - self.dropout for layer in self.pitch_model.layers]
                masks[0] = None
            else:
                masks = []

            new_states = self.pitch_model.forward(in_data, prev_hiddens=hiddens, dropout=masks)
            probabilities = get_last_layer(new_states)
                
            shouldPlay = self.srng.uniform() < (probabilities[0] ** self.conservativity)
            shouldArtic = shouldPlay * (self.srng.uniform() < probabilities[1])
     
            chosen = T.stack([T.cast(shouldPlay, 'int8'), T.cast(shouldArtic, 'int8')])
            return ensure_list(new_states) + [chosen]

    def setup_generate(self):
        print('{:25}'.format("Setup Generate"), end='', flush=True)
        
        self.generate_seed_input = T.btensor3()
        self.steps_to_simulate = T.iscalar()

        def step_time_seed(in_data, *hiddens):
            if self.dropout > 0:
                time_masks = [1 - self.dropout for layer in self.time_model.layers]
                time_masks[0] = None
            else:
                time_masks = []

            new_states = self.time_model.forward(in_data, prev_hiddens=hiddens, dropout=time_masks)
            return new_states

        time_inputs = self.generate_seed_input[0:-1]
        n_time, n_note, n_ipn = time_inputs.shape

        time_outputs_info_seed = [initial_state_with_taps(layer, n_note) for layer in self.time_model.layers]
        time_result, _ = theano.scan(fn=step_time_seed, sequences=[time_inputs], outputs_info=time_outputs_info_seed)    

        last_layer = get_last_layer(time_result)
        n_hidden = last_layer.shape[2]
        

        def step_time(*states):
            hiddens = list(states[:-2])
            in_data = states[-2]
            time = states[-1]

            if self.dropout > 0:
                masks = [1 - self.dropout for layer in self.time_model.layers]
                masks[0] = None
            else:
                masks = []

            new_states = self.time_model.forward(in_data, prev_hiddens=hiddens, dropout=masks)
            
            time_final = get_last_layer(new_states)
            
            start_note_values = theano.tensor.alloc(np.array(0,dtype=np.int8), self.output_size)
            note_outputs_info = ([ initial_state_with_taps(layer) for layer in self.pitch_model.layers ] +
                                 [ dict(initial=start_note_values, taps=[-1]) ])
            
            notes_result, updates = theano.scan(fn=self._predict_step_note, sequences=[time_final], outputs_info=note_outputs_info)
            output = get_last_layer(notes_result)
            next_input = OutputFormToInputFormOp(self.data_manager)(output, time + 1) 

            return (ensure_list(new_states) + [ next_input, time + 1, output ]), updates

        time_outputs_info = (  time_outputs_info_seed +
                             [ dict(initial=self.generate_seed_input[-1], taps=[-1]),
                               dict(initial=n_time, taps=[-1]),
                               None ])

        time_result, updates = theano.scan( fn=step_time, 
                                            outputs_info=time_outputs_info, 
                                            n_steps=self.steps_to_simulate ) 

        self.predicted_output = time_result[-1]

        self.generate_fun = theano.function(
            inputs=[self.steps_to_simulate, self.conservativity, self.generate_seed_input],
            outputs=self.predicted_output,
            updates=updates,
            allow_input_downcast=True,
            on_unused_input='warn')

        print("Done")