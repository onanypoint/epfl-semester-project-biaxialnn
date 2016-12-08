import itertools
import configparser
import numpy as np

config = configparser.ConfigParser()
config.read('config.ini')

lowerBound = config.getint('DEFAULT', 'pitch_lowerBound')
upperBound = config.getint('DEFAULT', 'pitch_upperBound')

class FeatureBuilder(object):
    """Used to go from output space to feature space
    
    Attributes
    ----------
    lowerBound : int
        The lowest pitch that will be used. Based on the midi scale.
    upperBound : int
        The highest pitch that will be used. Based on the midi scale.
    """
    
    @property
    def feature_count(self):
        """Return the number of features
        
        It is used during the model creation. The model can not infer the size 
        of each input before running time.

        Returns
        -------
        number: int
            Number of feature for each sample
        
        Raises
        ------
        NotImplementedError
            Has to be implemented in children class
        """
        raise NotImplementedError()
    
    def build_auxillary_info(self, note, state):
        """Method used to add information beside play and articulations
        
        Parameters
        ----------
        note : int
            [description]
        state : array_like
            An array containing the information about the full timestep. 
            It should be based on a statematrix (the ouput of the 
            `StateMatrixBuilder`.
        
        Raises
        ------
        NotImplementedError
            Has to be implemented in children class
        """
        raise NotImplementedError()
    
    def note_input_form(self, note, state, time):
        """Return the representation in feature space of a specific pitch
        
        Preprocess the state (from a statematrix) into feature space. Used to
        augmente the data space into a more usefull format than the simpler
        format found in the statematrix format.
        
        Parameters
        ----------
        note : int
            The pitch id (midi pitch - lowerbound)
        state : array_like
            An array containing the information about the full timestep. 
            It should be based on a statematrix (the ouput of the 
            `StateMatrixBuilder`.
        time: int
            The time (beat number) of the timestep
       
        Returns
        -------
        array_like
            The feature representation of the specified not at the 
            specified time based on the actual state.
        """

        def _get_or_default(l, i, d):
            """Return element at index i or d
            
            Parameters
            ----------
            l : array_like
            i : int
                Index of element of interest
            d : array_like
                Default output if index does not exists
            
            Returns
            -------
            array_like
                The element at index _i_ or the default

            """
            try:
                return l[i]
            except IndexError:
                return d

        def _build_context(state):
            """Build the context based on the state of the timestep
            
            The context is a representation of the current state of the timestep
            pitch wise. It gives information about which note are playing at 
            that time and how many of each.

            Note
            ----
            The full pitch spectrum is collapsed into the chromatic scale 
            of 12 pitches.
            
            Parameters
            ----------
            state : array_like
                An array containing the information about the full timestep. 
                It should be based on a statematrix (the ouput of the 
                `StateMatrixBuilder`.
            
            Returns
            -------
            array_like
                the context of the given state

            Example
            -------
            Let's say that at the timestep of interest the note A5 A3 and 
            B4 are played the returned context will be.
            
            >>> _build_context(state)
            [0,0,0,0,0,0,0,0,0,0,0,0]            
            
            """
            context = [0]*12
            for note, notestate in enumerate(state):
                if notestate[0] == 1:
                    pitchclass = (note + lowerBound) % 12
                    context[pitchclass] += 1
            return context

        def _build_beat(time):
            """Build time representation
            
            Build the time representation used as features. It is hardcoded 
            to be represented in 4 feature (quantization = 16). It returns 
            the binary representation of the time modula 16 (except that it 
            returns -1 instead of 0. 
            
            Parameters
            ----------
            time : int
                the current time in the sequence
            
            Example
            -------
            >>> _build_beat(10)
            [ 1,-1, 1,-1]
            >>> _build_beat(113)
            [-1, 1,-1,-1]
            
            """
            return [2*x-1 for x in [time%2, (time//2)%2, (time//4)%2, (time//8)%2]]

        position = note
        part_position = [position]

        beat = _build_beat(time)
        context = _build_context(state)

        pitchclass = (note + lowerBound) % 12
        part_pitchclass = [int(i == pitchclass) for i in range(12)]
        
        # part_prev_vicinity is the representation of the surrounding of note. 
        # Look one octave in each direction, keep the representation into 
        # play / articulations for each surrounding note.
        part_prev_vicinity = list(itertools.chain.from_iterable((_get_or_default(state[:,0:2], note+i, [0,0]) for i in range(-12, 13))))
        part_context = context[pitchclass:] + context[:pitchclass]
        part_aux = self.build_auxillary_info(note, state)

        return np.concatenate([part_position, part_pitchclass, part_prev_vicinity, part_context, part_aux, beat, [0]], axis=0)
    
    def note_state_single_to_input_form(self, state, time):
        """Return the feature representation of the state
        
        Given the state and the time, return the feature representation for
        each pitch.
        
        Parameters
        ----------
        state : array_like
            An array containing the information about the full timestep. 
            It should be based on a statematrix (the ouput of the 
            `StateMatrixBuilder`.
        time : int
            time of the timestep

        Example
        -------
        Let's say that the statematrix representation is 2 "wide" and 
        the feature representation 80. In this method we only look at one 
        timestep.

        >>> state.shape
        (87, 2)
        >>> f = note_state_single_to_input_form(state, 10)
        >>> f.shape
        (87, 80)

        """
        return [self.note_input_form(note, state, time) for note in range(len(state))]

    def note_state_matrix_to_input_form(self, statematrix):
        """Process statematrix from output space to feature space.
        
        Note
        ----
        Assume that the time state at 0.

        Parameters
        ----------
        statematrix : array_like
            The statematrix to process in feature space.
        
        Returns
        -------
        array_like
            The feature represenation of the statematrix.
        """
        inputform = [ self.note_state_single_to_input_form(state,time) for time,state in enumerate(statematrix) ]
        return inputform

class FeatureBuilderSimple(FeatureBuilder):
    """No auxillary information

    The feature space is the 80 basic feature infered from
    the play/articulations status of each note at each timestep.
    
    """
    
    @property
    def feature_count(self):
        return 80

    def build_auxillary_info(self, note, state):
        return []









