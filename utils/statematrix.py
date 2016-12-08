import math
import music21 as m21
import numpy as np
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

lowerBound = config['DEFAULT'].getint('pitch_lowerBound')
upperBound = config.getint('DEFAULT', 'pitch_upperBound')
quantization = config.getint('DEFAULT', 'measure_quantization')

assert(quantization % 4 == 0)

class StateMatrixBuilder(object):
    """Used to go from stream to output space or statematrix space.
    
    Attributes
    ----------
    lowerBound : int
        The lowest pitch that will be used. Based on the midi scale.
    upperBound : int
        The highest pitch that will be used. Based on the midi scale.        
    quantization : int
        The quantization scale used to divide the time. It is based on quarter
        length.
    """

    def preprocess_stream(self, stream):
        """
        Preprocess music21.stream object.

        Preprocess music21.stream object into a more suitable stream. 
        For now, only keep the first voice it finds. This is done to avoid 
        the cumbersome work of knowing in which part/voice the generate note
        belongs.
        
        Note
        ----
        Only keep the first part or voice it finds.
        
        Parameters
        ----------
        stream : music21.stream
            The stream to process.
        Returns
        -------
        music21.stream
            The preprocessed stream.
        """

        def verify_time_signature(stream, ts_numerator = [2,4]):
            """Whethever the timesignature is valid.

            Parameters
            ----------
            stream : music21.stream
                The stream to preprocess
            ts_numerator : {list of int}, optional
                The "autorized" timesignature numerator (the default is [2,4]).
            
            Returns
            -------
            boolean
                Return if the stream is considered valid based on time
                signature.
            """
            time_signature = stream.getElementsByClass("TimeSignature")
            ts_list = [t.numerator not in ts_numerator for t in time_signature]
            return sum(ts_list) > 0

        for part in stream.parts:
            if verify_time_signature(part):
                continue

            if not len(part.voices):
                return part

            for v in part.voices:
                return v

        return

    @property
    def information_count(self):
        """Return the number of features
        
        It is used during the model creation. The model can not infer the size 
        of output before running time.

        Returns
        -------
        number: int
            Number of output for each sample
        
        Raises
        ------
        NotImplementedError
            Has to be implemented in children class
        """
        raise NotImplementedError()

    def _get_note_tie(self, note):
        """Get the tie of the note or None
        
        Parameters
        ----------
        note : music21.Note            
            The note object to be checked for tie. Tie can either be 
            start, stop, or continue.
        
        Returns
        -------
        string
            the music21.tie.type of the note or None
        """
        if note.tie:
            return note.tie.type
        return None

    def _extract_chord_data(self, fx, c):
        """Extract Chord information
        
        Parameters
        ----------
        fx : function
            Function used to extract component information.
        c : m21.chord.Chord
            The chord object to decompose into its components.
        
        Returns
        -------
        array_like
            List of the inner component of the chord.
        """
        values = []
        value = None
        try:
            value = fx(c)
        except AttributeError:
            pass 
        
        if value is not None:
            values.append(value)

        if values == []:
            for n in c:
                value = None
                try:
                    value = fx(n)
                except AttributeError:
                    break
                if value is not None:
                    values.append(value)
        
        return values

    def stream_to_statematrix(self, stream):
        """Process a music21.stream into a statematrix representation.
        
        Parameters
        ----------
        stream : music21.stream
        
        Raises
        ------
        NotImplementedError
            Has to be implemented in children class
        """
        raise NotImplementedError()

    def statematrix_to_stream(self, statematrix):
        """Process a statematrix into a music21.stream.
        
        Parameters
        ----------
        statematrix : array_like
            Should be like the output of stream_to_statematrix
        
        Raises
        ------
        NotImplementedError
            Has to be implemented in children class
        """
        raise NotImplementedError()