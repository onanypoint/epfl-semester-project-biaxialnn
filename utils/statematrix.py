import math
import music21 as m21
import numpy as np
import configparser
import datetime

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

    def preprocess_sstream(self, stream):
        """
        Preprocess music21.stream object.
        
        Parameters
        ----------
        stream : music21.stream
            The stream to process.
        
        Returns
        -------
        music21.stream
            The preprocessed stream.
        """
        raise NotImplementedError()

    @property
    def information_count(self):
        """Return the number of features
        
        It is used during the model creation. The model cannot infer the size 
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

class StateMatrixBuilderSimple(StateMatrixBuilder):
    """Minimal statematrix builder
    
    This builder constructs statematrix (or stream) using only the play and 
    articulations informations. If going from state to stream, builds a 
    music21.stream object with the minimal construction based on a straight 
    forward approach.

    """

    @property
    def information_count(self):
        return 2

    def preprocess_stream(self, stream):
        return stream
        
    def stream_to_statematrix(self, stream):
        fy = lambda n: n.pitch.ps

        s = stream.flat

        duration = max(s.highestTime, s.duration.quarterLength)
        quantization_normalized = int(quantization / 4)
        duration_quantized = int(math.ceil(duration * quantization / 4)) + 1
        span = upperBound - lowerBound

        statematrix = np.zeros((duration_quantized, span, self.information_count))
        max_time = 0

        for obj in s.flat.getElementsByClass((m21.meter.TimeSignature, m21.note.Note, m21.chord.Chord)):
            valueObjPairs = []
            if isinstance(obj, m21.note.Note):
                valueObjPairs = [(fy(obj), obj)]

            elif isinstance(obj, m21.chord.Chord):
                values = self._extract_chord_data(fy, obj)
                valueObjPairs = [(v, obj) for v in values]

            elif isinstance(obj, m21.meter.TimeSignature):
                if obj.numerator not in (2, 4):
                    print("Found time signature event {}. Bailing!".format(
                        obj.ratioString))
                    return statematrix[:max_time]
                continue

            for v, objSub in valueObjPairs:
                numericValue = m21.common.roundToHalfInteger(v)

                if numericValue % 1 != 0:
                    print("Warning: note has been rounded from",
                          numericValue, "to", int(numericValue))
                    numericValue = int(numericValue)

                if (numericValue < lowerBound) or (numericValue >= upperBound):
                    print("Note {} at time {} out of bounds (ignoring)".format(
                        numericValue, objSub.offset))
                    pass

                else:
                    start = objSub.offset
                    length = objSub.quarterLength
                    tie = self._get_note_tie(objSub)

                    start_quant = int(start * quantization_normalized)
                    length_quant = int(length * quantization_normalized)
                    end_quant = start_quant + length_quant
                    max_time = max(max_time, end_quant)

                    if start_quant != end_quant:
                        statematrix[start_quant:end_quant, numericValue - lowerBound, 0] = 1

                    if not tie or tie == 'start':
                        statematrix[start_quant, numericValue - lowerBound, 0:2] = 1
                        
        return statematrix[:max_time]

    def statematrix_to_stream(self, statematrix, name='untitled'):
        s = m21.stream.Score()
        
        s.insert(0, m21.metadata.Metadata())
        s.metadata.title = name
        s.metadata.composer = 'Biaxial neural network'
        s.metadata.date = m21.metadata.DateSingle(str(datetime.datetime.now().year))
        
        p = m21.stream.Part()
        
        for i, values in enumerate(statematrix.transpose((1,0,2))):
            pitch = i + lowerBound
        
            note = None
            time = 0
            
            for t, values in enumerate(values):
                play, art = values[0:2]
                dynamics = values[-2:-1]
                
                if art and play:
                    if note:
                        note.quarterLength = time/4.0
                        p.insert(start/4.0, note)
                    
                    note = m21.note.Note(pitch)
                    time = 0
                    start = t
                
                if not play and note:
            
                    note.quarterLength = time/4.0
                    p.insert(start/4.0, note)   
                    note = None
                    time = 0
                    start = None
                
                if play:
                    time = time + 1
                    
        p.makeMeasures(inPlace=True) 
        s.append(p)
        
        return s