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

    def preprocess_stream(self, stream):

        def verify_time_signature(stream, ts_numerator = [2,4]):
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
        raise NotImplementedError()

    def _get_note_tie(self, note):
        if note.tie:
            return note.tie.type
        return None

    def _extract_chord_data(self, fx, c):
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
        raise NotImplementedError()

    def statematrix_to_stream(self, statematrix):
        raise NotImplementedError()