import itertools
import configparser
import numpy as np

config = configparser.ConfigParser()
config.read('config.ini')

lowerBound = config.getint('DEFAULT', 'pitch_lowerBound')
upperBound = config.getint('DEFAULT', 'pitch_upperBound')

class FeatureBuilder(object):
    
    @property
    def feature_count(self):
        raise NotImplementedError()
    
    def build_auxillary_info(self, note, state):
        raise NotImplementedError()
    
    def note_input_form(self, note, state, time):
        def _get_or_default(l, i, d):
            try:
                return l[i]
            except IndexError:
                return d

        def _build_context(state):
            context = [0]*12
            for note, notestate in enumerate(state):
                if notestate[0] == 1:
                    pitchclass = (note + lowerBound) % 12
                    context[pitchclass] += 1
            return context

        def _build_beat(time):
            return [2*x-1 for x in [time%2, (time//2)%2, (time//4)%2, (time//8)%2]]

        position = note
        part_position = [position]

        beat = _build_beat(time)
        context = _build_context(state)

        pitchclass = (note + lowerBound) % 12
        part_pitchclass = [int(i == pitchclass) for i in range(12)]
        part_prev_vicinity = list(itertools.chain.from_iterable((_get_or_default(state[:,0:2], note+i, [0,0]) for i in range(-12, 13))))
        part_context = context[pitchclass:] + context[:pitchclass]
        part_aux = self.build_auxillary_info(note, state)

        return np.concatenate([part_position, part_pitchclass, part_prev_vicinity, part_context, part_aux, beat, [0]], axis=0)
    
    def note_state_single_to_input_form(self, state, time):
        return [self.note_input_form(note, state, time) for note in range(len(state))]

    def note_state_matrix_to_input_form(self, statematrix):
        inputform = [ self.note_state_single_to_input_form(state,time) for time,state in enumerate(statematrix) ]
        return inputform