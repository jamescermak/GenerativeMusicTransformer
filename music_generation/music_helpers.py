import pretty_midi
import numpy as np

GRID_16TH = np.linspace(0.25, 4.0, 16)
GRID_TRIP = np.linspace(1/3, 4.0, 12)
GRID = np.union1d(GRID_16TH, GRID_TRIP)

#take a given note sequence [(pitch, duration, velocity), ...] and return/output it as a midi
def notes_to_midi(note_sequence, tempo=120, resolution=480.0):
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    instrument = pretty_midi.Instrument(program=0)
    
    seconds_per_beat = 60.0 / tempo
    elapsed_time = 0.0
    for pitch, duration_ticks, velocity in note_sequence:
        duration_seconds = duration_ticks / resolution * seconds_per_beat
        note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=elapsed_time, end=elapsed_time + duration_seconds)
        instrument.notes.append(note)
        elapsed_time += duration_seconds
    
    pm.instruments.append(instrument)
    return pm


# take a 3D Matrix representing a single sequence and output it to [(pitch, duration, velocity),...]
def format_notes(sequence, idx_to_note, idx_to_beat, idx_to_vel):    
    new_seq = sequence.flatten()
    idx_list = new_seq.tolist()

    note_sequence = list(
        zip(
            [idx_to_note[i] for i in idx_list[::3]],
            [idx_to_beat[j] for j in idx_list[1::3]],
            [idx_to_vel[k] for k in idx_list[2::3]],
        )
    )
    
    return note_sequence

# takes vocab, returns IDX -> VAL , VAL -> IDX
def make_mappings(vocab):
    idx_to_note = {idx : note for idx, note in enumerate(vocab)}
    note_to_idx = {note : idx for idx, note in idx_to_note.items()}
    return idx_to_note, note_to_idx

# helper for grid snapping in time
def quantize(duration, resolution):
    note_value = duration / resolution
    quantized = min(GRID, key=lambda g: abs(g - note_value))
    return round(quantized * resolution, 1)