import sys
sys.path.insert(0, 'E:\Code\music')
import mido, fluidsynth
import time
from music_transformer import MusicTransformer
import music_helpers as mh
import torch

# Loads the SoundFont and Model Endpoint
# You will need to download your own SF2 if you do not have one
SF2_PATH = r'c:\Users\James\Music\Samples\GeneralUser_GS_v2.0.3--doc_r6\GeneralUser-GS\GeneralUser-GS.sf2'
ENDPOINT = torch.load('music_transformer.pth', map_location='cpu', weights_only=False)

fs = fluidsynth.Synth()
fs.setting('midi.driver', 'none')
fs.start()
sfid = fs.sfload(SF2_PATH)
fs.program_select(0, sfid, 0, 0)
held, recorded = {}, []

'''
This block acquires info from the model mainly the mappings that
map from feature -> encodings -> feature
'''
pitch_vocab, beat_vocab, vel_vocab = ENDPOINT['pitch_vocab'], ENDPOINT['beat_vocab'], ENDPOINT['vel_vocab']

idx_to_pitch, pitch_to_idx = mh.make_mappings(pitch_vocab)
idx_to_beat, beat_to_idx = mh.make_mappings(beat_vocab)
idx_to_vel, vel_to_idx = mh.make_mappings(vel_vocab)

model = MusicTransformer(
    vocab_size=ENDPOINT['pitch_vocab_size'],
    beat_vocab_size=ENDPOINT['beat_vocab_size'],
    vel_vocab_size=ENDPOINT['vel_vocab_size']
)

model.load_state_dict(ENDPOINT['model_state_dict'])
model.eval()


def time_to_ticks(time_held, tempo=120.0, resolution=480.0):
    seconds_per_beat = 60.0 / tempo
    held_ticks = time_held / seconds_per_beat * resolution
    duration = mh.quantize(held_ticks, resolution)
    return duration    

def nearest(value, vocab):
    return min(vocab, key=lambda v: abs(v - value))

def encode_sequence(notes):
    return [
        (
            pitch_to_idx[nearest(pitch, pitch_vocab)], 
            beat_to_idx[nearest(beat, beat_vocab)], 
            vel_to_idx[nearest(vel, vel_vocab)]
        ) 
        for pitch, beat, vel in notes
    ]

# returns raw midi information from midi input device
def get_from_midi(*, seq_length, tempo=120.0):
    with mido.open_input('Arturia KeyLab Essential 49 0') as port:      #input the name of your midi device here
            for msg in port:
                if msg.type == 'note_on' and msg.velocity > 0:
                    fs.noteon(0, msg.note, msg.velocity)
                    held[msg.note] = (time.time(), msg.velocity)
                
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    fs.noteoff(0, msg.note)
                    if msg.note in held:
                        start, vel = held.pop(msg.note)
                        time_held = time.time() - start
                        recorded.append((msg.note, time_to_ticks(time_held, tempo=tempo), vel))

                        if len(recorded) >= seq_length:   
                            break
    return recorded

# preps midi sequence to be sent to the model for inference
def build_seed(encoded_seq):
    return torch.tensor(encoded_seq).unsqueeze(0)


def play_notes(note_sequence, tempo=120.0, resolution=480.0, channel=0):
    seconds_per_beat = 60.0 / tempo
    for pitch, duration_ticks, velocity in note_sequence:
        duration_seconds = duration_ticks / resolution * seconds_per_beat
        fs.noteon(channel, int(pitch), int(velocity))
        time.sleep(duration_seconds)
        fs.noteoff(channel, int(pitch))


while True:
    recorded.clear()   
    notes = get_from_midi(seq_length=16)    # adjust how many notes you will play here
    seed = build_seed(encode_sequence(notes))
    model_output = model.generate(seed, temp=1.0)
    model_notes = mh.format_notes(model_output, idx_to_pitch, idx_to_beat, idx_to_vel)
    play_notes(model_notes)

      
            

    
