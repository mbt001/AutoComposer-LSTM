# generate.py

import torch
import pretty_midi
from model import MusicLSTM
from parse_midi import extract_notes_from_midi
import numpy as np

# Load model and set to eval
vocab_size = 128  # since MIDI pitch range is 0–127
model = MusicLSTM(vocab_size=vocab_size)
model.load_state_dict(torch.load("music_lstm.pth"))
model.eval()

# Use MPS if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# Load MIDI and get a seed
midi_file = "data/maestro-v3.0.0/2018/MIDI-Unprocessed_Recital1-3_MID--AUDIO_02_R1_2018_wav--1.midi"
notes = extract_notes_from_midi(midi_file)
pitches = [pitch for (_, _, pitch) in notes]

# Take the first 50 pitches as seed
seed = pitches[:50]
generated = seed.copy()

# Generate 100 new notes
input_seq = torch.tensor(seed, dtype=torch.long).unsqueeze(0).to(device)  # shape: [1, 50]

for _ in range(100):
    with torch.no_grad():
        output, _ = model(input_seq)
        predicted_pitch = torch.argmax(output, dim=-1).item()

    generated.append(predicted_pitch)
    input_seq = torch.tensor(generated[-50:], dtype=torch.long).unsqueeze(0).to(device)

# Convert generated pitches to MIDI
output_midi = pretty_midi.PrettyMIDI()
instrument = pretty_midi.Instrument(program=0)

start = 0
duration = 0.5  # fixed duration
for pitch in generated:
    note = pretty_midi.Note(
        velocity=100,
        pitch=int(pitch),
        start=start,
        end=start + duration
    )
    instrument.notes.append(note)
    start += duration

output_midi.instruments.append(instrument)
output_midi.write("generated_output.mid")
print("✅ Generated music saved as generated_output.mid")
