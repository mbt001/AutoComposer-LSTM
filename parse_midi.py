import pretty_midi
import os
from dataset import NoteDataset
from torch.utils.data import DataLoader


def extract_notes_from_midi(midi_path):
    try:
        midi_data=pretty_midi.PrettyMIDI(midi_path)
        notes = [(note.start, note.end, note.pitch)
                for instrument in midi_data.instruments if not instrument.is_drum  for note in instrument.notes]

        notes.sort(key=lambda x : x[0])
        return notes
    except Exception as e:
        print(f"Error Parsing {midi_path}: {e}")
        return []

def create_pitch_Sequences(notes, seq_length=50):
    pitches =[pitch for (_,_,pitch) in notes]

    inputs , targets = [], []

    for i in range(len(pitches)- seq_length):
        inputs.append(pitches[i : i+seq_length])
        targets.append(pitches[i+seq_length])

    return inputs , targets



if __name__ == "__main__":
    midi_file = "data/maestro-v3.0.0/2018/MIDI-Unprocessed_Recital1-3_MID--AUDIO_02_R1_2018_wav--1.midi"
    note_sequence = extract_notes_from_midi(midi_file)

    #print(f"Extracted {len(note_sequence)} notes")
    #print("First 10 notes:")
    #for note in note_sequence[:10]:
    #    print(note)

    inputs, targets = create_pitch_Sequences(note_sequence, seq_length=50)
    #print(f"Generated {len(inputs)} training sequences.")
    #print("Sample input:", inputs[0])
    #print("Target pitch:", targets[0])

    dataset = NoteDataset(inputs, targets)
    loader = DataLoader(dataset,batch_size=32,shuffle=True)

    batch =iter(loader)
    inputs, targets = next(batch)

    print(inputs.shape, targets.shape)