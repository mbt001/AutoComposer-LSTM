# train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import NoteDataset
from model import MusicLSTM
from parse_midi import extract_notes_from_midi, create_pitch_Sequences

# Load & preprocess data
midi_file = "data/maestro-v3.0.0/2018/MIDI-Unprocessed_Recital1-3_MID--AUDIO_02_R1_2018_wav--1.midi"
notes = extract_notes_from_midi(midi_file)
inputs, targets = create_pitch_Sequences(notes, seq_length=50)

dataset = NoteDataset(inputs, targets)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Model setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = MusicLSTM(vocab_size=128).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 5
for epoch in range(epochs):
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        print(f"x.shape BEFORE model: {x.shape}")  # <- Add this

        output, _ = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f}")
torch.save(model.state_dict(), "music_lstm.pth")
print("✅ Model saved to music_lstm.pth")