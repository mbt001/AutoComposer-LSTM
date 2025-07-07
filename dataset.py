import torch
from torch.utils.data import Dataset 

class NoteDataset(Dataset):
    def __init__(self, input_Sequences, target_pitches):
        self.inputs=input_Sequences
        self.targets=target_pitches

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input_Seq= torch.tensor(self.inputs[idx], dtype=torch.long).squeeze()
        target = torch.tensor(self.targets[idx], dtype=torch.long)
        return input_Seq, target
      