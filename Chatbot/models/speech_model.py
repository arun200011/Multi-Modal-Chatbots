# models/speech_model.py
import torch
import torch.nn as nn

class SpeechClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128):
        super(SpeechClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch, time, input_dim)
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])  # take last hidden state
        return out
