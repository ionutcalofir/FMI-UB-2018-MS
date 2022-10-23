import torch
import torch.nn as nn

class RNN(nn.Module):
  def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, weights=None):
    super(RNN, self).__init__()

    if weights is None:
      self.embeddings = nn.Embedding(vocab_size, embedding_dim)
    else:
      self.embeddings = nn.Embedding.from_pretrained(weights, freeze=False)
    self.hidden_size = hidden_size

    self.i2h = nn.Linear(embedding_dim + hidden_size, hidden_size)
    self.i2o_1 = nn.Linear(embedding_dim + hidden_size, 256)
    self.i2o_2 = nn.Linear(256, output_size)

  def forward(self, input, hidden):
    embeds = self.embeddings(input).view((1, -1))

    combined = torch.cat((embeds, hidden), 1)
    hidden = self.i2h(combined)
    output = self.i2o_1(combined)
    output = self.i2o_2(output)
    output = torch.abs(output)

    return output, hidden

  def initHidden(self):
    return torch.zeros(1, self.hidden_size)

class MyLSTM(nn.Module):
  def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, weights=None):
    super(MyLSTM, self).__init__()

    if weights is None:
      self.embeddings = nn.Embedding(vocab_size, embedding_dim)
    else:
      self.embeddings = nn.Embedding.from_pretrained(weights, freeze=False)
    self.hidden_size = hidden_size

    self.lstm = nn.LSTM(embedding_dim, hidden_size)
    self.fc = nn.Linear(hidden_size, output_size)
    self.sigmoid = nn.Sigmoid()

  def forward(self, input, hidden):
    embeds = self.embeddings(input).view((1, 1, -1))

    output, hidden = self.lstm(embeds)
    output = torch.flatten(output, start_dim=1)
    output = self.fc(output)
    output = self.sigmoid(output)
    # output = torch.abs(output)

    return output, hidden

  def initHidden(self):
    return (torch.randn(1, 1, self.hidden_size),
            torch.randn(1, 1, self.hidden_size))

class CNN(nn.Module):
  def __init__(self, vocab_size, embedding_dim, output_size, weights=None):
    super(CNN, self).__init__()

    if weights is None:
      self.embeddings = nn.Embedding(vocab_size, embedding_dim)
    else:
      self.embeddings = nn.Embedding.from_pretrained(weights, freeze=False)

    self.droput = nn.Dropout(0.5)
    self.fc = nn.Linear(3 * 128, 128)
    self.fc_last = nn.Linear(128, 1)
    self.sigmoid = nn.Sigmoid()

    self.conv1_1 = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(3, 300)),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(28, 1))
    )

    self.conv2_1 = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(4, 300)),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(27, 1))
    )

    self.conv3_1 = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(5, 300)),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(26, 1))
    )

  def forward(self, input):
    embeds = self.embeddings(input).view(1, 1, 30, 300)

    conv1_1 = self.conv1_1(embeds).flatten(start_dim=1)

    conv2_1 = self.conv2_1(embeds).flatten(start_dim=1)

    conv3_1 = self.conv3_1(embeds).flatten(start_dim=1)

    out = torch.cat((conv1_1, conv2_1, conv3_1), 1)

    out = self.droput(out)
    out = self.fc(out)
    out = self.fc_last(out)
    out = self.sigmoid(out)

    return out
