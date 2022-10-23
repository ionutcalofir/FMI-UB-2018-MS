import os
import shutil
import time
import math
import random
import torch
from dataset import Dataset
from models import RNN, MyLSTM, CNN
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

class Engine():
  def __init__(self, args):
    self.args = args
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.model_name = self.args.model_name

    self.dataset = Dataset(args.train_config_path)
    self.X_train, self.X_test, self.y_train, self.y_test = self.dataset.get_data(args.train_config_path, args.test_config_path)

    self.pretrained = self.args.pretrained
    self.lr = self.args.lr
    self.n_hidden = self.args.n_hidden

    self.is_zero = False

    self._build_model()
    self._build_train_utils()

    self.writer = None

    if self.args.phase == 'train':
      self._create_dirs(logdir_path=args.logdir_path,
                        exp_name=args.exp_name)

      self._build_writer(model=self.rnn,
                         summaries_path=self.summaries_path)

  def _build_train_utils(self):
    self.criterion = torch.nn.L1Loss()
    # self.optimizer = torch.optim.SGD(self.rnn.parameters(), lr=self.lr, momentum=0.9)
    self.optimizer = torch.optim.Adam(self.rnn.parameters(), lr=self.lr)

  def _build_model(self):
    if self.model_name == 'lstm':
      self.rnn = MyLSTM(len(self.dataset._word_to_idx), 300, self.n_hidden, 1, torch.FloatTensor(self.dataset._weights))
    elif self.model_name == 'rnn':
      self.rnn = RNN(len(self.dataset._word_to_idx), 300, self.n_hidden, 1, torch.FloatTensor(self.dataset._weights))
    elif self.model_name == 'cnn':
      if self.pretrained == 1:
        self.rnn = CNN(len(self.dataset._word_to_idx), 300, 1, torch.FloatTensor(self.dataset._weights))
      else:
        self.rnn = CNN(len(self.dataset._word_to_idx), 300, 1, None)
    self.rnn = self.rnn.to(self.device)

  def _build_writer(self,
                    model,
                    summaries_path):
    self.writer = SummaryWriter(summaries_path)

    # dummy_input = None
    # self.writer.add_graph(model, dummy_input)

  def _create_dirs(self,
                   logdir_path,
                   exp_name):
    if not os.path.isdir(logdir_path):
      os.makedirs(logdir_path)

    if os.path.isdir(os.path.join(logdir_path, exp_name)):
      shutil.rmtree(os.path.join(logdir_path, exp_name))
    os.makedirs(os.path.join(logdir_path, exp_name))
    self.exp_path = os.path.join(logdir_path, exp_name)

    self.models_path = os.path.join(self.exp_path, 'models')
    self.summaries_path = os.path.join(self.exp_path, 'summaries')
    os.makedirs(self.models_path)
    os.makedirs(self.summaries_path)

  def validation_step(self):
    self.rnn.eval()
    score = 0
    running_loss = 0

    for X, y in zip(self.X_test, self.y_test):
      X_idxs = self.dataset.sentenceToIdxs(X[0])

      with torch.no_grad():
        if self.model_name in ['lstm', 'rnn']:
          hidden = self.rnn.initHidden()
        if self.model_name == 'lstm':
          hidden = (hidden[0].to(self.device),
                    hidden[1].to(self.device))
        elif self.model_name == 'rnn':
          hidden = hidden.to(self.device)

        if self.model_name in ['lstm', 'rnn']:
          for i in range(len(X_idxs)):
            input = torch.LongTensor([X_idxs[i]])
            input = input.to(self.device)
            output, hidden = self.rnn(input, hidden)
        elif self.model_name == 'cnn':
          X_idxs = X_idxs[:30]
          X_idxs.extend((30 - len(X_idxs)) * [0])
          input = torch.LongTensor(X_idxs)
          input = input.to(self.device)
          output = self.rnn(input)

      y_tensor = torch.tensor([y])
      y_tensor = y_tensor.to(self.device)
      loss = self.criterion(output[0], y_tensor)

      running_loss += loss.item()

      final_output = output.item()
      score += abs(final_output - y)

    print('-------Score on validation: {}-------'.format(score / len(self.y_test)))
    print('-------Loss on validation: {}-------'.format(running_loss / len(self.y_test)))
    return score / len(self.y_test), running_loss / len(self.y_test)

  def train_step(self, X, y, X_idxs):
    self.rnn.train()
    if self.model_name in ['lstm', 'rnn']:
      hidden = self.rnn.initHidden()
    if self.model_name == 'lstm':
      hidden = (hidden[0].to(self.device),
                hidden[1].to(self.device))
    elif self.model_name == 'rnn':
      hidden = hidden.to(self.device)

    self.rnn.zero_grad()

    if self.model_name in ['lstm', 'rnn']:
      for i in range(len(X_idxs)):
        input = torch.LongTensor([X_idxs[i]])
        input = input.to(self.device)
        output, hidden = self.rnn(input, hidden)
    elif self.model_name == 'cnn':
      X_idxs = X_idxs[:30]
      X_idxs.extend((30 - len(X_idxs)) * [0])
      input = torch.LongTensor(X_idxs)
      input = input.to(self.device)
      output = self.rnn(input)

    y_tensor = torch.tensor([y])
    y_tensor = y_tensor.to(self.device)
    loss = self.criterion(output[0], y_tensor)

    if torch.abs(y_tensor - output[0]) > 0.1:
      loss *= 10
    loss.backward()
    self.optimizer.step()

    return output[0].cpu(), loss.item()

  def train(self,
            n_iters):
    start = time.time()

    score = 0
    running_loss = 0
    for iter in range(1, n_iters + 1):
      X, y, X_idxs, X_processed = self.randomTrainingExample()

      output, loss = self.train_step(X, y, X_idxs)
      guess = self.scoreFromOutput(output)

      score += abs(guess - y)
      running_loss += loss

      if iter % 20 == 0:
        guess = self.scoreFromOutput(output)
        print('Iter: {}, Pred: {}, Correct: {}, Loss: {}'.format(iter, guess, y, loss))
        self.writer.add_scalar('Loss/train', running_loss / 20, iter)
        self.writer.add_scalar('Score/train', score / 20, iter)
        score = 0
        running_loss = 0
      if iter % 10000 == 0:
        val_score, val_running_loss = self.validation_step()
        self.writer.add_scalar('Loss/test', val_running_loss, iter)
        self.writer.add_scalar('Score/test', val_score, iter)

      if iter % 20000 == 0:
        print('Time elapsed: {}s'.format(time.time() - start))
        start = time.time()
        torch.save(self.rnn.state_dict(), os.path.join(self.models_path, 'model_{}.pt'.format(iter)))

    self.writer.close()

  def submission(self,
                 path_to_ckpt,
                 submission_test_config_path):
    self.rnn.load_state_dict(torch.load(path_to_ckpt))
    self.rnn.to(self.device)
    self.rnn.eval()

    X_test = self.dataset.get_data_submission(submission_test_config_path)
    preds = []

    for X in X_test:
      X_idxs = self.dataset.sentenceToIdxs(X[0])

      with torch.no_grad():
        if self.model_name in ['lstm', 'rnn']:
          hidden = self.rnn.initHidden()
        if self.model_name == 'lstm':
          hidden = (hidden[0].to(self.device),
                    hidden[1].to(self.device))
        elif self.model_name == 'rnn':
          hidden = hidden.to(self.device)

        if self.model_name in ['lstm', 'rnn']:
          for i in range(len(X_idxs)):
            input = torch.LongTensor([X_idxs[i]])
            input = input.to(self.device)
            output, hidden = self.rnn(input, hidden)
        elif self.model_name == 'cnn':
          X_idxs = X_idxs[:30]
          X_idxs.extend((30 - len(X_idxs)) * [0])
          input = torch.LongTensor(X_idxs)
          input = input.to(self.device)
          output = self.rnn(input)

      final_output = output.item()
      preds.append((X[len(X) - 1], final_output))

    pred_submission = '/'.join(submission_test_config_path.split('/')[:-1]) + '/test_prediction.txt'
    with open(pred_submission, 'w') as f:
      f.write('{},{}\n'.format('id', 'label'))
      for id_pred, out_pred in preds:
        f.write('{},{}\n'.format(id_pred, out_pred))

    # Sanity check -------------------------------------------------------------
    _, _ = self.validation_step()
    # --------------------------------------------------------------------------

  def scoreFromOutput(self, output):
    return output[0].detach().numpy()

  def randomChoice(self, l):
    return random.randint(0, len(l) - 1)

  def randomTrainingExample(self):
    while True:
      idx = self.randomChoice(self.y_train)

      if self.is_zero:
        if self.y_train[idx] == 0.0:
          continue
        else:
          self.is_zero = False
          break
      else:
        if self.y_train[idx] != 0.0:
          continue
        else:
          self.is_zero = True
          break

    X = self.X_train[idx][0]
    y = self.y_train[idx]

    X_idxs = self.dataset.sentenceToIdxs(X)
    X_processed = self.dataset._preprocess_sentence(X)

    return X, y, X_idxs, X_processed
