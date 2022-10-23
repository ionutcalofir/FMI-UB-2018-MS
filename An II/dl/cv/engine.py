import os
import shutil
import time
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score

from dataset import BrainHemorrhageDataset
from models import resnext101_32x8d
from utils.my_transforms import my_transforms

class Engine():
  def __init__(self, args):
    self.args = args
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    self._build_model()
    self._build_train_utils(lr=args.lr,
                            class_weights=args.class_weights,
                            optimizer=args.optimizer)

    if self.args.phase == 'train':
      self._create_dirs(logdir_path=args.logdir_path,
                      exp_name=args.exp_name)

      self._build_writer(model=self.model,
                         summaries_path=self.summaries_path)

  def _build_model(self):
    self.model = resnext101_32x8d()
    self.model.fc = torch.nn.Linear(2048, 2)

  def _build_train_utils(self,
                         lr,
                         class_weights,
                         optimizer):
    class_weights = [float(weight) for weight in class_weights]
    class_weights = torch.FloatTensor(class_weights).cuda()
    self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    self.criterion_val = torch.nn.CrossEntropyLoss()

    if optimizer == 'sgd':
      self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
    elif optimizer == 'adam':
      self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

  def _build_transforms(self, transforms_list):
    transforms_list = [my_transforms[transform_name] for transform_name in transforms_list]
    transforms_list = transforms.Compose(transforms_list)

    return transforms_list

  def _build_dataset_loader(self,
                            config_path,
                            data_path,
                            transforms_list,
                            batch_size,
                            shuffle,
                            submission=False):

    dataset = BrainHemorrhageDataset(config_path, data_path, transform=transforms_list, submission=submission)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=shuffle, num_workers=2)

    return dataloader

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

  def _build_writer(self,
                    model,
                    summaries_path):
    self.writer = SummaryWriter(summaries_path)

    # dummy_input = torch.zeros(1, 3, 224, 224)
    # self.writer.add_graph(model, dummy_input)

  def _train_step(self, data):
    self.model.train()
    self.optimizer.zero_grad()
    y_true = []
    y_pred = []

    inputs, labels = data[0].to(self.device), data[1].to(self.device)

    outputs = self.model(inputs)
    _, predicted = torch.max(outputs.data, 1)

    loss = self.criterion(outputs, labels)
    loss.backward()
    self.optimizer.step()

    y_true.extend(labels.to('cpu').numpy().tolist())
    y_pred.extend(predicted.to('cpu').numpy().tolist())

    return y_true, y_pred, loss.item()

  def _test_step(self, datasetloader, submission=False):
    self.model.eval()

    if not submission:
      y_true = []
      y_pred = []
      running_loss = 0
      with torch.no_grad():
        for data in datasetloader:
          inputs, labels = data[0].to(self.device), data[1].to(self.device)

          outputs = self.model(inputs)
          _, predicted = torch.max(outputs.data, 1)
          loss = self.criterion_val(outputs, labels)

          running_loss += loss.item()
          y_true.extend(labels.to('cpu').numpy().tolist())
          y_pred.extend(predicted.to('cpu').numpy().tolist())

      running_loss = running_loss / len(datasetloader)
      return y_true, y_pred, running_loss
    else:
      inputs_name = []
      y_pred = []
      with torch.no_grad():
        for data in datasetloader:
          inputs, batch_inputs_name = data[0].to(self.device), data[1]

          outputs = self.model(inputs)
          _, predicted = torch.max(outputs.data, 1)

          y_pred.extend(predicted.to('cpu').numpy().tolist())
          inputs_name.extend(batch_inputs_name)

      return y_pred, inputs_name

  def train(self,
            data_path,
            train_transforms,
            test_transforms,
            train_config_path,
            test_config_path,
            batch_size,
            epochs):
    self.model.to(self.device)

    transform_train = self._build_transforms(train_transforms)
    transform_test = self._build_transforms(test_transforms)

    trainloader = self._build_dataset_loader(config_path=train_config_path,
                                             data_path=data_path,
                                             transforms_list=transform_train,
                                             batch_size=batch_size,
                                             shuffle=True)

    testloader = self._build_dataset_loader(config_path=test_config_path,
                                            data_path=data_path,
                                            transforms_list=transform_test,
                                            batch_size=batch_size,
                                            shuffle=False)

    for epoch in range(epochs):
      start = time.time()
      y_true = []
      y_pred = []
      running_loss = 0.0

      for i, data in enumerate(trainloader, 0):
        y_true_batch, y_pred_batch, loss = self._train_step(data)

        self.writer.add_scalar('Loss/train', loss, (epoch * len(trainloader) + i))
        running_loss += loss
        if i % 10 == 9:
          print('[{}, {}] loss: {}'.format(epoch + 1, i + 1, running_loss / 10))
          running_loss = 0.0

        y_true.extend(y_true_batch)
        y_pred.extend(y_pred_batch)

      self.writer.add_scalar('Acc/train', f1_score(y_true, y_pred), epoch)
      print('F1 train: {}'.format(f1_score(y_true, y_pred)))
      torch.save(self.model.state_dict(), os.path.join(self.models_path, 'model_{}.pt'.format(epoch + 1)))

      val_y_true, val_y_pred, val_running_loss = self._test_step(datasetloader=testloader)
      self.writer.add_scalar('Acc/test', f1_score(val_y_true, val_y_pred), epoch)
      self.writer.add_scalar('Loss/test', val_running_loss, epoch)
      print('Loss test: {}'.format(val_running_loss))
      print('F1 test: {}'.format(f1_score(val_y_true, val_y_pred)))

      print('Time elapsed: {}s'.format(time.time() - start))

    self.writer.close()
    print('Finished Training')

  def submission(self,
                 path_to_ckpt,
                 test_config_path,
                 data_path,
                 test_transforms,
                 submission_test_config_path,
                 batch_size):
    self.model.load_state_dict(torch.load(path_to_ckpt))
    self.model.to(self.device)

    transform_test = self._build_transforms(test_transforms)

    # Sanity check -------------------------------------------------------------
    testloader = self._build_dataset_loader(config_path=test_config_path,
                                            data_path=data_path,
                                            transforms_list=transform_test,
                                            batch_size=batch_size,
                                            shuffle=False)
    y_true, y_pred, running_loss = self._test_step(datasetloader=testloader)
    print('Loss test: {}'.format(running_loss))
    print('F1 test: {}'.format(f1_score(y_true, y_pred)))
    # --------------------------------------------------------------------------

    submissionloader = self._build_dataset_loader(config_path=submission_test_config_path,
                                                  data_path=data_path,
                                                  transforms_list=transform_test,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  submission=True)
    y_pred, inputs_name = self._test_step(submissionloader, submission=True)
    with open('./data/configs/submission_test/submission_test_predicted.txt', 'w') as f:
      f.write('id,class\n')
      for name, y in zip(inputs_name, y_pred):
        f.write('{},{}\n'.format(name.split('/')[-1][:-4], y))
