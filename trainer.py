from utils import MetricLogger, ProgressLogger
from models import ClassificationNet, build_classification_model
import numpy as np
import time
import torch
from tqdm import tqdm

from gmml.model_utils import metric_AUROC


def train_one_epoch(data_loader_train, device,model, criterion, optimizer, epoch):
  batch_time = MetricLogger('Time', ':6.3f')
  losses = MetricLogger('Loss', ':.4e')
  progress = ProgressLogger(
    len(data_loader_train),
    [batch_time, losses],
    prefix="Epoch: [{}]".format(epoch))

  model.train()

  end = time.time()
  for i, (samples, targets) in enumerate(data_loader_train):
    samples, targets = samples.float().to(device), targets.float().to(device)
    
    outputs = model(samples)
    loss = criterion(outputs, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.update(loss.item(), samples.size(0))
    batch_time.update(time.time() - end)
    end = time.time()

    if i % 50 == 0:
      progress.display(i)


def evaluate(data_loader_val, device, model, criterion, args=None):
  model.eval()

  with torch.no_grad():
    batch_time = MetricLogger('Time', ':6.3f')
    losses = MetricLogger('Loss', ':.4e')
    progress = ProgressLogger(
      len(data_loader_val),
      [batch_time, losses], prefix='Val: ')
    
    p_out = torch.FloatTensor().to(device)
    t_out = torch.FloatTensor().to(device)

    end = time.time()
    for i, (samples, targets) in enumerate(data_loader_val):
      samples, targets = samples.float().to(device), targets.float().to(device)

      outputs = model(samples)
      loss = criterion(outputs, targets)

      p_out = torch.cat((p_out, outputs), 0)
      t_out = torch.cat((t_out, targets), 0)

      losses.update(loss.item(), samples.size(0))
      losses.update(loss.item(), samples.size(0))
      batch_time.update(time.time() - end)
      end = time.time()

      if i % 50 == 0:
        progress.display(i)

    AUC_all = metric_AUROC(t_out, p_out)
    AUC_mean = np.mean(AUC_all)

    print(f"Validation AUC_mean: {AUC_mean:.4f}, AUC_all: {AUC_all}")
    if args is not None and args.data_set == "CheXpert":
      AUC_mean_5 = np.mean(np.array(AUC_all)[[2,5,6,8,10]])
      print(f"Validation AUC_mean_5: {AUC_mean_5:.4f}")

  return losses.avg


def test_classification(checkpoint, data_loader_test, device, args):
  model = build_classification_model(args)
  print(model)

  modelCheckpoint = torch.load(checkpoint)
  state_dict = modelCheckpoint['state_dict']
  for k in list(state_dict.keys()):
    if k.startswith('module.'):
      state_dict[k[len("module."):]] = state_dict[k]
      del state_dict[k]

  msg = model.load_state_dict(state_dict)
  assert len(msg.missing_keys) == 0
  print("=> loaded pre-trained model '{}'".format(checkpoint))

  if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
  model.to(device)

  model.eval()

  y_test = torch.FloatTensor().to(device)
  p_test = torch.FloatTensor().to(device)

  with torch.no_grad():
    for i, (samples, targets) in enumerate(tqdm(data_loader_test)):
      targets = targets.to(device)
      y_test = torch.cat((y_test, targets), 0)

      if len(samples.size()) == 4:
        bs, c, h, w = samples.size()
        n_crops = 1
      elif len(samples.size()) == 5:
        bs, n_crops, c, h, w = samples.size()

      varInput = torch.autograd.Variable(samples.view(-1, c, h, w).to(device))

      out = model(varInput)
      if args.data_set == "RSNAPneumonia":
        out = torch.softmax(out,dim = 1)
      else:
        out = torch.sigmoid(out)
      outMean = out.view(bs, n_crops, -1).mean(1)
      p_test = torch.cat((p_test, outMean.data), 0)

  return y_test, p_test




