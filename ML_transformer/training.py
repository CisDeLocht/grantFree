import os
import torch
from typing import Tuple
import numpy as np

def train_loop(dataloader, model, loss_fn, optimizer, device):
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  train_loss = 0.0
  train_accuracy = 0.0
  proportion_accuracy = 0.0
  for batch, (X, y) in enumerate(dataloader):
    # Compute prediction and loss
    out, pred = model(X.to(device))
    loss = loss_fn(pred, y.to(device))
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_loss += loss.item()
    s_pred = torch.sigmoid(pred)
    b_pred = (s_pred>0.3).type(torch.float)
    matches = (b_pred == y.to(device))
    num_matches = matches.sum().item()
    total_elements = b_pred.numel()
    proportion = num_matches/total_elements
    proportion_accuracy += proportion
    correct = torch.all(b_pred == y.to(device), dim=1).type(torch.float).sum().item()
    train_accuracy += correct


    if batch % 10 == 0:
      loss, current = loss.item(), batch * len(X)
      print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
      print()

  train_loss /= num_batches
  proportion_accuracy /= num_batches
  train_accuracy /= size
  print(f"Train Error: \n Accuracy: {(100*train_accuracy):>0.5f}%, Avg proportion: {(100*proportion_accuracy):>0.4f}%, Avg loss: {train_loss:>8f} \n")
  return train_loss, train_accuracy


def test_loop(dataloader, model, loss_fn, device):
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  test_loss, test_accuracy = 0, 0

  with torch.no_grad():
    for X, y in dataloader:
      out, pred = model(X.to(device))
      test_loss += loss_fn(pred, y.to(device)).item()
      s_pred = torch.sigmoid(pred)
      b_pred = (s_pred > 0.5).type(torch.float)
      correct = torch.all(b_pred == y.to(device), dim=1).type(torch.float).sum().item()
      test_accuracy += correct

  test_loss /= num_batches
  test_accuracy /= size
  print(f"Test Error: \n Accuracy: {(100*test_accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")
  return test_loss, test_accuracy


def start_training(epochs, train_loader, test_loader, model, loss_fn, optimizer, device):
  train_losses = []
  train_accuracies = []

  test_losses = []
  test_accuracies = []

  for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loss, train_acc = train_loop(train_loader,
                                       model,
                                       loss_fn,
                                       optimizer, device)

    test_loss, test_acc = test_loop(test_loader,
                                    model,
                                    loss_fn, device)

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    test_losses.append(test_loss)
    test_accuracies.append(test_acc)
  torch.save(model.state_dict(), os.path.join("output_dir", f"checkpoint_l{model.getDim()[0]}_m{model.getDim()[1]}_e{epochs}.pth"))
  print("Done!")