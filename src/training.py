# import necessary libraries
from sklearn.metrics import r2_score
import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
import numpy as np

# training the lstm model
def train_lstm_model(
    model,
    train_loader,
    test_loader,
    num_epochs = 10,
    lr = 1e-3,
    device = "cpu"
):
    '''
    We train the lstm here
        model: input for choosing the model
        train_loader: training data
        test_loader: test data
        num_epochs: we choose 20 epochs as the default training epochs
        device: we train on the cpu by default, no gpu
    '''

    # moves neural network onto the cpu
    model = model.to(device)

    # adam gradient descent optimizer, MSE loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    train_losses = []
    test_r2_scores = []

    # start of training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # everytime we do backprop. the gradient gets added, so we clear the gradient here
            optimizer.zero_grad()

            # train the model on X_batch data and predict
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)

            loss.backward()             # backpropagation, compute gradients
            optimizer.step()            # update weights
            total_loss += loss.item()   # sum up losses

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}, batch {batch_idx}/{len(train_loader)}, loss={loss.item():.4f}", flush=True)

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)  

        # get r2 loss and add to test loss
        r2 = evaluate_lstm_model(model, test_loader, device)
        test_r2_scores.append(r2)

        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Test R2: {r2:.4f}")

    return train_losses, test_r2_scores

def evaluate_lstm_model(model, data_loader, device="cpu"):

    # tell the model to just run forward passes for inference
    model.eval()

    all_preds = []
    all_true = []

    # don't save intermediate values since only inference
    with torch.no_grad():

        # for loop to run inference on dataset
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)

            # get predictions on the input sbp data
            y_pred = model(X_batch).cpu().numpy()

            all_preds.append(y_pred)
            all_true.append(y_batch.numpy())

    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)

    return r2_score(all_true, all_preds)