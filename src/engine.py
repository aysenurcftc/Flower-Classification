import torch
from tqdm.auto import tqdm
import wandb

from config.config import config


def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    """Perform a single training step for a pytorch model.

    Args:
        model : the neural network model to train.
        dataloader : dataloader providing the training data in batches.
        loss_fn : the loss function to evaluate the model's predictions.
        optimizer : the optimizer to update the model's parameters.
        device: target device
    Returns:
        tuple:
            *train_loss (float) :  The average loss over the training set.
            *train_acc (float) : The average accuracy over the training set.
    """

    # put model in train mode
    model.train()

    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        # send data to target device
        X, y = X.to(device), y.to(device)

        # forward pass
        y_pred = model(X)

        # calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # optimizer zero grad
        optimizer.zero_grad()

        # loss backward
        loss.backward()

        # optimizer step
        optimizer.step()

        # calculate and acc metric
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    return train_loss, train_acc


def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    log_counter: int,
    test_table: wandb.Table,
):
    """Performs a single evaluation step for a PyTorch model.

    Args:
        model : the neural network model to evaluate.
        dataloader : DataLoader providing the test/validation data in batches.
        loss_fn : The loss function to evaluate the model's predictions.
        device: the target device.

    Returns:
        tuple :
            * test_loss (float) : The average loss over the test set.
            * test_acc (float) : The average accuracy over the test set.
    """

    # put model in eval mode
    model.eval()

    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            # send data to target device
            X, y = X.to(device), y.to(device)

            # forward pass
            test_pred_logits = model(X)

            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)

            if log_counter < len(dataloader):
                scores = torch.softmax(test_pred_logits, dim=1).cpu().numpy()
                images = X.cpu().numpy()
                labels = y.cpu().numpy()
                preds = test_pred_labels.cpu().numpy()

                for img, label, pred, score in zip(images, labels, preds, scores):
                    test_table.add_data(
                        log_counter,
                        wandb.Image(img.transpose(1, 2, 0)),
                        pred,
                        label,
                        *score
                    )
                log_counter += 1

        test_loss = test_loss / len(dataloader)
        test_acc = test_acc / len(dataloader)

        return test_loss, test_acc, test_table


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    device: torch.device,
):
    """Trains and tests a PyTorch model."""

    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    columns = ["id", "image", "guess", "truth"] + [f"score_{i}" for i in range(config["class_names"])]
    test_table = wandb.Table(columns=columns)

    # loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )
        test_loss, test_acc, test_table = test_step(
            model=model, dataloader=test_dataloader, loss_fn=loss_fn, device=device, log_counter=0, test_table=test_table,
        )

        wandb.log({"test_predictions": test_table})

        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "test_loss": test_loss,
                "test_accuracy": test_acc,
            }
        )

    wandb.finish()
    return results

