import torch
import wandb
from config.config import config
from src import data_setup, model, engine, utils
from src.utils import save_model

name = f"ViT_{config['BATCH_SIZE']}_bs_{config['LEARNING_RATE']}_lr_{config['NUM_EPOCHS']}_epochs"

wandb.init(
    project="vision-transformer-training",
    name= name,
    config={
        "epochs": config["NUM_EPOCHS"],
        "batch_size": config["BATCH_SIZE"],
        "learning_rate": config["LEARNING_RATE"],
        "hidden_units": config["HIDDEN_UNITS"],
        "train_dir": config["train_dir"],
        "test_dir": config["test_dir"],
    },
)

run_config = wandb.config

NUM_EPOCHS = run_config.epochs
BATCH_SIZE = run_config.batch_size
LEARNING_RATE = run_config.learning_rate
HIDDEN_UNITS = run_config.hidden_units
train_dir = run_config.train_dir
test_dir = run_config.test_dir


def get_device():
    """Returns the device to be used for training."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def setup_dataloaders(train_dir, test_dir, transform, batch_size):
    """Sets up train and test DataLoaders."""
    return data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=transform,
        batch_size=batch_size,
    )


def setup_model(num_classes, device):
    """Initializes the model and returns it."""
    vit_model = model.VisionTransformer(num_classes=num_classes).to(device)
    return vit_model


def setup_loss_and_optimizer(model, learning_rate):
    """Sets up loss function and optimizer."""
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return loss_fn, optimizer


def main():
    # Setup device
    device = get_device()

    # Initialize model
    model_instance = setup_model(num_classes=config["class_names"], device=device)

    # Get transforms
    data_transform = model_instance.get_transforms()

    # Setup DataLoaders
    train_dataloader, test_dataloader, class_names = setup_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=data_transform,
        batch_size=BATCH_SIZE,
    )

    # Setup loss function and optimizer
    loss_fn, optimizer = setup_loss_and_optimizer(model_instance, LEARNING_RATE)

    results= engine.train(
        model=model_instance,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=NUM_EPOCHS,
        device=device,
    )

    target_dir = "../models/"

    save_model(
       model_instance,
        target_dir,
        model_name=f"pretrained_vit16_{BATCH_SIZE}_batch-size_{LEARNING_RATE}_learning-rate_{NUM_EPOCHS}_epochs.pt",
    )


if __name__ == "__main__":
    main()
