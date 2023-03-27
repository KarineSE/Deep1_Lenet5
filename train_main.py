"""Main training script."""
import argparse
from utils import load_dataset, load_model
from trainer import LoggingParameters, Trainer
from torch import nn  # building blocks of a CNN
import torch.optim as optim  # optimizer algorithms package


DATA_DIR = "../Fashion_mnist"
OUTPUT_DIR = "out"
CHECKPOINT_DIR = "checkpoints"


# Arguments
def parse_args():
    """Parse script arguments.

    Get training hyper-parameters such as: learning rate, momentum,
    batch size, number of training epochs and optimizer.
    Get training dataset and the model name.
    """
    parser = argparse.ArgumentParser(description='Training models with Pytorch')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='SGD momentum')
    parser.add_argument('--batch_size', '-b', default=128, type=int,
                        help='Training batch size')
    parser.add_argument('--epochs', '-e', default=50, type=int,
                        help='Number of epochs to run')
    parser.add_argument('--model_name', '-m', default='Lenet5', type=str,
                        help='Model name: Lenet5')
    parser.add_argument('--optimizer', '-o', default='SGD', type=str,
                        help='Optimization Algorithm')
    parser.add_argument('--regularization', '-reg',
                        default='none', type=str,
                        help='regularization technique: "none", "dr" - dropout, "nb" - batch_normalization, '
                             '"wd" - weight_decay')
    parser.add_argument('--wd_param', '-wd_p',
                        default=0.001, type=float,
                        help='Weight decay parameter') # Usually between 0.0001 to 0.001
    parser.add_argument('--dataset', '-d',
                        default='fashionmnist', type=str,
                        help='dataset name')


    return parser.parse_args()


def main():
    """Parse arguments and train model on dataset."""
    args = parse_args()

    # Data
    print(f'==> Preparing data: {args.dataset}..')
    train_dataset = load_dataset(dataset_part='train', data_dir=DATA_DIR)
    val_dataset = load_dataset(dataset_part='val', data_dir=DATA_DIR)
    test_dataset = load_dataset(dataset_part='test', data_dir=DATA_DIR)

    # Model
    model_name = args.model_name
    reg = args.regularization
    model = load_model(model_name=model_name, reg=args.regularization)

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Build optimizer
    wd_param = 0 # Weight decay parameter
    if reg == "wd":
        wd_param = args.wd_param
    optimizers = {
        'SGD': lambda: optim.SGD(model.parameters(),
                                 lr=args.lr,
                                 momentum=args.momentum,
                                 weight_decay=wd_param),
        'Adam': lambda: optim.Adam(model.parameters(), lr=args.lr, weight_decay=wd_param),
    }
    optimizer_name = args.optimizer
    if optimizer_name not in optimizers:
        raise ValueError(f'Invalid Optimizer name: {optimizer_name}')
    print(f"Building optimizer {optimizer_name}...")

    optimizer = optimizers[args.optimizer]()

    print(optimizer)
    optimizer_params = optimizer.param_groups[0].copy()
    # remove the parameter values from the optimizer parameters for a cleaner log
    del optimizer_params['params']

    # Batch size
    batch_size = args.batch_size

    # Training Logging Parameters
    logging_parameters = LoggingParameters(model_name=model_name,
                                           dataset_name=args.dataset,
                                           optimizer_name=optimizer_name,
                                           optimizer_params=optimizer_params,
                                           regularization=args.regularization)

    # Create an abstract trainer to train the model with the data and parameters
    # above:
    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      criterion=criterion,
                      batch_size=batch_size,
                      train_dataset=train_dataset,
                      validation_dataset=val_dataset,
                      test_dataset=test_dataset)

    # Train, evaluate and test the model:
    trainer.run(epochs=args.epochs, logging_parameters=logging_parameters, checkpoint_dir=CHECKPOINT_DIR, output_dir=OUTPUT_DIR)


if __name__ == '__main__':
    main()
