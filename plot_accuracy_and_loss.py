"""Parse json and plot accuracy and loss graphs."""
import os
import json
import argparse
import matplotlib.pyplot as plt

FIGURES_DIR = "figures"


# Arguments
def parse_args():
    """Parse script arguments.

    Returns:
        Namespace with model name, json path and dataset name.
    """
    parser = argparse.ArgumentParser(description='Analyze network performance.')
    parser.add_argument('--model', '-m',
                        default='Lenet5', type=str,
                        help='Model name')
    parser.add_argument('--train_details_json', '-j',
                        default='out/Lenet5_dr_SGD___dr.json', type=str,
                        help='Json containing loss and accuracy.')
    parser.add_argument('--dataset', '-d',
                        default='Lenet5', type=str,
                        help='Dataset')
    parser.add_argument('--regularization', '-reg',
                        default='none', type=str,
                        help='regularization technique: none, "dr" - dropout, "nb" - batch_normalization, '
                             '"wd" - weight_decay')

    return parser.parse_args()


def main():
    """Parse script arguments, read json and plot accuracy and loss graphs."""
    args = parse_args()

    with open(args.train_details_json, mode='r', encoding='utf-8') as json_f:
        results_dict = json.load(json_f)[-1]

    losses_plot = plt.figure()
    plt.plot(range(1, len(results_dict['train_loss']) + 1),
             results_dict['train_loss'])
    plt.plot(range(1, len(results_dict['val_loss']) + 1),
             results_dict['val_loss'])
    plt.plot(range(1, len(results_dict['test_loss']) + 1),
             results_dict['test_loss'])
    plt.legend(['train', 'val', 'test'])

    # Create graph title suffix depending on regularization type
    if args.regularization == "none":
        title_suffix = ""
    if args.regularization == "wd":
        title_suffix = ' With Weight Decay'
    if args.regularization == "nb":
        title_suffix = f' With Batch Normalization'
    if args.regularization == "dr":
        title_suffix = f' With Dropout'

    plt.title('Loss as a Function of Epoch Number'+title_suffix)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    losses_plot.set_size_inches((8, 8))

    # Create file name for graph figure depending on regularization type
    if args.regularization == "none":
        file_name = f'losses_plot.png'
    if args.regularization == "wd":
        file_name = f'losses_plot_weight_decay.png'
    if args.regularization == "nb":
        file_name = f'losses_plot_batch_normalization.png'
    if args.regularization == "dr":
        file_name = f'losses_plot_dropout.png'

    losses_plot.savefig(
        os.path.join(FIGURES_DIR,file_name))

    accuracies_plot = plt.figure()
    plt.plot(range(1, len(results_dict['train_acc']) + 1),
             results_dict['train_acc'])
    plt.plot(range(1, len(results_dict['val_acc']) + 1),
             results_dict['val_acc'])
    plt.plot(range(1, len(results_dict['test_acc']) + 1),
             results_dict['test_acc'])
    plt.legend(['train', 'val', 'test'])
    plt.title(f'Accuracy as a Function of Epoch Number'+title_suffix)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    accuracies_plot.set_size_inches((8, 8))
    accuracies_plot.savefig(
        os.path.join(FIGURES_DIR,
                     f'{args.dataset}_{args.model}_accuracies_plot_{title_suffix}.png'))
    plt.show()

if __name__ == '__main__':
    main()
