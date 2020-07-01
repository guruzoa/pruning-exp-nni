import argparse
import json
import os
import pandas as pd
import torchvision.models as models
import matplotlib.pyplot as plt

from nni.compression.torch import LevelPruner
from models.cifar10.vgg import VGG
from models.mnist.lenet import LeNet


def get_config_lists_from_pruning_history(files):
    pruning_histories = []
    performances = []
    config_lists = []
    overall_sparsities = []

    for f in files:
        pruning_histories.append(pd.read_csv(f))

    for history in pruning_histories:
        overall_sparsities.append(history.loc[0, 'sparsity'])
        performances.append(history['performance'].max())
        idx = history['performance'].idxmax()
        config_list = history.loc[idx, 'config_list']
        config_lists.append(json.loads(config_list))

    return config_lists, overall_sparsities, performances


def get_config_lists_from_search_result(files):
    performances = []
    config_lists = []

    for f in files:
        with open(f, 'r') as jsonfile:
            j = json.load(jsonfile)
            performances.append(j['performance'])
            config_lists.append(json.loads(j['config_list']))

    return config_lists, performances


def get_performances_fine_tuned(files):
    performances = []

    for f in files:
        with open(f, 'r') as jsonfile:
            j = json.load(jsonfile)
            performances.append(j['finetuned'])

    return performances


def get_original_op(model):
    op_names = []
    op_weights = []

    pruner = LevelPruner(model, [{
        'sparsity': 0.1,
        'op_types': ['default']
    }])

    for wrapper in pruner.get_modules_wrapper():
        op_names.append(wrapper.name)
        op_weights.append(wrapper.module.weight.data.numel())

    return op_names, op_weights


def plot_performance_comparison(args):
    # performances = {
    #     'NetAdaptPruner': [0.8, 0.7, 0.6, 0.5, 0.4],
    #     'SimulatedAnnealingPruenr': [0.7, 0.6, 0.5, 0.4, 0.3]
    # }
    if args.model == 'vgg16':
        performances = {'original': 0.9298}
        sparsities = [0.1, 0.3, 0.5, 0.7, 0.9]
        pruners = ['L1FilterPruner', 'NetAdaptPruner',
                   'SimulatedAnnealingPruner', 'AutoCompressPruner']

    elif args.model == 'resnet18':
        performances = {'original': 0.87}
        sparsities = [0.1, 0.3, 0.5, 0.7, 0.9]
        pruners = ['L1FilterPruner', 'NetAdaptPruner',
                   'SimulatedAnnealingPruner']

    for pruner in pruners:
        performances[pruner] = []
        for sparsity in sparsities:
            with open(os.path.join('experiment_data/cifar10/', args.model, pruner, str(sparsity).replace('.', ''), 'performance.json'), 'r') as jsonfile:
                performance = json.load(jsonfile)
                performances[pruner].append(performance['finetuned'])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for pruner in pruners:
        ax.plot(sparsities, performances[pruner], label=pruner)

    ax.hlines(performances['original'], sparsities[0],
              sparsities[-1], linestyles='dashed', label='original model')

    ax.legend()
    # ax.set_ylim(0.9, 1)

    plt.title('Channel Pruning Comparison on {}/CIFAR10'.format(args.model))
    plt.xlabel('Sparsity')
    plt.ylabel('Accuracy after fine-tuning')
    plt.savefig(
        'experiment_data/performance_comparison_{}.png'.format(args.model))
    plt.close()


def plot_sparsities_distribution(args):
    # if model == 'lenet':
    #     files = list(os.walk('lenet'))[0][-1]
    #     model = LeNet()
    if args.model == 'lenet':
        files = ['lenet/pruning_history.csv']
        model = LeNet()
        notes = 'LeNet, MNIST, SAPruner, fine-grained'
        config_lists, overall_sparsities, performances = get_config_lists_from_pruning_history(
            files)
    elif args.model == 'mobilenet_v2' and args.base_algo == 'level':
        files = ['mobilenet_sapruner_fine_grained/pruning_history_01.csv', 'mobilenet_sapruner_fine_grained/pruning_history_03.csv',
                 'mobilenet_sapruner_fine_grained/pruning_history_04.csv', 'mobilenet_sapruner_fine_grained/pruning_history_05.csv']
        model = models.mobilenet_v2()
        notes = 'MobileNet V2, ImageNet, SAPruner, fine-grained'
        config_lists, overall_sparsities, performances = get_config_lists_from_pruning_history(
            files)
    elif args.model == 'mobilenet_v2' and args.base_algo in ['l1', 'l2']:
        files = ['imagenet_sapruner_channel/search_result_01.json', 'imagenet_sapruner_channel/search_result_02.json',
                 'imagenet_sapruner_channel/search_result_03.json', 'imagenet_sapruner_channel/search_result_04.json', 'imagenet_sapruner_channel/search_result_05.json']
        model = models.mobilenet_v2()
        notes = 'MobileNet V2, ImageNet, SAPruner, channel pruning'
        config_lists, performances = get_config_lists_from_search_result(
            files)
        overall_sparsities = [0.1, 0.2, 0.3, 0.4, 0.5]
        fine_tune_epochs = 10
        performances_fine_tuned = [
            0.47528, 0.4668, 0.46174, 0.4447, 0.4421]
    elif args.model == 'vgg16' or args.model == 'resnet18':
        overall_sparsities = [0.1, 0.3, 0.5, 0.7, 0.9]
        files = []
        for sparsity in overall_sparsities:
            files.append(os.path.join('experiment_data/cifar10/', args.model,
                                      args.pruner, str(sparsity).replace('.', ''), 'search_result.json'))
        if args.model == 'vgg16':
            model = VGG(depth=16)
        elif args.model == 'resnet18':
            model = models.resnet18()
        notes = '{}, CIFAR10, {}, channel pruning'.format(
            args.model, args.pruner)
        config_lists, performances = get_config_lists_from_search_result(
            files)
        fine_tune_epochs = 50
        files = []
        for sparsity in overall_sparsities:
            files.append(os.path.join('experiment_data/cifar10/', args.model,
                                      args.pruner, str(sparsity).replace('.', ''), 'performance.json'))
        performances_fine_tuned = get_performances_fine_tuned(files)
    # elif args.model == 'retinaface':
    #     files = ['01/pruning_history.csv', '02/pruning_history.csv', '03/pruning_history.csv', '04/pruning_history.csv', '05/pruning_history.csv',
    #     '06/pruning_history.csv', '07/pruning_history.csv', '08/pruning_history.csv', '09/pruning_history.csv']
    #     model = RetinaFace(cfg=cfg_re50, phase = 'test')
    #     notes = 'Retinaface, backbone : resnet50'

    fig, axs = plt.subplots(3, 1, figsize=(15, 15))
    fig.suptitle("Pruning Sparsities Distribution ({})".format(notes))
    fig.subplots_adjust(hspace=1)

    # Fig 0 : layer weights
    op_names_original, op_weights_original = get_original_op(model)
    axs[0].plot(op_names_original, op_weights_original, label='op_weights')
    axs[0].set_title("op weights")
    axs[0].legend()

    # Fig 1 : original sparsities
    sparsities = [0]*len(op_names_original)
    for idx, config_list in enumerate(config_lists):
        for config in config_list:
            op_name = config['op_names'][0]
            i = op_names_original.index(op_name)
            sparsities[i] = config['sparsity']
        axs[1].plot(op_names_original, sparsities,
                    label='sparsity: {}, performance: {:.4f}, fine-tuned performance ({} epochs): {:.4f}'.format(overall_sparsities[idx], performances[idx], fine_tune_epochs, performances_fine_tuned[idx]))
        # label='sparsity: {}, performance: {:.4f}'.format(overall_sparsities[idx], performances[idx]))
    axs[1].set_title('original order')
    axs[1].legend()

    # Fig 2 : storted sparsities
    for idx, config_list in enumerate(config_lists):
        op_names_sorted = []
        sparsities = []
        # sorted by layer weights
        for config in config_list:
            sparsities.append(config['sparsity'])
            op_names_sorted.append(config['op_names'][0])

        axs[2].plot(op_names_sorted, sparsities,
                    label='sparsity: {}, performance: {:.4f}, fine-tuned performance ({} epochs): {:.4f}'.format(overall_sparsities[idx], performances[idx], fine_tune_epochs, performances_fine_tuned[idx]))
        # label='sparsity: {}, performance: {:.4f}'.format(overall_sparsities[idx], performances[idx]))
    axs[2].set_title('Sorted by op weights')
    axs[2].legend()

    for ax in axs:
        plt.sca(ax)
        plt.xticks(rotation=90)

    # plt.tight_layout()
    plt.savefig(
        'experiment_data/sparsities_distribution_{}_{}_{}.png'.format(args.model, args.pruner, args.base_algo))
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--model', type=str, default='lenet',
                        help='lenet, mobilenet_v2 or retinaface')
    parser.add_argument('--pruner', type=str, default='SimulatedAnnealingPruner',
                        help='SimulatedAnnealingPruner')
    parser.add_argument('--base-algo', type=str, default='channel',
                        help='channel, or fine-grained')
    args = parser.parse_args()

    # plot_sparsities_distribution(args)
    plot_performance_comparison(args)