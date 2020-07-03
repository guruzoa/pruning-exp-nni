import argparse
import json
import os
import pandas as pd
import torchvision
import matplotlib.pyplot as plt

from nni.compression.torch import LevelPruner
from examples.models.cifar10.vgg import VGG
from examples.models.mnist.lenet import LeNet


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
            performances.append(j['performance']['finetuned'])

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


def plot_performance_comparison(args, normalize=False):
    references={
        'AutoCompressPruner':{
            'cifar10':{
                'vgg16':{
                    'performance': 0.9321,
                    'params':52.2,
                    'flops':8.8
                },
                'resnet18':{
                    'performance': 0.9381,
                    'params':54.2,
                    'flops':12.2
                }
            }
        }
    }
    target_sparsities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975]
    pruners = ['L1FilterPruner', 'NetAdaptPruner', 'SimulatedAnnealingPruner', 'AutoCompressPruner']
    
    performances = {}
    flops = {}
    params = {}
    sparsities = {}
    for pruner in pruners:
        performances[pruner] = []
        flops[pruner] = []
        params[pruner] = []
        sparsities[pruner] = []
        for sparsity in target_sparsities:
            f = os.path.join('experiment_data/cifar10/', args.model, pruner, str(sparsity).replace('.', ''), 'result.json')
            if os.path.exists(f):
                with open(f, 'r') as jsonfile:
                    result = json.load(jsonfile)
                    sparsities[pruner].append(sparsity)
                    performances[pruner].append(result['performance']['finetuned'])
                    if normalize:
                        flops[pruner].append(result['flops']['original']/result['flops']['speedup'])
                        params[pruner].append(result['params']['original']/result['params']['speedup'])
                    else:
                        flops[pruner].append(result['flops']['speedup'])
                        params[pruner].append(result['params']['speedup'])
            

    fig, axs = plt.subplots(3, 1, figsize=(8, 15))
    fig.suptitle('Channel Pruning Comparison on {}/CIFAR10'.format(args.model))
    fig.subplots_adjust(hspace=0.5)


    for pruner in pruners:
        axs[0].scatter(flops[pruner], performances[pruner], label=pruner)
        axs[1].scatter(params[pruner], performances[pruner], label=pruner)
        axs[2].scatter(sparsities[pruner], performances[pruner], label=pruner)

    if normalize:
        axs[0].annotate("original", (1, result['performance']['original']))
        axs[0].set_xscale('log')
    else:
        # axs[0].annotate("original", (result['flops']['original'], result['performance']['original']))
        axs[0].plot(result['flops']['original'], result['performance']['original'], 'rx', label='original')
        axs[0].plot(result['flops']['original']/references['AutoCompressPruner']['cifar10'][args.model]['flops'], references['AutoCompressPruner']['cifar10'][args.model]['performance'], 'bx', label='AutoCompress Paper')
    axs[0].set_title("performance v.s. FLOPS")
    axs[0].set_xlabel("FLOPS (calculated after speedup)")
    axs[0].set_ylabel('Accuracy after fine-tuning')
    axs[0].legend()

    if normalize:
        axs[1].annotate("original", (1, result['performance']['original']))
        axs[1].set_xscale('log')
    else:
        # axs[1].annotate("original", (result['params']['original'], result['performance']['original']))
        axs[1].plot(result['params']['original'], result['performance']['original'], 'rx', label='original')
        axs[1].plot(result['params']['original']/references['AutoCompressPruner']['cifar10'][args.model]['params'], references['AutoCompressPruner']['cifar10'][args.model]['performance'], 'bx', label='AutoCompress Paper')
    axs[1].set_title("performance v.s. params")
    axs[1].set_xlabel("number of weight parameters (calculated after speedup)")
    axs[1].set_ylabel('Accuracy after fine-tuning')
    axs[1].legend()

    # axs[2].annotate("original", (0, result['performance']['original']))
    axs[2].hlines(result['performance']['original'], sparsities[pruner][0], sparsities[pruner][-1], linestyles='dashed', label='original model')
    axs[2].set_title("performance v.s. sparsities")
    axs[2].set_xlabel("target sparsity")
    axs[2].set_ylabel('Accuracy after fine-tuning')
    axs[2].legend()

    plt.savefig('experiment_data/performance_comparison_{}.png'.format(args.model))
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
        model = torchsivion.models.mobilenet_v2()
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
    elif args.model in ['vgg16', 'resnet18']:
        overall_sparsities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975]
        files = []
        for sparsity in overall_sparsities:
            f = os.path.join('experiment_data/cifar10/', args.model, args.pruner, str(sparsity).replace('.', ''), 'search_result.json')
            if os.path.exists(f):
                files.append(f)
        if args.model == 'vgg16':
            model = VGG(depth=16)
        elif args.model == 'resnet18':
            model = torchvision.models.resnet18()
        notes = '{}, CIFAR10, {}, channel pruning'.format(args.model, args.pruner)
        config_lists, performances = get_config_lists_from_search_result(files)
        fine_tune_epochs = 50
        files = []
        for sparsity in overall_sparsities:
            f = os.path.join('experiment_data/cifar10/', args.model, args.pruner, str(sparsity).replace('.', ''), 'result.json')
            if os.path.exists(f):
                files.append(f)
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
    for idx, config_list in enumerate(config_lists):
        sparsities = [0]*len(op_names_original)
        for config in config_list:
            op_name = config['op_names'][0]
            i = op_names_original.index(op_name)
            sparsities[i] = config['sparsity']
        axs[1].plot(op_names_original, sparsities, label='sparsity: {}, performance: {:.4f}'.format(overall_sparsities[idx], performances_fine_tuned[idx]))
    axs[1].set_title('original order')
    axs[1].legend()

    # Fig 2 : storted sparsities
    op_names_sorted = [op_name for _,op_name in sorted(zip(op_weights_original, op_names_original), key=lambda pair: pair[0])]
    for idx, config_list in enumerate(config_lists):
        sparsities = [0] * len(op_names_sorted)
        # sorted by layer weights
        for config in config_list:
            op_name = config['op_names'][0]
            i = op_names_sorted.index(op_name)
            sparsities[i] = config['sparsity']
        axs[2].plot(op_names_sorted, sparsities, label='sparsity: {}, performance: {:.4f}'.format(overall_sparsities[idx], performances_fine_tuned[idx]))
    axs[2].set_title('Sorted by op weights')
    axs[2].legend()

    for ax in axs:
        plt.sca(ax)
        plt.xticks(rotation=90)

    # plt.tight_layout()
    plt.savefig('experiment_data/sparsities_distribution_{}_{}_{}.png'.format(args.model, args.pruner, args.base_algo))
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

    # models = ['vgg16', 'resnet18']
    # pruners = ['SimulatedAnnealingPruner', 'NetAdaptPruner']

    # for model in models:
    #     for pruner in pruners:
    #         args.model = model
    #         args.pruner = pruner
            
    #         plot_sparsities_distribution(args)
    # plot_sparsities_distribution(args)
    
    plot_performance_comparison(args)
