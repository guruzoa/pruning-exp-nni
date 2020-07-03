import argparse
import os
import json
import torch
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torchvision import datasets, transforms, models

from models.mnist.lenet import LeNet
from models.cifar10.vgg import VGG
from nni.compression.torch import L1FilterPruner, SimulatedAnnealingPruner, ADMMPruner, NetAdaptPruner, AutoCompressPruner
from nni.compression.torch import ModelSpeedup
from nni.compression.torch.utils.counter import count_flops_params


def get_data(args):
    '''
    get data
    '''
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {
    }

    if args.dataset == 'mnist':
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(args.data_dir, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            datasets.MNIST(args.data_dir, train=False,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
        criterion = torch.nn.NLLLoss()
    elif args.dataset == 'cifar10':
        normalize = transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(args.data_dir, train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(args.data_dir, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False, **kwargs)
        criterion = torch.nn.CrossEntropyLoss()
    elif args.dataset == 'imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(args.data_dir, 'train'),
                                 transform=transforms.Compose([
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     normalize,
                                 ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(args.data_dir, 'val'),
                                 transform=transforms.Compose([
                                     transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     normalize,
                                 ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
        criterion = torch.nn.CrossEntropyLoss()

    return train_loader, val_loader, criterion

def test(model, device, criterion, val_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += criterion(output, target).item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(val_loader.dataset)
    accuracy = correct / len(val_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset), 100. * accuracy))

    return accuracy

def get_dummy_input(args, device):
    if args.dataset == 'mnist':
        dummy_input = torch.randn([args.test_batch_size, 1, 28, 28]).to(device)
    elif args.dataset in ['cifar10', 'imagenet']:
        dummy_input = torch.randn([args.test_batch_size, 3, 32, 32]).to(device)
    return dummy_input

def flops_counter(args):
    # model speed up
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, criterion = get_data(args)
    
    if args.pruner != 'AutoCompressPruner':
        if args.model == 'LeNet':
            model = LeNet().to(device)
        elif args.model == 'vgg16':
            model = VGG(depth=16).to(device)
        elif args.model == 'resnet18':
            model = models.resnet18(pretrained=False, num_classes=10).to(device)
        elif args.model == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained=False).to(device)
        
        def evaluator(model):
            return test(model, device, criterion, val_loader)

        model.load_state_dict(torch.load(os.path.join(args.experiment_data_dir, 'model_fine_tuned.pth')))
        masks_file = os.path.join(args.experiment_data_dir, 'mask.pth')
        
        dummy_input = get_dummy_input(args, device)
        
        m_speedup = ModelSpeedup(model, dummy_input, masks_file, device)
        m_speedup.speedup_model()
        evaluation_result = evaluator(model)
        print('Evaluation result (speed up model): %s' % evaluation_result)

        with open(os.path.join(args.experiment_data_dir, 'performance.json')) as f:
            result = json.load(f)

        result['speedup'] = evaluation_result
        with open(os.path.join(args.experiment_data_dir, 'performance.json'), 'w+') as f:
            json.dump(result, f)

        torch.save(model.state_dict(), os.path.join(args.experiment_data_dir, 'model_speed_up.pth'))
        print('Speed up model saved to %s', args.experiment_data_dir)
    else:
        model = torch.load(os.path.join(args.experiment_data_dir, 'model_fine_tuned.pth'))
        model.eval()
        flops, params = count_flops_params(model, (1,3,32,32))
        with open(os.path.join(args.experiment_data_dir, 'flops.json'), 'w+') as f:
            json.dump({'FLOPS':int(flops), 'params':int(params) }, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--model', type=str, default='vgg16')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset to use, mnist, cifar10 or imagenet')
    parser.add_argument('--data-dir', type=str, default='./data/',
                        help='dataset directory')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64,
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--pruner', type=str, default='SimulatedAnnealingPruner')
    parser.add_argument('--experiment-data-dir', type=str, default='./experiment_data',
                        help='For saving experiment data')
    args = parser.parse_args()

    # pruners = ['L1FilterPruner', 'SimulatedAnnealingPruner', 'NetAdaptPruner']
    pruners = ['AutoCompressPruner']
    sparsities = ['01', '02', '03', '04', '05', '06', '07', '08', '09']

    for pruner in pruners:
        for sparsity in sparsities:
            args.experiment_data_dir = os.path.join('./experiment_data/cifar10/vgg16/', pruner, sparsity)

            flops_counter(args)
