import requests
import yaml
import json

url = 'https://ne.openpai.org/rest-server/api/v2/jobs'

with open('examples/token.txt', 'r') as file:
    auth_token = file.read()

headers = {'Authorization': 'Bearer ' + auth_token, 'Content-Type': 'text/plain'}

def submit_job(prefix, model, prune_type, sparsity):
    with open("examples/auto_pruners_torch_pai_ne_template.yml", 'r') as stream:
        job_config = yaml.safe_load(stream)

    commands = "- ls /mnt/nniblob/yugzhan \
    - cd /nni && git fetch && git reset --hard HEAD && git checkout constraint_pruner_tmp \
    - cd examples/model_compress \
    - python3 -m pip install tensorboard thop \
    - python3 constrained_pruner.py \
      --model {model} \
      --pruner {type} \
      --sparsity {sparsity} \
      --finetune_epochs 20 \
      --data-dir /mnt/nniblob/yugzhan/imagenet/all".format(
        model=model, type=prune_type, sparsity=sparsity)
    
    job_config["taskRoles"]["taskrole"]["commands"] = commands.split('- ')
    job_config["name"] = "{prefix}_{model}_{prune_type}_{sparsity}".format(prefix=prefix, model=model, prune_type=prune_type, sparsity=str(sparsity).replace('.', ''))

    print(yaml.dump(job_config))
    r = requests.post(url, headers=headers, data=yaml.dump(job_config))
    print(r.text)


if __name__ == '__main__':
    # models = ['vgg16']
    # pruners = ['ActivationMeanRankFilterPruner', 'ActivationAPoZRankFilterPruner']#'L1FilterPruner', 'SimulatedAnnealingPruner', 'NetAdaptPruner', 'AutoCompressPruner']
    # sparsities = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '0.95', '0.975']
    # for model in models:
    #     for pruner in pruners:
    #         for sparsity in sparsities:
    #             submit_job('0723', model, pruner, sparsity, pretrain_epochs=100, fine_tune_epochs=100)

    models = ['resnet18']
    pruners = ['l1']#'L1FilterPruner', 'SimulatedAnnealingPruner', 'NetAdaptPruner', 'AutoCompressPruner']
    sparsities = ['0.1']
    for model in models:
        for pruner in pruners:
            for sparsity in sparsities:
                submit_job('tmp7', model, pruner, sparsity)

    # models = ['resnet50']
    # pruners = ['L1FilterPruner']
    # sparsities = ['0.1', '0.975']
    # for model in models:
    #     for pruner in pruners:
    #         for sparsity in sparsities:
    #             submit_job('0723', model, pruner, sparsity, fine_tune_epochs=200)
    
    # # prepare pretrained models
    # models = ['resnet18', 'vgg16']
    # pruners = ['L1FilterPruner']
    # sparsities = ['0.1']
    # for model in models:
    #     for pruner in pruners:
    #         for sparsity in sparsities:
    #             submit_job(model, pruner, sparsity)
