import requests
import yaml
import json
import time
url = 'https://ne.openpai.org/rest-server/api/v2/jobs'

with open('examples/token.txt', 'r') as file:
    auth_token = file.read()

headers = {'Authorization': 'Bearer ' + auth_token, 'Content-Type': 'text/plain'}

def submit_job(prefix, model, prune_type, sparsity, lr):
    with open("examples/auto_pruners_torch_pai_ne_template.yml", 'r') as stream:
        job_config = yaml.safe_load(stream)

    job_config["taskRoles"]["taskrole"]["commands"] = commands.split('- ')
    job_config["name"] = "{prefix}_{model}_{prune_type}_{sparsity}_{lr}".format(prefix=prefix, model=model, prune_type=prune_type, sparsity=str(sparsity).replace('.', ''), lr=str(lr).replace('.', ''))

    commands = "- ls /mnt/nniblob/yugzhan \
    - mkdir -p /mnt/imagenet \
    - export AZCOPY_CRED_TYPE=\"Anonymous\"; azcopy copy \"https://nennistorage.blob.core.windows.net/nni/yugzhan/imagenet/all/?sv=2019-12-12&ss=b&srt=sco&sp=rwdlacx&se=2020-08-28T14:42:31Z&st=2020-07-27T06:42:31Z&spr=https&sig=EQCe7PQHP9lqflE6PVQpwJpQ9arDStkseBvqm9RakQU%3D\" \"/mnt\" --overwrite=prompt --check-md5 FailIfDifferent --from-to=BlobLocal --blob-type Detect --recursive; export AZCOPY_CRED_TYPE=\"\"; \
    - cd /nni && git fetch && git reset --hard HEAD && git checkout constraint_pruner_tmp && git pull \
    - cd examples/model_compress \
    - python3 -m pip install tensorboard thop \
    - python3 -u constrained_pruner.py \
      --model {model} \
      --type {type} \
      --sparsity {sparsity} \
      --finetune_epochs 20 \
      --data-dir /mnt/all \
      --lr {lr} \
    - cp result.txt /mnt/confignfs-data/znx/{fname}".format(
        model=model, type=prune_type, sparsity=sparsity, lr=lr, fname=job_config["name"])
    

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

    LRs = [0.1, 0.01, 0.001]
    models = ['resnet34', 'mobilenet_v2']
    pruners = ['l1']#'L1FilterPruner', 'SimulatedAnnealingPruner', 'NetAdaptPruner', 'AutoCompressPruner']
    sparsities = ['0.1','0.3','0.5','0.7']
    for model in models:
        for pruner in pruners:
            for sparsity in sparsities:
                for lr in LRs:
                    submit_job('nizhen_0729', model, pruner, sparsity, lr)
                    time.sleep(1)

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
