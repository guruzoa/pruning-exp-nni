import requests
import yaml
import json
import time
# url = 'https://rr.openpai.org/rest-server/api/v2/jobs'
url = 'https://ne.openpai.org/rest-server/api/v2/jobs'

with open('examples/token.txt', 'r') as file:
    auth_token = file.read().strip()

headers = {'Authorization': 'Bearer ' + auth_token, 'Content-Type': 'text/plain'}

def submit_job(prefix, model, pruner, sparsity, pretrain_epochs, fine_tune_epochs, dataset='cifar10', constrained=True):
    # with open("examples/auto_pruners_torch_pai_rr_template.yml", 'r') as stream:
    with open("examples/auto_pruners_torch_pai_rr_template.yml", 'r') as stream:
        job_config = yaml.safe_load(stream)

    
    commands = "- apt update \
    - apt install -y nfs-common \
    - cd /nni/examples/model_compress && git pull\
    - export AZCOPY_CRED_TYPE=\"Anonymous\"; #azcopy copy \"https://nennistorage.blob.core.windows.net/nni/yugzhan/imagenet/all/?sv=2019-12-12&ss=b&srt=sco&sp=rwdlacx&se=2020-08-28T14:42:31Z&st=2020-07-27T06:42:31Z&spr=https&sig=EQCe7PQHP9lqflE6PVQpwJpQ9arDStkseBvqm9RakQU%3D\" \"/mnt\" --overwrite=prompt --check-md5 FailIfDifferent --from-to=BlobLocal --blob-type Detect --recursive; export AZCOPY_CRED_TYPE=\"\"; \
    - python3 -m pip install tensorboard thop \
    - python3 -u auto_pruners_torch.py \
      --model {model} \
      --dataset {dataset} \
      --load-pretrained-model True \
      --pretrain-epochs {pretrain_epochs} \
      --pretrained-model-dir /mnt/confignfs-data/znx/pretrianed_model/{dataset}/{model}/model_trained.pth \
      --pruner {pruner} --base-algo l1 --cool-down-rate 0.9 \
      --sparsity {sparsity} \
      --speed-up True \
      --fine-tune True --fine-tune-epochs {fine_tune_epochs} \
      --data-dir /mnt/all --constrained {constrained}\
      --experiment-data-dir /mnt/confignfs-data/znx/experiment_data/{dataset}/{model}/{prefix}_{pruner}_{constrained}/{sparsity_str}".format(
          dataset=dataset, model=model, pruner=pruner, sparsity=sparsity, sparsity_str=str(sparsity).replace('.', ''), pretrain_epochs=pretrain_epochs, fine_tune_epochs=fine_tune_epochs, constrained=str(constrained), prefix=prefix)
    
    job_config["taskRoles"]["taskrole"]["commands"] = commands.split('- ')
    job_config["name"] = "{prefix}_{model}_{dataset}_{pruner}_{sparsity_str}_{constrained}".format(prefix=prefix, dataset=dataset, model=model, pruner=pruner, sparsity=sparsity, sparsity_str=str(sparsity).replace('.', ''), constrained=str(constrained))

    print(yaml.dump(job_config))
    r = requests.post(url, headers=headers, data=yaml.dump(job_config))
    print(r.text)

if __name__ == '__main__':

    # models = ['resnet18']
    # pruners = [ 'SimulatedAnnealingPruner']
    # sparsities = ['0.1']
    # for model in models:
    #     for pruner in pruners:
    #         for sparsity in sparsities:
    #             submit_job('tdsmp', model, pruner, sparsity, pretrain_epochs=200, fine_tune_epochs=200)

    models = ['resnet18']
    pruners = ['AutoCompressPruner', 'SimulatedAnnealingPruner']
    sparsities = ['0.1', '0.3', '0.5' ,'0.7' ,'0.9', '0.95', '0.975']
    for model in models:
        for pruner in pruners:
            for sparsity in sparsities:
                submit_job('newimage', model, pruner, sparsity, pretrain_epochs=0, fine_tune_epochs=15, dataset='cifar10')