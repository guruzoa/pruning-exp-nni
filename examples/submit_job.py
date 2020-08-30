import requests
import yaml
import json
import time
# url = 'https://rr.openpai.org/rest-server/api/v2/jobs'
url = 'https://ne.openpai.org/rest-server/api/v2/jobs'

with open('examples/token.txt', 'r') as file:
    auth_token = file.read().strip()

headers = {'Authorization': 'Bearer ' + auth_token, 'Content-Type': 'text/plain'}

def submit_job(prefix, model, pruner, sparsity, pretrain_epochs, fine_tune_epochs, dataset='cifar10', lr=0.001, constrained=True, load_checkpoint=True, no_depen=False):
    # with open("examples/auto_pruners_torch_pai_rr_template.yml", 'r') as stream:
    with open("examples/auto_pruners_torch_pai_rr_template.yml", 'r') as stream:
        job_config = yaml.safe_load(stream)

    
    commands = "- apt update \
    - apt install -y nfs-common \
    - cd /root/nni/examples/model_compress && git checkout constraint_pruner_benchmark && git pull\
    - export AZCOPY_CRED_TYPE=\"Anonymous\"; # azcopy copy \"https://nennistorage.blob.core.windows.net/nni/yugzhan/imagenet/all/?sv=2019-12-12&ss=b&srt=sco&sp=rwdlacx&se=2020-08-28T14:42:31Z&st=2020-07-27T06:42:31Z&spr=https&sig=EQCe7PQHP9lqflE6PVQpwJpQ9arDStkseBvqm9RakQU%3D\" \"/mnt\" --overwrite=prompt --check-md5 FailIfDifferent --from-to=BlobLocal --blob-type Detect --recursive; export AZCOPY_CRED_TYPE=\"\"; \
    - pip install tensorboard thop \
    - python -u auto_pruners_torch.py \
      --model {model} \
      --dataset {dataset} \
      --load-pretrained-model {load_checkpoint} \
      --pretrain-epochs {pretrain_epochs} \
      --pretrained-model-dir /mnt/confignfs-data/znx/pretrianed_model/{dataset}/{model}/model_trained.pth \
      --pruner {pruner} --base-algo l1 --cool-down-rate 0.95 \
      --sparsity {sparsity} \
      --speed-up True \
      --lr {lr} \
      --only_no_dependency {no_depen} \
      --fine-tune True --fine-tune-epochs {fine_tune_epochs} \
      --data-dir /mnt/all --constrained {constrained}\
      --experiment-data-dir /mnt/confignfs-data/znx/experiment_data/{dataset}/{model}/{prefix}_{pruner}_{constrained}_nodepen{no_depen}/{sparsity_str}".format(
          dataset=dataset, model=model, pruner=pruner, sparsity=sparsity, sparsity_str=str(sparsity).replace('.', ''), pretrain_epochs=pretrain_epochs, fine_tune_epochs=fine_tune_epochs, constrained=str(constrained), prefix=prefix, lr=lr, load_checkpoint=load_checkpoint, no_depen=no_depen)
    
    job_config["taskRoles"]["taskrole"]["commands"] = commands.split('- ')
    job_config["name"] = "{prefix}_{model}_{dataset}_{pruner}_{sparsity_str}_cons{constrained}_nodepen{no_depen}".format(prefix=prefix, dataset=dataset, model=model, pruner=pruner, sparsity=sparsity, sparsity_str=str(sparsity).replace('.', ''), constrained=str(constrained), no_depen=no_depen)

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

    # models = ['resnet18', 'resnet50']
    models = ['mobilenet_v2', 'resnet18', 'resnet50']
    # pruners = ['AutoCompressPruner' ]
    pruners = ['L1FilterPruner' ]
    # pruners = ['AttentionPruner' ]
    # pruners = [ 'SimulatedAnnealingPruner']
    LRs = [0.001]
    # sparsities = ['0.1', '0.3', '0.5' ,'0.7' ,'0.9', '0.95', '0.975', '0.98', '0.99', '0.993', '0.995', '0.997', '0.999']
    sparsities = ['0.1', '0.2','0.3', '0.4', '0.5', '0.6' ,'0.7', '0.8' ,'0.9', '0.92', '0.93', '0.95' , '0.96', '0.97', '0.98']
    # sparsities = [ '0.98', '0.99', '0.993', '0.995', '0.997', '0.999']
    for model in models:
        for pruner in pruners:
            for sparsity in sparsities:
                for no_depen in [True]: 
                    for lr in LRs:
                        for constrained in [False]:
                            # submit_job('0812', model, pruner, sparsity, pretrain_epochs=0, fine_tune_epochs=20, dataset='imagenet', lr=lr, constrained=constrained, load_checkpoint=False)
                            submit_job('0829', model, pruner, sparsity, pretrain_epochs=0, fine_tune_epochs=200, dataset='cifar10', load_checkpoint=True, constrained=constrained, no_depen=no_depen)