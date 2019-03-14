import train
import test
import evaluate_gpu
import yaml
with open('result.yml', 'r') as stream:
    result = yaml.load(stream)

# args :
# backbone=('resnet', 'resnetmid', 'dense')
# loss = ('softmax', 'arcface', 'cosface', 'sphere', 'center')
# dataset = ('market', 'duke', 'cuhk03')
# optimizer = ('SGD', 'ADAM')
# scheduler_type=('step', 'one_cycle')
# three type of lr and weight
# weight_cent=1, lr=0.05, weight_lr=0.1
# embedding size embedding=512

# metric scale!!!!!!!!
# for arc  scale=30, margin=0.01 default.
# cosine  scale=30, margin =0.4 default
# sphere scale =4  no margin
# balanced sampled

epochs = 60
# lr = 0.02
optimizer='SGD'
backbone='resnet'
ids = '0'
balanced_sample=True

# arc_margin = [0.4]
# test_list = ['9', '19', '29', '39', '49', '59', 'last']

# test_list = ['last']
###########################################################3
# iteration the lr 

# lrs = [1e-1, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
# for i, lr in enumerate(lrs):
#     name = backbone+optimizer+'_lr'+str(lr)+'_epochs'+str(epochs)
#     # train.main(ids = ids, name=name, balanced_sample=balanced_sample,
#         # lr=lr, arc=use_arc, epochs=epochs)
#     # # train.main(id=ids, name='resnet_adam')
#     # 
#     # test.main(ids=ids, name=name, which_epoch='last', arc=use_arc)
#     # evaluate_gpu.main(name=name)
#     # f_name = name + test_epoch
#     result[name] = evaluate_gpu.main(name=name)
    
    
##############################################################]
# iteration weight_decay
weight_decays = [5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
lr = 5e-2
for i,  wd in enumerate(weight_decays):
    name = backbone+optimizer+'_lr'+str(lr)+'_weight_decay'+str(wd)+'_epochs'+str(epochs)
    train.main(ids = ids, name=name, balanced_sample=balanced_sample,
        lr=lr, epochs=epochs)
    # #train.main(id=ids, name='resnet_adam')
    # 
    test.main(ids=ids, name=name, use_dense=, which_epoch='last')
    # # evaluate_gpu.main(name=name)
    # f_name = name + test_epoch
    # result[f_name] = evaluate_gpu.main(name=name)
    result[name] = evaluate_gpu.main(name=name)


with open('result.yml', 'w') as outfile:
    yaml.dump(result, outfile, default_flow_style=False)
