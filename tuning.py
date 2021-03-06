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
# sphere margin =4  no scale
# balanced sampled

epochs = 60
# lr = 0.02
optimizer='SGD'

#loss = 'sphere'
#margin = 4
#scale = 1
# loss = 'cosface'
# scale = 30
# margin = 0.4

loss = 'arcface'
scale = 30
margin = 0.01

# loss = 'center'
# margin = 1
# scale = 1


# backbone='resnet'
# backbone='resnetmid'
backbone='dense'
ids = '3'
dataset='market'
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
# weight_decays = [5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6]

# lr = 1e-2
# weight_decays=[5e-4]
# epochs = 60
# for i,  wd in enumerate(weight_decays):
#     name = dataset+'_'+backbone+'_'+optimizer+loss+'_lr'+str(lr)+'_weight_decay'+str(wd)+'_epochs'+str(epochs)
#     train.main(ids = ids, name=name, balanced_sample=balanced_sample,
#         backbone=backbone, loss=loss, weight_decay=wd,
#         dataset=dataset, lr=lr, epochs=epochs,
#         margin=margin, scale=scale)
#     # #train.main(id=ids, name='resnet_adam')
#     # 
#     test.main(ids=ids, name=name,  which_epoch='last', backbone=backbone)
#     # # evaluate_gpu.main(name=name)
#     # f_name = name + test_epoch
#     # result[f_name] = evaluate_gpu.main(name=name)
#     result[name] = evaluate_gpu.main(name=name)

lr = 1e-2
weight_decays=[5e-4]
epochs = 60
weight_lrs = [10, 5, 1 , 0.5 , 0.01]
for i, wl in enumerate(weight_lrs):
    print('{}: th weight_lr: {}'.format(i, wl))
    name = dataset+'_'+backbone+'_'+optimizer+loss+'_lr'+str(lr)+ \
        '_weight_decay'+str(weight_decays[0])+'_weight_lr'+str(wl)+ \
        '_epochs'+str(epochs)

    train.main(ids = ids, name=name, balanced_sample=balanced_sample,
        backbone=backbone, loss=loss, weight_decay=weight_decays[0],
        dataset=dataset, lr=lr, weight_lr=wl, epochs=epochs,
        margin=margin, scale=scale)
    test.main(ids=ids, name=name,  which_epoch='last', backbone=backbone)
    result[name] = evaluate_gpu.main(name=name)
with open('result.yml', 'w') as outfile:
    yaml.dump(result, outfile, default_flow_style=False)
