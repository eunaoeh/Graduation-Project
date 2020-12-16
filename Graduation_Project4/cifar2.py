import argparse
import os
import json
import numpy as np
from tqdm import tqdm
import torch
from torch.optim import SGD
import torchvision.transforms as T
from torchvision import datasets
import torch.nn.functional as F
import torchnet as tnt
from torchnet.engine import Engine
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import utils

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Wide Residual Networks')
# Model options
parser.add_argument('--depth', default=16, type=int)
parser.add_argument('--width', default=1, type=float)
parser.add_argument('--dataset', default='CIFAR10', type=str)
parser.add_argument('--dataroot', default='.', type=str)
parser.add_argument('--dtype', default='float', type=str)
parser.add_argument('--nthread', default=4, type=int)
parser.add_argument('--teacher_id', default='', type=str)

# Training options
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--weight_decay', default=0.0005, type=float)
parser.add_argument('--epoch_step', default='[60,120,160]', type=str,
                    help='json list with epochs to drop lr on')
parser.add_argument('--lr_decay_ratio', default=0.2, type=float)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--randomcrop_pad', default=4, type=float)
parser.add_argument('--temperature', default=4, type=float)
parser.add_argument('--alpha', default=0, type=float)
parser.add_argument('--beta', default=0, type=float)

# Device options
parser.add_argument('--cuda', default='True', type=bool)
parser.add_argument('--save', default='', type=str,
                    help='save parameters and logs in this folder')
parser.add_argument('--ngpu', default=1, type=int,
                    help='number of GPUs to use for training')
parser.add_argument('--gpu_id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

cluster_idx_0 = []
cluster_centers_0 = []
cluster_idx_1 = []
cluster_centers_1 = []
cluster_idx_2 = []
cluster_centers_2 = []
at_idx = 0
c_epoch = 0

def create_dataset(opt, train):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                    np.array([63.0, 62.1, 66.7]) / 255.0),
    ])
    if train:
        transform = T.Compose([
            T.Pad(4, padding_mode='reflect'),
            T.RandomHorizontalFlip(),
            T.RandomCrop(32),
            transform
        ])
    return getattr(datasets, opt.dataset)(opt.dataroot, train=train, download=True, transform=transform)


def resnet(depth, width, num_classes):
    assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
    n = (depth - 4) // 6
    widths = [int(v * width) for v in (16, 32, 64)]

    def gen_block_params(ni, no):
        return {
            'conv0': utils.conv_params(ni, no, 3),
            'conv1': utils.conv_params(no, no, 3),
            'bn0': utils.bnparams(ni),
            'bn1': utils.bnparams(no),
            'convdim': utils.conv_params(ni, no, 1) if ni != no else None,
        }

    def gen_group_params(ni, no, count):
        return {'block%d' % i: gen_block_params(ni if i == 0 else no, no)
                for i in range(count)}

    flat_params = utils.cast(utils.flatten({
        'conv0': utils.conv_params(3, 16, 3),
        'group0': gen_group_params(16, widths[0], n),
        'group1': gen_group_params(widths[0], widths[1], n),
        'group2': gen_group_params(widths[1], widths[2], n),
        'bn': utils.bnparams(widths[2]),
        'fc': utils.linear_params(widths[2], num_classes),
    }))

    utils.set_requires_grad_except_bn_(flat_params)

    def block(x, params, base, mode, stride):
        o1 = F.relu(utils.batch_norm(x, params, base + '.bn0', mode), inplace=True)
        y = F.conv2d(o1, params[base + '.conv0'], stride=stride, padding=1)
        o2 = F.relu(utils.batch_norm(y, params, base + '.bn1', mode), inplace=True)
        z = F.conv2d(o2, params[base + '.conv1'], stride=1, padding=1)
        if base + '.convdim' in params:
            return z + F.conv2d(o1, params[base + '.convdim'], stride=stride)
        else:
            return z + x

    def group(o, params, base, mode, stride):
        for i in range(n):
            o = block(o, params, f'{base}.block{i}', mode, stride if i == 0 else 1)
        return o

    def f(input, params, mode, base=''):
        def replace(g_s, n, at_idx):
            global cluster_centers_0, cluster_centers_1, cluster_centers_2, cluster_idx_0, cluster_idx_1, cluster_idx_2
            if len(g_s) == 128:
                center = []
                if n == 0:
                    cluster_center = cluster_centers_0
                    cluster_idx = cluster_idx_0
                elif n == 1:
                    cluster_center = cluster_centers_1
                    cluster_idx = cluster_idx_1
                else:
                    cluster_center = cluster_centers_2
                    cluster_idx = cluster_idx_2

                if at_idx >= 391:
                    center = cluster_center[cluster_idx[at_idx-1]]
                else:
                    center = cluster_center[cluster_idx[at_idx]]

                return center

        global epoch, at_idx, c_epoch
        x = F.conv2d(input, params[f'{base}conv0'], padding=1)
        g0 = group(x, params, f'{base}group0', mode, 1)
        g1 = group(g0, params, f'{base}group1', mode, 2)
        print(g1.shape)
        if c_epoch > 1:
            g_1 = replace(g1, 2, at_idx)
            if g_1 != None:
                g1 = g_1

        g2 = group(g1, params, f'{base}group2', mode, 2)
        if c_epoch > 1:
            g_2 = replace(g2, 2, at_idx)
            if g_2 != None:
                g2 = g_2
        print(g2.shape)
        o = F.relu(utils.batch_norm(g2, params, f'{base}bn', mode))
        o = F.avg_pool2d(o, 8, 1, 0)
        o = o.view(o.size(0), -1)
        o = F.linear(o, params[f'{base}fc.weight'], params[f'{base}fc.bias'])
        at_idx += 1

        '''if epoch == 0 and base == 'student.':
            print(g0.shape, g1.shape, g2.shape)
            with open('resnet_16_2_0.txt', 'a') as f:
                for i in range(g0.size(0)):
                    f.write(str(g0[i].pow(2).mean(0).cpu().detach().numpy()))
                    f.write('\n')
            with open('resnet_16_2_1.txt', 'a') as f:
                for i in range(g1.size(0)):
                    f.write(str(g1[i].pow(2).mean(0).cpu().detach().numpy()))
                    f.write('\n')
            with open('resnet_16_2_2.txt', 'a') as f:
                for i in range(g2.size(0)):
                    f.write(str(g2[i].pow(2).mean(0).cpu().detach().numpy()))
                    f.write('\n')'''


        return o, (g0, g1, g2)

    return f, flat_params


def main():
    global epoch
    opt = parser.parse_args()
    print('parsed options:', vars(opt))
    epoch_step = json.loads(opt.epoch_step)
    num_classes = 10 if opt.dataset == 'CIFAR10' else 100

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

    print("cuda : ", torch.cuda.is_available())
    def create_iterator(mode):
        return DataLoader(create_dataset(opt, mode), opt.batch_size, shuffle=mode,
                          num_workers=opt.nthread, pin_memory=torch.cuda.is_available())

    train_loader = create_iterator(True)
    test_loader = create_iterator(False)

    # deal with student first
    f_s, params_s = resnet(opt.depth, opt.width, num_classes)

    # deal with teacher
    if opt.teacher_id:
        with open(os.path.join('logs', opt.teacher_id, 'log.txt'), 'r') as ff:
            line = ff.readline()
            r = line.find('json_stats')
            info = json.loads(line[r + 12:])
        f_t = resnet(info['depth'], info['width'], num_classes)[0]
        model_data = torch.load(os.path.join('logs', opt.teacher_id, 'model.pt7'))
        params_t = model_data['params']

        # merge teacher and student params
        params = {'student.' + k: v for k, v in params_s.items()}
        for k, v in params_t.items():
            params['teacher.' + k] = v.detach().requires_grad_(False)

        def f(inputs, params, mode):
            global c_epoch, at_idx, cluster_idx_0, cluster_idx_1, cluster_idx_2, cluster_centers_1, cluster_centers_2, cluster_centers_0
            y_s, g_s = f_s(inputs, params, mode, 'student.')
            # 처음 epoch에서 학생의 3번째 attention map을 (128, 64)로 변환 후 파일에 저장
            if c_epoch == 1:
                def memo(filename, g_s):
                    if len(g_s) == 128:
                        f = open(filename, 'a')
                        tmp = F.normalize(g_s.pow(2).mean(1).view(g_s.size(0), -1))
                        for i in range(len(tmp)):
                            f.write(str(tmp[i].cpu().detach().numpy()))
                            f.write(', ')

                memo('resnet_16_2_1.txt', g_s[1])
                memo('resnet_16_2_2.txt', g_s[2])

            with torch.no_grad():
                y_t, g_t = f_t(inputs, params, False, 'teacher.')

            # 저장 후 clustering 한 대표값으로 attention map을 대체하고 at_loss 구함 
            if c_epoch > 1:
                '''def replace(g_s, n, at_idx):
                    global cluster_centers_0, cluster_centers_1, cluster_centers_2, cluster_idx_0, cluster_idx_1, cluster_idx_2
                    if len(g_s) == 128:
                        center = []
                        if n == 0:
                            cluster_center = cluster_centers_0
                            cluster_idx = cluster_idx_0
                        elif n == 1:
                            cluster_center = cluster_centers_1
                            cluster_idx = cluster_idx_1
                        else:
                            cluster_center = cluster_centers_2
                            cluster_idx = cluster_idx_2

                        if at_idx >= 391:
                            center = cluster_center[cluster_idx[at_idx-1]]
                        else:
                            center = cluster_center[cluster_idx[at_idx]]

                        return center
                
                g_s_1 = replace(g_s[1], 1, at_idx)
                g_s_2 = replace(g_s[2], 2, at_idx)
                
                if g_s_1 == None:
                    g_s_1 = g_s[1]
                if g_s_2 == None:
                    g_s_2 = g_s[2]

                g_s = (g_s[0], g_s_1, g_s_2)'''
                at_idx += 1

            return y_s, y_t, [utils.at_loss(x, y) for x, y in zip(g_s, g_t)]
    else:
        f, params = f_s, params_s

    def create_optimizer(opt, lr):
        print('creating optimizer with lr = ', lr)
        return SGD((v for v in params.values() if v.requires_grad), lr,
                   momentum=0.9, weight_decay=opt.weight_decay)

    optimizer = create_optimizer(opt, opt.lr)

    epoch = 0
    if opt.resume != '':
        state_dict = torch.load(opt.resume)
        epoch = state_dict['epoch']
        params_tensors = state_dict['params']
        for k, v in params.items():
            v.data.copy_(params_tensors[k])
        optimizer.load_state_dict(state_dict['optimizer'])

    print('\nParameters:')
    utils.print_tensor_dict(params)

    n_parameters = sum(p.numel() for p in list(params_s.values()))
    print('\nTotal number of parameters:', n_parameters)

    meter_loss = tnt.meter.AverageValueMeter()
    classacc = tnt.meter.ClassErrorMeter(accuracy=True)
    timer_train = tnt.meter.TimeMeter('s')
    timer_test = tnt.meter.TimeMeter('s')
    meters_at = [tnt.meter.AverageValueMeter() for i in range(3)]

    if not os.path.exists(opt.save):
        os.mkdir(opt.save)

    def h(sample):
        inputs = utils.cast(sample[0], opt.dtype).detach()
        targets = utils.cast(sample[1], 'long')
        if opt.teacher_id != '':
            y_s, y_t, loss_groups = utils.data_parallel(f, inputs, params, sample[2], range(opt.ngpu))
            loss_groups = [v.sum() for v in loss_groups]
            [m.add(v.item()) for m, v in zip(meters_at, loss_groups)]
            return utils.distillation(y_s, y_t, targets, opt.temperature, opt.alpha) \
                   + opt.beta * sum(loss_groups), y_s
        else:
            y = utils.data_parallel(f, inputs, params, sample[2], range(opt.ngpu))[0]
            return F.cross_entropy(y, targets), y

    def log(t, state):
        torch.save(dict(params={k: v.data for k, v in params.items()},
                        optimizer=state['optimizer'].state_dict(),
                        epoch=t['epoch']),
                   os.path.join(opt.save, 'model.pt7'))
        z = vars(opt).copy(); z.update(t)
        logname = os.path.join(opt.save, 'log.txt')
        with open(logname, 'a') as f:
            f.write('json_stats: ' + json.dumps(z) + '\n')
        print(z)

    def on_sample(state):
        state['sample'].append(state['train'])

    def on_forward(state):
        classacc.add(state['output'].data, state['sample'][1])
        meter_loss.add(state['loss'].item())

    def on_start(state):
        state['epoch'] = epoch

    def on_start_epoch(state):
        global cluster_idx_0, cluster_idx_1, cluster_idx_2, cluster_centers_0, cluster_centers_1, cluster_centers_2, at_idx, c_epoch, epoch
        classacc.reset()
        meter_loss.reset()
        timer_train.reset()
        [meter.reset() for meter in meters_at]
        state['iterator'] = tqdm(train_loader)
        epoch = state['epoch'] + 1
        c_epoch = epoch
        at_idx = 0

        # epoch 1에서 2로 넘어갈 때 clustering 한 후 cluster center값 마스킹 
        if epoch == 2:
            n = 9
            #cluster_idx_0, cluster_centers_0 = utils.clustering("resnet_16_2_0.txt", n, 1024)
            cluster_idx_1, cluster_centers_1 = utils.clustering("resnet_16_2_1.txt", n, 256)
            cluster_idx_2, cluster_centers_2 = utils.clustering("resnet_16_2_2.txt", n, 64)
            #cluster_centers_0 = cluster_centers_0.reshape(n, 128, 1024)
            cluster_centers_1 = cluster_centers_1.reshape(n, 128, 256)
            cluster_centers_2 = cluster_centers_2.reshape(n, 128, 64)
            
            def masking(cluster_centers):
                size = cluster_centers.shape
                for center in cluster_centers:
                    m = 0
                    idx = 0
                    jdx = 0
                    for i in range(len(center)):
                        for j in range(len(center[i])):
                            if center[i][j] > m:
                                m = center[i][j]
                                idx = i
                                jdx = j
                    for i in range(len(center)):
                        if i > idx + (size[1]//4) or i < idx - (size[1]//4):
                            for j in range(len(center[i])):
                                center[i][j] = 0
                
                    for j in range(size[2]):
                        if j > jdx + (size[2]//4) or j < jdx - (size[2]//4):
                            for i in range(128):
                                center[i][j] = 0
                return cluster_centers

            #cluster_centers_0 = masking(cluster_centers_0)
            cluster_centers_1 = masking(cluster_centers_1)
            cluster_centers_2 = masking(cluster_centers_2)
            
            print(cluster_centers_1[0], cluster_centers_1[1])
            print(cluster_centers_2[0], cluster_centers_2[1])

        if epoch in epoch_step:
            lr = state['optimizer'].param_groups[0]['lr']
            state['optimizer'] = create_optimizer(opt, lr * opt.lr_decay_ratio)

    def on_end_epoch(state):
        global cepoch
        train_loss = meter_loss.mean
        train_acc = classacc.value()
        train_time = timer_train.value()
        meter_loss.reset()
        classacc.reset()
        timer_test.reset()
        cepoch = state['epoch']
        engine.test(h, test_loader)

        test_acc = classacc.value()[0]
        print(log({
            "train_loss": train_loss,
            "train_acc": train_acc[0],
            "test_loss": meter_loss.mean,
            "test_acc": test_acc,
            "epoch": state['epoch'],
            "num_classes": num_classes,
            "n_parameters": n_parameters,
            "train_time": train_time,
            "test_time": timer_test.value(),
            "at_losses": [m.value() for m in meters_at],
           }, state))
        print('==> id: %s (%d/%d), test_acc: \33[91m%.2f\033[0m' % \
                       (opt.save, state['epoch'], opt.epochs, test_acc))

    engine = Engine()
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch
    engine.hooks['on_start'] = on_start
    engine.train(h, train_loader, opt.epochs, optimizer)

    


if __name__ == '__main__':
    main()
