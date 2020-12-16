"""
    PyTorch training code for
    "Paying More Attention to Attention: Improving the Performance of
                Convolutional Neural Networks via Attention Transfer"
    https://arxiv.org/abs/1612.03928
    
    This file includes:
     * CIFAR ResNet and Wide ResNet training code which exactly reproduces
       https://github.com/szagoruyko/wide-residual-networks
     * Activation-based attention transfer
     * Knowledge distillation implementation

    2017 Sergey Zagoruyko
"""
from PIL import Image
import pickle
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
from torch.utils.data import DataLoader, Dataset
import torch.backends.cudnn as cudnn
import utils
import torchvision
from torchsummary import summary
from collections import OrderedDict

cudnn.benchmark = True
cluster_idx_0 = []
cluster_centers_0 = []
cluster_idx_1 = []
cluster_centers_1 = []
cluster_idx_2 = []
cluster_centers_2 = []
at_idx = 0
c_epoch = 0

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
parser.add_argument('--cuda')
parser.add_argument('--save', default='', type=str,
                    help='save parameters and logs in this folder')
parser.add_argument('--ngpu', default=1, type=int,
                    help='number of GPUs to use for training')
parser.add_argument('--gpu_id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

opt = parser.parse_args()

test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                    np.array([63.0, 62.1, 66.7]) / 255.0),
    ])
train_transform = T.Compose([
            T.Pad(4, padding_mode='reflect'),
            T.RandomHorizontalFlip(),
            T.RandomCrop(32),
            test_transform
        ])

train_data = datasets.CIFAR10(root="~/.data",
                              train=True,
                              download=(not os.path.exists("~/.data")),
                              transform=train_transform)


test_data = datasets.CIFAR10(root="~/.data",
                             train=False,
                             download=(not os.path.exists("~/.data")),
                             transform=test_transform)


class CustomDataset(torchvision.datasets.CIFAR10):
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f']
        # ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        # ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        # ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        # ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):

        super(CustomDataset, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()
    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        # if not check_integrity(path, self.meta['md5']):
        #     raise RuntimeError('Dataset metadata file not found or corrupted.' +
        #                        ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            # if not check_integrity(fpath, md5):
            #     return False
        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")



def create_dataset(opt, train):
    # Transforms are image transformations. 
    # Composes several transforms together
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
    # getattr(object, 'x') == object.x 즉 object의 x() 를 실행하는 것과 같다
    # torchvision.datasets?? object는 어딨지 
    return getattr(datasets, opt.dataset)(opt.dataroot, train=train, download=True, transform=transform)


def resnet(depth, width, num_classes):
    global cluster_centers_0, cluster_centers_1, cluster_centers_2, cluster_idx_0, cluster_idx_1, cluster_idx_2
    global c_epoch, at_idx
    assert (depth - 4) % 6 == 0, 'depth should be 6n+4'

    # n 은 언제 사용? depth = 40인 경우는 6
    n = (depth - 4) // 6

    # 만약 width = 1 [16,32,64] --> 세 가지 경우를 다 확인하는 건가..?
    widths = [int(v * width) for v in (16, 32, 64)]

    # ni, no에 들어가는 값이 무엇인지 확인
    def gen_block_params(ni, no):
        return {
            'conv0': utils.conv_params(ni, no, 3),
            'conv1': utils.conv_params(no, no, 3),
            'bn0': utils.bnparams(ni), # weight, bias, mean, var를 dict형태로 리턴
            'bn1': utils.bnparams(no),
            'convdim': utils.conv_params(ni, no, 1) if ni != no else None,
        }

    def gen_group_params(ni, no, count):
        return {'block%d' % i: gen_block_params(ni if i == 0 else no, no)
                for i in range(count)}
    # 16, 
    
    flat_params = utils.cast(utils.flatten({
        'conv0': utils.conv_params(3, 16, 3), # 얘는 그냥 초기화
        'group0': gen_group_params(16, widths[0], n), # 16, 16, 6 일 경우 블록 6개를 dict형태로 리턴..?
        'group1': gen_group_params(widths[0], widths[1], n),
        'group2': gen_group_params(widths[1], widths[2], n),
        'bn': utils.bnparams(widths[2]),
        'fc': utils.linear_params(widths[2], num_classes),
    }))

    # grad 할건지??  bn만 빼고 다 true로 변경
    # bn 은 batch normalization ?
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
    
    # attention map => g0, g1, g2
    def f(input, params, mode, base=''):
        global c_epoch, at_idx
        x = F.conv2d(input, params[f'{base}conv0'], padding=1)
        g0 = group(x, params, f'{base}group0', mode, 1) # 위에 그룹함수 있음...ㅎ
        g1 = group(g0, params, f'{base}group1', mode, 2)
        g2 = group(g1, params, f'{base}group2', mode, 2)
        o = F.relu(utils.batch_norm(g2, params, f'{base}bn', mode))
        o = F.avg_pool2d(o, 8, 1, 0)
        o = o.view(o.size(0), -1) # view는 reshape 할 때 사용
        # 마지막 레이어만 unfreeze 하고 다시 학습 시키는 것이 fine tuning
        o = F.linear(o, params[f'{base}fc.weight'], params[f'{base}fc.bias']) # linear transform == x*A.T + b 적용

        return o, (g0, g1, g2)

    return f, flat_params

def main():
    string = ""
    # 필요한 인자들 넣어서 파싱하는 거?
    opt = parser.parse_args()
    
    print(torch.cuda.device(0))
    print(torch.cuda.is_available())
    print('parsed options:', vars(opt))

    epoch_step = json.loads(opt.epoch_step)
    num_classes = 10 if opt.dataset == 'CIFAR10' else 100

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

    # Data 받아오기
    def create_iterator(data, mode):
        return DataLoader(data, opt.batch_size, shuffle=mode,
                          num_workers=opt.nthread, pin_memory=torch.cuda.is_available())
   
    # train_loader = create_iterator(train_data, True) # True 이면 train 아니면 test 해서 필요한? 셋?? 설정
    test_loader = create_iterator(test_data, False)
    # train_loader = create_iterator(train_data, True)
    train_loader = create_iterator(CustomDataset('~/.data', transform=train_transform), True)
    # print(len(CustomDataset('~/.data', transform=train_transform)))

    # 필요한? 인자들 설정하는 것 같음
    # deal with student first
    f_s, params_s = resnet(opt.depth, opt.width, num_classes)
    # deal with teacher
    # 첫번 째 경우가 티쳐 아이디가 존재하는 경우임 -> 즉 학생이 학습하는 경우
    if opt.teacher_id and opt.teacher_id != 'at_16_1_16_2':
        # student를 train 하는 경우? teacher_id에 맞는 걸 불러와서 사용
        with open(os.path.join('logs', opt.teacher_id, 'log.txt'), 'r') as ff:
            line = ff.readline()
            r = line.find('json_stats')
            info = json.loads(line[r + 12:]) # json file decoding
        f_t = resnet(info['depth'], info['width'], num_classes)[0]
        model_data = torch.load(os.path.join('logs', opt.teacher_id, 'model.pt7')) # 모델 클래스 불러오기
        params_t = model_data['params'] # teacher의 param 불러오기?
        
        new_params = OrderedDict()
        param_s = torch.load('./logs/at+clustering+before+ft2/model.pt7')['params']
        for k, v in param_s.items():
            if 'teacher' in k:
                break
            
            new_params[k] = v
            if 'fc' in k:
                new_params[k] = v.requires_grad_(True)
            else:
                new_params[k] = v.detach().requires_grad_(False)
        params = new_params
        
        # merge teacher and student params
        # params = {'student.' + k: v for k, v in params_s.items()}
        for k, v in params_t.items():
            params['teacher.' + k] = v.detach().requires_grad_(False)

        # attention map loss 구함
        # f_s 는 resnet의 f 함수
        # y_s, y_t 의 네트워크의 결과값? == o
        # for AT??
        
        def f(inputs, params, mode):
            global c_epoch, at_idx, cluster_idx_0, cluster_idx_1, cluster_idx_2, cluster_centers_1, cluster_centers_2, cluster_centers_0
            
            y_s, g_s = f_s(inputs, params, mode, 'student.')
            
            if c_epoch == 1:
                def memo(filename, g_s):
                    if len(g_s) == opt.batch_size:
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
            if 1 < c_epoch and c_epoch <= opt.epochs:
                def replace(g_s, n, at_idx):
                    global cluster_centers_0, cluster_centers_1, cluster_centers_2, cluster_idx_0, cluster_idx_1, cluster_idx_2
                    if len(g_s) == opt.batch_size:
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
                            
                            # idx = 0
                            # if 50000 % opt.batch_size == 0:
                            #     idx = 50000 // opt.batch_size
                            # else:
                            #     idx = 50000 // opt.batch_size + 1
                        
                        if at_idx >= 79:
                            center = cluster_center[cluster_idx[at_idx-1]]
                        else:
                            center = cluster_center[cluster_idx[at_idx]]
                        
                        # center = cluster_center[int(cluster_idx[at_idx])]

                        return center
                
                g_s_1 = replace(g_s[1], 1, at_idx)
                g_s_2 = replace(g_s[2], 2, at_idx)
                
                if g_s_1 == None:
                    g_s_1 = g_s[1]
                if g_s_2 == None:
                    g_s_2 = g_s[2]

                g_s = (g_s[0], g_s_1, g_s_2)
                at_idx += 1
                # g_s_1 = torch.empty(128, 32, 16, 16, dtype=torch.float)
                
                # g_s_2 = torch.empty(128, 64, 8, 8, dtype=torch.float)
                
                # if len(g_s[1]) == opt.batch_size:
                #     for i in range(at_idx, at_idx + opt.batch_size):
                #         g_s_1[i - at_idx] = replace(g_s[1], 1, i)
                # else:
                #     g_s_1 = g_s[1]

                # if len(g_s[2]) == opt.batch_size:
                #     for i in range(at_idx, at_idx + opt.batch_size):
                #         g_s_2[i - at_idx] = replace(g_s[2], 2, i)
                #     # print(at_idx, cluster_idx_2[at_idx])
                #     # print(g_s[2][0].mean(0))
                #     # print(replace(g_s[2], 2, at_idx))
                #     # print("------------------------------")

                # else:
                #     g_s_2 = g_s[2]
                

                g_s = (g_s[0], g_s_1, g_s_2)
                # if len(g_s[1]) == opt.batch_size:
                #     at_idx += len(g_s_1)    
            return y_s, y_t, [utils.at_loss(x, y) for x, y in zip(g_s, g_t)]
    else:
        f, params = f_s, params_s
    
    
    def create_optimizer(opt, lr):
        print('creating optimizer with lr = ', lr)
        return SGD((v for v in params.values() if v.requires_grad), lr,
                   momentum=0.9, weight_decay=opt.weight_decay)
    
    # optimizer 설정
    optimizer = create_optimizer(opt, opt.lr)
    
    epoch = 0
    if opt.resume != '':
        print("RESUME!!")
        state_dict = torch.load(opt.resume)
        epoch = state_dict['epoch']
        params_tensors = state_dict['params']
        for k, v in params.items():
            v.data.copy_(params_tensors[k])
        optimizer.load_state_dict(state_dict['optimizer'])


    print('\nParameters:')
    utils.print_tensor_dict(params)

    n_parameters = sum(p.numel() for p in list(params_s.values()))
    # print('\nTotal number of parameters:', n_parameters)

    # Meters provide a way to keep track of important statistics in an online manner
    # TNT provides dataloading, logging, visualization
    meter_loss = tnt.meter.AverageValueMeter() 
    classacc = tnt.meter.ClassErrorMeter(accuracy=True) 
    timer_train = tnt.meter.TimeMeter('s')
    timer_test = tnt.meter.TimeMeter('s')
    meters_at = [tnt.meter.AverageValueMeter() for i in range(3)]

    if not os.path.exists(opt.save):
        os.mkdir(opt.save)

    # 여기서 h 사용!
    def h(sample):
        # 아마 여기서 freeze  
        inputs = utils.cast(sample[0], opt.dtype).detach()  # train img
        targets = utils.cast(sample[1], 'long') # label
        if opt.teacher_id != '':
            # for distillation?
            # loss groups?
            y_s, y_t, loss_groups = utils.data_parallel(f, inputs, params, sample[2], range(opt.ngpu))
            loss_groups = [v.sum() for v in loss_groups]
            [m.add(v.item()) for m, v in zip(meters_at, loss_groups)]
            return utils.distillation(y_s, y_t, targets, opt.temperature, opt.alpha) \
                   + opt.beta * sum(loss_groups), y_s
        else:
            y, g = utils.data_parallel(f, inputs, params, sample[2], range(opt.ngpu))
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

        
        if state['epoch'] == opt.epochs:
            with open(os.path.join(opt.save, "last_result.txt"), "a") as f:
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
        # 얘가 제일 먼저 시작함!
        global cluster_idx_0, cluster_idx_1, cluster_idx_2, cluster_centers_0, cluster_centers_1, cluster_centers_2, at_idx, c_epoch, epoch
        
        classacc.reset()
        meter_loss.reset()
        timer_train.reset()
        [meter.reset() for meter in meters_at]
        state['iterator'] = tqdm(train_loader)
        # print(state['iterator'])
        epoch = state['epoch'] + 1
        c_epoch = epoch
        at_idx = 0

        
        # epoch 1에서 2로 넘어갈 때 clustering 한 후 cluster center값 마스킹 
        if epoch == 2:
            n = 9
            #cluster_idx_0, cluster_centers_0 = utils.clustering("resnet_16_2_0.txt", n, 1024)
            cluster_idx_1, cluster_centers_1 = utils.clustering("resnet_16_2_1.txt", n, 256, opt.batch_size)
            cluster_idx_2, cluster_centers_2 = utils.clustering("resnet_16_2_2.txt", n, 64, opt.batch_size)
            #cluster_centers_0 = cluster_centers_0.reshape(n, 128, 1024)
            cluster_centers_1 = cluster_centers_1.reshape(n, opt.batch_size, 256)
            cluster_centers_2 = cluster_centers_2.reshape(n, opt.batch_size, 64)
            
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
                            for i in range(size[1]):
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
        train_loss = meter_loss.mean
        train_acc = classacc.value()
        train_time = timer_train.value()
        meter_loss.reset()
        classacc.reset()
        timer_test.reset()

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
        



    # training loop 를 wrap 하는 것이 engine
    # hook 은 backward hook 를 저장할 때 사용
    engine = Engine()
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch
    engine.hooks['on_start'] = on_start


    # summary(h, (3, 32, 32))
    engine.train(h, train_loader, opt.epochs, optimizer) # training start


if __name__ == '__main__':
    repeat = 30
    for _ in range(repeat):
        cur_dir = os.getcwd()
        os.remove(os.path.join(cur_dir, 'resnet_16_2_1.txt'))
        os.remove(os.path.join(cur_dir, 'resnet_16_2_2.txt'))
        main()

