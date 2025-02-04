from nested_dict import nested_dict
from functools import partial
import torch
from torch.nn.init import kaiming_normal_
from torch.nn.parallel._functions import Broadcast
from torch.nn.parallel import scatter, parallel_apply, gather
import torch.nn.functional as F
from kmeans_pytorch import kmeans
import numpy as np

def distillation(y, teacher_scores, labels, T, alpha):
    p = F.log_softmax(y/T, dim=1)
    q = F.softmax(teacher_scores/T, dim=1)
    l_kl = F.kl_div(p, q, size_average=False) * (T**2) / y.shape[0]
    l_ce = F.cross_entropy(y, labels)
    return l_kl * alpha + l_ce * (1. - alpha)


def at(x):
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

def clustering(filename, n, size, batch_size):
    f = open(filename, mode='rt', encoding='utf-8')
    data = f.read()
    data = data.split(', ')
#    stack = data.split('[]')
    stack = np.array([]) 
    i = 0
    tmp = ""
    del data[-1]
    print("data : ", )
    for line in data:
        line = line.replace('\n', " ")
        i += 1
        line = line.replace('[', "")
        line = line.replace(']', " ")
        tmp += line
        
        if i % batch_size == 0:
            #tmp = tmp.replace('\n', " ")
            arr = np.fromstring(tmp, dtype=float, sep=' ')
            if stack.size == 0:
                stack = torch.from_numpy(arr)
            else:
                stack = torch.cat([stack,torch.from_numpy(arr)], dim=0)
            tmp = ""
    print(stack.shape)
    stack = stack.reshape(i, size*size)

    
    f.close()
    cluster_ids_x, cluster_centers = kmeans(X=stack, num_clusters=n, distance='euclidean', device=torch.device('cuda:0'))

    return cluster_ids_x, cluster_centers
 
def at_loss(x, y):
    if x.dim() == 2:
        # 학생의 3번째 attention map 계산 
        return (x.cuda() - at(y)).pow(2).mean()
    else:
        x = x.cuda()
        y = y.cuda()
        return (at(x) - at(y)).pow(2).mean()


def cast(params, dtype='float'):
    if isinstance(params, dict):
        return {k: cast(v, dtype) for k,v in params.items()}
    else:
        return getattr(params.cuda() if torch.cuda.is_available() else params, dtype)()


def conv_params(ni, no, k=1):
    return kaiming_normal_(torch.Tensor(no, ni, k, k))


def linear_params(ni, no):
    return {'weight': kaiming_normal_(torch.Tensor(no, ni)), 'bias': torch.zeros(no)}


def bnparams(n):
    return {'weight': torch.rand(n),
            'bias': torch.zeros(n),
            'running_mean': torch.zeros(n),
            'running_var': torch.ones(n)}


def data_parallel(f, input, params, mode, device_ids, output_device=None):
    device_ids = list(device_ids)
    if output_device is None:
        output_device = device_ids[0]

    if len(device_ids) == 1:
        return f(input, params, mode)

    params_all = Broadcast.apply(device_ids, *params.values())
    params_replicas = [{k: params_all[i + j*len(params)] for i, k in enumerate(params.keys())}
                       for j in range(len(device_ids))]

    replicas = [partial(f, params=p, mode=mode)
                for p in params_replicas]
    inputs = scatter([input], device_ids)
    outputs = parallel_apply(replicas, inputs)
    return gather(outputs, output_device)


def flatten(params):
    return {'.'.join(k): v for k, v in nested_dict(params).items_flat() if v is not None}


def batch_norm(x, params, base, mode):
    return F.batch_norm(x, weight=params[base + '.weight'],
                        bias=params[base + '.bias'],
                        running_mean=params[base + '.running_mean'],
                        running_var=params[base + '.running_var'],
                        training=mode)


def print_tensor_dict(params):
    kmax = max(len(key) for key in params.keys())
    for i, (key, v) in enumerate(params.items()):
        print(str(i).ljust(5), key.ljust(kmax + 3), str(tuple(v.shape)).ljust(23), torch.typename(v), v.requires_grad)


def set_requires_grad_except_bn_(params):
    for k, v in params.items():
        if not k.endswith('running_mean') and not k.endswith('running_var'):
            v.requires_grad = True
