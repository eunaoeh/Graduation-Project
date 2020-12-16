from nested_dict import nested_dict
from functools import partial
import torch
from torch.nn.init import kaiming_normal_
from torch.nn.parallel._functions import Broadcast
from torch.nn.parallel import scatter, parallel_apply, gather
import torch.nn.functional as F
import numpy as np
from kmeans_pytorch import kmeans


def clustering(filename, n, size, batch_size):
    f = open(filename, mode='rt', encoding='utf-8')
    data = f.read()
    data = data.split(', ')
#    stack = data.split('[]')
    stack = np.array([]) 
    i = 0
    tmp = ""
    # del data[-1]
    print(len(data))
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
        
    stack = stack.reshape(int(i/batch_size), batch_size*size)

    
    f.close()
    cluster_ids_x, cluster_centers = kmeans(X=stack, num_clusters=n, distance='euclidean')

    return cluster_ids_x, cluster_centers
 

def distillation(y, teacher_scores, labels, T, alpha):
    p = F.log_softmax(y/T, dim=1)
    q = F.softmax(teacher_scores/T, dim=1)
    # The Kullback-Leibler divergence Loss -> kl_div(input, taget, ...)
    # == 두 확률 분포의 다름의 정도를 설명?
    l_kl = F.kl_div(p, q, size_average=False) * (T**2) / y.shape[0]
    l_ce = F.cross_entropy(y, labels)
    return l_kl * alpha + l_ce * (1. - alpha)


def at(x):
    # print(x.size())
    # print(x.mean(1).size())
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

# def check_max(x):
#     #for 

# x: student, y: teacher
def at_loss(x, y):
    if x.dim() == 2:
        # 학생의 3번째 attention map 계산 
        return (x.cuda() - at(y)).pow(2).mean()
    else:
        return (at(x) - at(y)).pow(2).mean()
    # print(y.size())
    # print(at(y).size())

    # if y.size() == torch.Size([128, 128, 8, 8]):
    #     result = 0
    #     y = y.pow(2).mean(1)
        # for i in range(y.size(0)):

        #     if torch.sum(y[i][1:5, 1:7]) < torch.sum(y[i][3:7, 1:7]):
        #         for j in range(0,8):
        #             for k in range(0,8):
        #                 if j <= 6 and j >= 3 and k >= 1 and k <= 6:
        #                     continue
        #                 else:
        #                     y[i][j][k] = 0
        #     else:
        #         for j in range(0,8):
        #             for k in range(0,8):
        #                 if j <= 4 and j >= 1 and k >= 1 and k <= 6:
        #                     continue
        #                 else:
        #                     y[i][j][k] = 0
        #print(y.size())
        # return (at(x)-F.normalize(y.view(y.size(0), -1))).pow(2).mean()

    # return (at(x) - at(y)).pow(2).mean()


def cast(params, dtype='float'):
    # params 가 dict 이면 if 에 걸려서 dictionary 형태로 param 리턴
    if isinstance(params, dict):
        return {k: cast(v, dtype) for k,v in params.items()}
    else:
        # 아닐 경우 
        return getattr(params.cuda() if torch.cuda.is_available() else params, dtype)()


def conv_params(ni, no, k=1):
    # ni, no ??
    # 레이어의 tensor를 초기화할 때 사용하는 것 
    # kaiming_normal_ return resulting tensor sampled from N(0, std^2) 
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
