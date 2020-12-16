import torch
import sys
import numpy as np
from kmeans_pytorch import kmeans

def clustering(filename):
    f = open(filename, mode='rt', encoding='utf-8')
    data = f.read()
    data = data.split('\n')
#    stack = data.split('[]')
    stack = np.array([]) 
    i = 0
    tmp = ""
    print(len(data))
    for line in data:
#        line = f.readline()

#        if not line:
#            break

        i += 1
        line = line.replace('[', "")
        line = line.replace(']', "")
        tmp += line

        if i % 16 == 0:
            tmp = tmp.replace('\n', " ")
            arr = np.fromstring(tmp, dtype=float, sep=' ')
            if stack.size == 0:
                stack = torch.from_numpy(arr)
            else:
                stack = torch.cat([stack,torch.from_numpy(arr)], dim=0)
            tmp = ""
    stack = stack.reshape(int(i/16), 64)
#    stack = torch.from_numpy(np.asarray(stack))

    f.close()
    n = 9
    cluster_ids_x, cluster_centers = kmeans(X=stack, num_clusters=n, distance='euclidean', device=torch.device('cuda:0'))
    
    print(cluster_ids_x.shape)
    print(cluster_ids_x)
    '''data = np.empty((0, 8, 8), float)
    for center in cluster_centers:
        arr = np.empty((0, 8), float)
        tmp = np.array([])
        for i in range(64):
            tmp = np.append(tmp, float(center[i]))
            if i % 8 == 7:
                tmp = tmp.reshape(1, 8)
                arr = np.append(arr, tmp)
                tmp = np.array([])

        arr = arr.reshape(1, 8, 8)
        data = np.append(data, arr)


    data = data.reshape(n, 8, 8)'''
    
    data = "["
    for c, center in enumerate(cluster_centers):
        arr = "["
        tmp = "["
        for i in range(64):
            tmp += str(float(center[i]))
            if i % 8 == 7:
                tmp += "]"
                arr += tmp
                if i < 63:
                    arr += ",\n"
                tmp = "["
            else:
                tmp += ","
        arr += "]"
        data += arr
        if c != len(cluster_centers) - 1:
            data += ",\n"
        else:
            data += "]"


    f = open('cluster_centers'+str(n)+'.txt', mode='wt', encoding='utf-8')
    f.write(data)
    
    #print(data)


if __name__ == '__main__':
    clustering('resnet_16_2.txt')



