import numpy as np
from PIL import Image
from os import makedirs

def cifar10_pngs(): 
    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    size = 1024

    data_arr = []
    for i in range(1,6): 
        c = unpickle(f'./assets/cifar-10-batches-py/data_batch_{i}')
        data_arr.append(np.array(c[b'data']))

    data_arr = np.concatenate(data_arr, axis=0).reshape((-1,3,32,32))

    idx = np.random.choice(data_arr.shape[0], size=size, replace=False)

    data_arr = data_arr[idx]

    for i in range(size): 
        im = Image.fromarray(data_arr[i].transpose((1,2,0)))
        im.save(f"./assets/cifar10_pngs/cifar10_img_{idx[i]}.png")

def gen_pngs(fp): 
    sample_file = np.load(fp + 'samples_0.npz')
    samples = sample_file['samples']
    
    fp += 'gen_imgs/'
    makedirs(fp, exist_ok=True)

    for i in range(samples.shape[0]): 
        im = Image.fromarray(samples[i])
        im.save(f'{fp}img_{i}.png')

fp = './exp/13M/workdir/eval_folder/ckpt_26/'
gen_pngs(fp)