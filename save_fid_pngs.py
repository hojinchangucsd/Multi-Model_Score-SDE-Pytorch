import numpy as np
from PIL import Image
import os


SIZE = 50000


def cifar10_pngs(): 
    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    data_arr = []
    for i in range(1,6): 
        c = unpickle(f'./assets/cifar-10-batches-py/data_batch_{i}')
        data_arr.append(np.array(c[b'data']))

    data_arr = np.concatenate(data_arr, axis=0).reshape((-1,3,32,32))

    idx = np.random.choice(data_arr.shape[0], size=SIZE, replace=False)

    data_arr = data_arr[idx]

    for i in range(SIZE): 
        im = Image.fromarray(data_arr[i].transpose((1,2,0)))
        im.save(f"./assets/cifar10_pngs/cifar10_img_{idx[i]}.png")

def gen_pngs(fp): 
    npz_idx, img_idx = 0, 0
    
    while True: 
        sample_file_path = f'{fp}samples_{npz_idx}.npz'
        if not os.path.isfile(sample_file_path): break
        sample_file = np.load(sample_file_path)
        samples = sample_file['samples']
        
        samples_folder = fp + 'gen_imgs/'
        os.makedirs(samples_folder, exist_ok=True)
        
        num_samples = samples.shape[0]
        for k in range(num_samples): 
            if img_idx+k >= SIZE: break

            im = Image.fromarray(samples[k])
            im.save(f'{samples_folder}img_{img_idx+k}.png')
        
        img_idx += num_samples
        npz_idx += 1



# cifar10_pngs()



fp = './exp/13M/workdir/mult_63M_750-250_eval_folder/ckpt_26/'
gen_pngs(fp)
