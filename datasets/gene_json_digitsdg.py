import os
import random
import json

def gene_filename(root):
    '''
    为每个domain的category都构建一个文件
    比如mnist -> mnist_train_0.txt, mnist_train_1.txt, ... , mnist_val_0.txt, mnist_val_1.txt, ...
    '''
    domains = ['mnist', 'mnist_m', 'svhn', 'syn']
    modes = ['train', 'val']
    categories = [str(i) for i in range(10)]
    splits = os.path.join(root, 'splits')
    if not os.path.exists(splits):
        os.makedirs(splits)
    for domain in domains:
        for mode in modes:
            category_path = os.path.join(root, domain, mode) # root/mnist/train ...
            for category in categories:
                path_file = os.path.join(splits, domain + '_' + mode + '_' + category + '.txt')
                img_files = sorted(os.listdir(os.path.join(category_path, category)))
                img_files = [os.path.join(category_path, category, img_file) + '\n' for img_file in img_files] # abs path
                with open(path_file, 'a') as f:
                    f.writelines(img_files)

def gene_json(root, target, num, seed):
    '''
    对于splits ('/home/ljw/datasets/digits_dg/splits')中的文件
    如果是train, 则从中任选5或10个作为train_x, 剩下的作为train_u. 如果是val, 则暂时不管它
    构建json文件，最终结果存在ssdg_splits中
    target: 该domain作为target, 剩下的3个domains作为source
    num: 5或10
    seed: 1, 2, 3 ...
    '''
    splits = os.path.join(root, 'splits')
    ssdg_splits = os.path.join(root, 'ssdg_splits')
    if not os.path.exists(ssdg_splits):
        os.makedirs(ssdg_splits)
    path_files = sorted(os.listdir(splits))
    path_files = [os.path.join(splits, path_file) for path_file in path_files]
    domain_lines1, domain_lines2 = [], []
    for path_file in path_files:
        mode, category = path_file[-11: -6], path_file[-5]
        length = 480 # train每一类有480个
        if mode == 'train':
            domain = path_file.split('/')[-1][:-12]
            if domain != target:
                idxs = random.sample(range(length), num)
                with open(path_file, 'r') as f:
                    lines = f.readlines()
                    lines1 = [[lines[i], int(category), domain] for i in idxs]
                    lines2 = [[lines[i], int(category), domain] for i in range(length) if i not in idxs]
                    domain_lines1 += lines1
                    domain_lines2 += lines2
    dict1 = {"train_x": domain_lines1, "train_u": domain_lines2}
    with open(os.path.join(ssdg_splits, target + '_nlab' + str(num * 10) + '_seed' + str(seed) + '.json'), 'w') as f:
        json.dump(dict1, f)


            
    


if __name__ == '__main__':
    root = '/home/ljw/datasets/digits_dg'
    domains = ['mnist', 'mnist_m', 'svhn', 'syn']
    seeds = [1, 2, 3, 4, 5]
    nums = [5, 10]
    # gene_filename(root)
    for domain in domains:
        for num in nums:
            for seed in seeds:
                gene_json(root, domain, num, seed)