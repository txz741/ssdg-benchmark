import re
import os
def table_generate(root='/home/ljw/DA_DG/ssdg-benchmark/output_work4/ssdg_digitsdg/nlab_300/MeanTeacher/cnn_digitsdg'):
    domains = os.listdir(root)
    domains.sort()
    data = ''
    for i in range(1, 11):
        # data = ''
        for d in domains:
            file = sorted(os.listdir(os.path.join(root, d, 'seed' + str(i))))
            print(file)
            file = file[0]
            f = open(os.path.join(root, d, 'seed' + str(i), file), 'rb')
            content = f.read()
            result_list = re.findall('accuracy: \d*\.\d*', content.decode('utf-8'))
            result_list = [float(acc[10:-1]) for acc in result_list]
            acc = max(result_list)
            data += str(acc) + '\t'
        data += '\n'
    with open(os.path.join(root, 'acc_table.txt'), 'a') as f:
        root = root.split('/')[-2]
        f.write(root + '\n')
        f.write(data)

if __name__=='__main__':
    table_generate()