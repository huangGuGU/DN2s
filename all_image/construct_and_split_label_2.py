import sys
import os
import random

if __name__ == '__main__':

    # 数据集名称
    image_folder = r'image'
    label_folder =  r'label'

    # 标签文件名称
    all_sample_txt = 'label.txt'

    # 训练集划分
    train_ratio = 0.8
    k = 10

    # 打标签模式
    random_mode = 'all'
    if random_mode == 'k':
        all_random = False
        k_random = True
    elif random_mode == 'all':
        all_random = True
        k_random = False
    else:
        print('Error! Please set random_mode file or all')
        sys.exit()

    input_folderList = os.listdir(image_folder)
    try:
        input_folderList.remove('.DS_Store')
    except:
        pass
    label_imgList = os.listdir(label_folder)

    print(f'the number of labels is: {len(label_imgList)}.')

    sample_all_dict = {}

    for folder in input_folderList:
        sample_list = []
        img_list = os.listdir(os.path.join(image_folder, folder))
        print(f'the number of {folder} is: {len(img_list)}.')

        for imgs_fn in img_list[:20000]:

            sample_list.append('{} {}\n'.format(os.path.join(image_folder, folder, imgs_fn),
                                                    os.path.join(label_folder, folder,imgs_fn)))
        sample_all_dict[f'{folder}'] = sample_list

    with open(all_sample_txt, 'w') as f:
        for _, sample_list in sample_all_dict.items():
            f.writelines(sample_list)


    # 全部随机
    if all_random == True:
        train_list = []
        test_list = []
        sapmle_len = len(list(sample_all_dict.values())[0])
        all_sapmle_len = len(list(range(sapmle_len * len(list(sample_all_dict.values())))))
        index_list = list(range(all_sapmle_len))
        random.shuffle(index_list)
        all_sample_list = []

        for i in range(len(list(sample_all_dict.values()))):
            all_sample_list.extend(list(sample_all_dict.values())[i])

        train_index = index_list[:int(all_sapmle_len * train_ratio)]
        test_index = index_list[int(all_sapmle_len * train_ratio):]

        train_list.extend([all_sample_list[idx] for idx in train_index])
        test_list.extend([all_sample_list[idx] for idx in test_index])

        with open('label_train.txt', 'w') as f:
            f.writelines(train_list)

        with open('label_test.txt', 'w') as f:
            f.writelines(test_list)

    # k_random
    if k_random == True:
        indexs = locals()
        test_lists = locals()
        test_txts = locals()
        train_lists = locals()
        train_txts = locals()
        sapmle_len = len(list(sample_all_dict.values())[0])
        for folder_name, sample_list in sample_all_dict.items():
            index_list = list(range(sapmle_len))
            folder = int(folder_name.split(',')[1])
            random.shuffle(index_list)

            for i in range(k):
                start = int(sapmle_len * i / k)
                end = int(sapmle_len * (i + 1) / k)
                indexs[f'index_{folder}_{i + 1}'] = index_list[start: end]
                test_lists[f'testfn_{k}_{i + 1}'] = []
                train_lists[f'trainfn_{k}_{i + 1}'] = []

        for i in range(k):
            for folder_name, sample_list in sample_all_dict.items():
                folder = int(folder_name.split(',')[1])
                test_lists[f'testfn_{k}_{i + 1}'].extend([sample_list[idx] for idx in indexs[f'index_{folder}_{i + 1}']])

                for j in range(k):
                    if j != i:
                        train_lists[f'trainfn_{k}_{i + 1}'].extend([sample_list[idx] for idx in indexs[f'index_{folder}_{j + 1}']])

        for i in range(k):
            test_txts[f'label_test{i + 1}'] = 'label_test_{}.txt'.format(i + 1)
            train_txts[f'label_train{i + 1}'] = 'label_train_{}.txt'.format(i + 1)
            with open(test_txts[f'label_test{i + 1}'], 'w') as f:
                f.writelines(test_lists[f'testfn_{k}_{i + 1}'])
            with open(train_txts[f'label_train{i + 1}'], 'w') as f:
                f.writelines(train_lists[f'trainfn_{k}_{i + 1}'])

    print(f'test_txts and train_txts, have been constructed!')
