from metrics import dice, hausdorff_distance_95, jaccard, ConfusionMatrix
from process_mask import process_mask_brats, process_mask_nnunet

from glob import glob
from time import time
import os

import SimpleITK as sitk
import pandas as pd
import numpy as np


def get_metric(label_path, reference_path,
               label_type_path='/Users/qlc/Desktop/Result/label_type.csv', verbose=True):
    """
    计算两个文件夹内对应数据的指标，并保存到 file_name 中

    Args:
        label_path (str): 标签根目录
        
            label_path:
                1.nii.gz
                2.nii.gz
                ...
                *.nii.gz
                    
        reference_path (str): 推断根目录
        file_name (str): csv 文件保存目录
    """
    label_list = sorted(glob(os.path.join(label_path, '*.nii.gz')))
    reference_list = sorted(glob(os.path.join(reference_path, '*.nii.gz')))
    assert{len(label_list) == len(reference_list)}, f'文件夹中文件个数不同: len(label_list): {len(label_list)}; ' \
                                                    f'len(reference_list): {len(reference_list)}'

    file_name = root + '.csv'

    label_type = pd.read_csv(label_type_path)
    idx = label_type['Idx']

    name_list = []

    for path in idx:
        name = os.path.split(path)[-1]
        name = str.split(name, sep='.')[0]

        name_list.append(name)

    classes = ['WT', 'TC', 'ET']

    dice_socres_per_classes = {key: list() for key in classes}
    haus_socres_per_classes = {key: list() for key in classes}

    mean_list = {'Dice': [],
                 'Haus 95': []}
    id_list = []

    if os.path.exists(file_name):
        print(f'file exists, skip: {file_name}')
        return
    else:
        print('save file to:', file_name)
        for label, reference in zip(label_list, reference_list):

            start_time = time()
            id = os.path.split(label)[-1]
            id = str.split(id, sep='.')[0]

            id_list.append(id)

            target = sitk.ReadImage(label)
            target = sitk.GetArrayFromImage(target)

            refer = sitk.ReadImage(reference)
            refer = sitk.GetArrayFromImage(refer)

            label_processed = process_mask_nnunet(target)
            reference_processed = process_mask_nnunet(refer)

            confusion_matrix_wt = ConfusionMatrix(label_processed[0], reference_processed[0])
            dice_score_wt = dice(None, None, confusion_matrix_wt)
            haus_score_wt = hausdorff_distance_95(None, None, confusion_matrix_wt)

            dice_socres_per_classes['WT'].append(dice_score_wt)
            haus_socres_per_classes['WT'].append(haus_score_wt)

            confusion_matrix_tc = ConfusionMatrix(label_processed[1], reference_processed[1])
            dice_score_tc = dice(None, None, confusion_matrix_tc)
            haus_score_tc = hausdorff_distance_95(None, None, confusion_matrix_tc)

            dice_socres_per_classes['TC'].append(dice_score_tc)
            haus_socres_per_classes['TC'].append(haus_score_tc)

            confusion_matrix_et = ConfusionMatrix(label_processed[2], reference_processed[2])
            dice_score_et = dice(None, None, confusion_matrix_et)
            haus_score_et = hausdorff_distance_95(None, None, confusion_matrix_et)

            if id in name_list:
                if dice_score_et != 0:
                    dice_score_et = 0
                    haus_score_et = 377
                elif dice_score_et == 0:
                    dice_score_et = 1
                    haus_score_et = 0

            dice_socres_per_classes['ET'].append(dice_score_et)
            haus_socres_per_classes['ET'].append(haus_score_et)

            mean_list['Dice'].append((dice_score_wt + dice_score_tc + dice_score_et) / 3)
            mean_list['Haus 95'].append((haus_score_wt + haus_score_tc + haus_score_et) / 3)

            end_time = time()

            if verbose:
                print('-' * 20)
                print(id)

                print('WT:')
                print('dice_score_wt:   |', dice_score_wt)
                print('haus_score_wt:   |', haus_score_wt)

                print('TC:')
                print('dice_score_tc:   |', dice_score_tc)
                print('haus_score_tc:   |', haus_score_tc)

                print('ET:')
                print('dice_score_et:   |', dice_score_et)
                print('haus_score_et:   |', haus_score_et)

                print('\ntotal time cost:', end_time - start_time)
                print('-' * 20)
            else:
                print(id)

    dice_df = pd.DataFrame(dice_socres_per_classes)
    dice_df.columns = ['WT dice', 'TC dice', 'ET dice']
    haus_df = pd.DataFrame(haus_socres_per_classes)
    haus_df.columns = ['WT haus', 'TC haus', 'ET haus']
    mean_df = pd.DataFrame(mean_list)
    mean_df.columns = ['Mean Dice', 'Mean Haus']

    name_df = pd.DataFrame(id_list)
    name_df.columns = ['Id']

    val_metrics_df = pd.concat([name_df, mean_df, dice_df, haus_df], axis=1, sort=True)
    val_metrics_df = val_metrics_df.set_index(['Id'])
    val_metrics_df = val_metrics_df.fillna(0)

    label_type = pd.read_csv(label_type_path)
    idx = label_type['Idx']

    name_list = []

    for path in idx:
        name = os.path.split(path)[-1]
        name = str.split(name, sep='.')[0]

        name_list.append(name)

    for name in name_list:
        if val_metrics_df.loc[name, 'ET dice'] != 0:
            val_metrics_df.loc[name, 'ET dice'] = 0
        elif val_metrics_df.loc[name, 'ET dice'] == 0:
            val_metrics_df.loc[name, 'ET dice'] = 1

        if val_metrics_df.loc[name, 'ET haus'] != 0:
            val_metrics_df.loc[name, 'ET haus'] = 377
        else:
            val_metrics_df.loc[name, 'ET haus'] = 0

    val_metrics_df.to_csv(file_name)


def merge_csv(file_list, file_name):

    name_list = []
    dice_wt = []
    haus_wt = []
    dice_tc = []
    haus_tc = []
    dice_et = []
    haus_et = []
    mean_dice = []
    mean_haus = []

    for path in file_list:
        name = os.path.split(path)[-1]
        name = str.split(name, sep='.')[0]
        name_list.append(name)

        df = pd.read_csv(path)

        dice_wt.append(np.round(np.mean(df['WT dice']), 4))
        haus_wt.append(np.round(np.mean(df['WT haus']), 4))

        dice_tc.append(np.round(np.mean(df['TC dice']), 4))
        haus_tc.append(np.round(np.mean(df['TC haus']), 4))

        dice_et.append(np.round(np.mean(df['ET dice']), 4))
        haus_et.append(np.round(np.mean(df['ET haus']), 4))

        mean_dice.append(np.round((np.mean(df['WT dice'] + df['TC dice'] + df['ET dice']) / 3), 4))
        mean_haus.append(np.round(np.mean(df['WT haus'] + df['TC haus'] + df['ET haus'] / 3), 4))

    df = {
        'CSV': name_list,
        'Mean Dice': mean_dice,
        'Mean Haus': mean_haus,
        'WT Dice': dice_wt,
        'WT Haus': haus_wt,
        'TC Dice': dice_tc,
        'TC Haus': haus_tc,
        'ET Dice': dice_et,
        'ET Haus': haus_et,
    }

    df = pd.DataFrame(df)
    df.to_csv(file_name, index=False)


if __name__ == '__main__':
    root = '/Users/qlc/Desktop/Result/nnunet/inference/none/predict/syn_block/05'

    label_path = '/Users/qlc/Desktop/Result/nnunet/label/test_gt'

    # for root, dirs, files in os.walk(root):
    #     if not dirs:
    #         get_metric(label_path=label_path, reference_path=root, verbose=False)
    #         print(glob(root + '*.csv'))

    file_list = []
    for root_dir, dirs, files in os.walk(root):
        for file in files:
            if file.endswith('.csv'):
                file_list.append(os.path.join(root_dir, file))

    print(file_list)
    merge_csv(file_list, '/Users/qlc/Desktop/a.csv')




















    
