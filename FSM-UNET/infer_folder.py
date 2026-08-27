import os
import sys
import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader
from collections import OrderedDict
from tqdm import tqdm

from loader_folder import folder_loader
from models.FSM_UNet import FSM_UNet
from utils import *
from configs.config_setting import setting_config

import warnings
warnings.filterwarnings("ignore")

ASPP_VARIANT = "D"


def draw_contour_overlay(img_np, gt_mask, pred_mask, threshold=0.5):
    bg = img_np.astype(np.uint8).copy()
    if bg.ndim == 2:
        bg = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
    elif bg.shape[2] == 3:
        bg = cv2.cvtColor(bg, cv2.COLOR_RGB2BGR)

    gt_bin = (gt_mask > 0.5).astype(np.uint8) * 255
    pred_bin = (pred_mask > threshold).astype(np.uint8) * 255

    gt_contours, _ = cv2.findContours(gt_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pred_contours, _ = cv2.findContours(pred_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(bg, gt_contours, -1, (0, 0, 255), thickness=2)
    cv2.drawContours(bg, pred_contours, -1, (255, 0, 0), thickness=2)

    return bg


def main(config):
    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    outputs = os.path.join(config.work_dir, 'outputs')
    contour_dir = os.path.join(config.work_dir, 'contour_overlay')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(outputs, exist_ok=True)
    os.makedirs(contour_dir, exist_ok=True)

    global logger
    logger = get_logger('infer_folder', log_dir)
    log_config_info(config, logger)

    print('#----------GPU init----------#')
    set_seed(config.seed)
    gpu_ids = [0]
    torch.cuda.empty_cache()

    print('#----------Preparing Model----------#')
    model_cfg = config.model_config
    model = FSM_UNet(num_classes=model_cfg['num_classes'],
                     input_channels=model_cfg['input_channels'],
                     c_list=model_cfg['c_list'],
                     use_dle_skips=model_cfg.get('use_dle_skips', True),
                     use_gssm_blocks=model_cfg.get('use_gssm_blocks', True),
                     use_fsda_aspp=model_cfg.get('use_fsda_aspp', True),
                     deep_supervision=model_cfg.get('deep_supervision', True),
                     gssm_r=model_cfg.get('gssm_r', 4),
                     gssm_g=model_cfg.get('gssm_g', 2),
                     aspp_variant=ASPP_VARIANT,
                     )
    model = torch.nn.DataParallel(model.cuda(), device_ids=gpu_ids, output_device=gpu_ids[0])

    print('#----------Preparing Dataset----------#')
    data_dir = getattr(config, 'custom_test_dir', '')
    if not data_dir:
        raise ValueError('Please set config.custom_test_dir to your folder containing image/ and label/.')
    print(f'Dataset root: {data_dir}')
    test_dataset = folder_loader(data_dir, size=(config.input_size_h, config.input_size_w))
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             pin_memory=True,
                             num_workers=config.num_workers,
                             drop_last=False)

    print('#----------Preparing criterion----------#')
    criterion = config.criterion

    print('#----------Loading Weights----------#')
    weights_path = getattr(config, 'test_weights', '') or getattr(config, 'resume_model', '')
    if not weights_path or not os.path.isfile(weights_path):
        raise FileNotFoundError('Please set config.test_weights (or config.resume_model) to a valid checkpoint file path.')
    state = torch.load(weights_path, map_location=torch.device('cpu'))
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    if isinstance(state, dict) and any(k.startswith('module.') for k in state.keys()):
        state = OrderedDict((k[7:], v) for k, v in state.items())
    model.module.load_state_dict(state, strict=False)

    print('#----------Inferencing & Drawing Contours----------#')
    model.eval()
    loss_list = []
    all_preds, all_gts = [], []

    with torch.no_grad():
        for i, (img, msk) in enumerate(tqdm(test_loader, desc='Inference')):
            img = img.cuda(non_blocking=True).float()
            msk = msk.cuda(non_blocking=True).float()

            out = model(img)
            loss = criterion(out, msk)
            loss_list.append(loss.item())

            out_main = out[0] if isinstance(out, (tuple, list)) else out

            img_np = img.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
            gt_np = msk.squeeze().cpu().numpy()
            pred_np = out_main.squeeze().cpu().numpy()

            overlay = draw_contour_overlay(img_np, gt_np, pred_np, config.threshold)

            fname = test_dataset.filenames[i]
            save_name = os.path.splitext(fname)[0] + '_contour.png'
            cv2.imwrite(os.path.join(contour_dir, save_name), overlay)

            all_preds.append(pred_np.flatten())
            all_gts.append(gt_np.flatten())

    preds = np.concatenate(all_preds)
    gts = np.concatenate(all_gts)
    y_pre = np.where(preds >= config.threshold, 1, 0)
    y_true = np.where(gts >= 0.5, 1, 0)

    from sklearn.metrics import confusion_matrix
    confusion = confusion_matrix(y_true, y_pre)
    TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]
    accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
    sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
    specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
    f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
    miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

    log_info = (f'\n[Contour Overlay Inference] loss: {np.mean(loss_list):.4f}, '
                f'mIoU: {miou:.4f}, Dice: {f1_or_dsc:.4f}, '
                f'Acc: {accuracy:.4f}, SE: {sensitivity:.4f}, SP: {specificity:.4f}')
    print(log_info)
    logger.info(log_info)
    print(f'Contour overlays saved to: {contour_dir}')


if __name__ == '__main__':
    config = setting_config
    main(config)
