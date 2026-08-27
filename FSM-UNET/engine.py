import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils import save_imgs
import cv2
import os


def _ds_breakdown_for_log(out, targets, criterion):
    logs = {}
    with torch.no_grad():
        if isinstance(out, (tuple, list)):
            seg = out[0]
            aux4 = out[1] if len(out) > 1 else None
            aux8 = out[2] if len(out) > 2 else None
        else:
            seg, aux4, aux8 = out, None, None

        if hasattr(criterion, "main"):
            logs["main"] = float(criterion.main(seg, targets).item())
        else:
            logs["main"] = float(criterion(seg, targets).item())

        if aux4 is not None and hasattr(criterion, "main"):
            t4 = F.interpolate(targets, size=aux4.shape[-2:], mode="nearest")
            logs["aux4"] = float(criterion.main(aux4, t4).item())
        if aux8 is not None and hasattr(criterion, "main"):
            t8 = F.interpolate(targets, size=aux8.shape[-2:], mode="nearest")
            logs["aux8"] = float(criterion.main(aux8, t8).item())
    return logs


def train_one_epoch(train_loader,
                    model,
                    criterion,
                    optimizer,
                    scheduler,
                    epoch,
                    logger,
                    config,
                    scaler=None):
    model.train()
    loss_list = []

    for iter, data in enumerate(train_loader):
        optimizer.zero_grad()
        images, targets = data
        images = images.cuda(non_blocking=True).float()
        targets = targets.cuda(non_blocking=True).float()

        if config.amp:
            with autocast():
                out = model(images)
                loss = criterion(out, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(images)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()

        loss_list.append(loss.item())

        if iter % config.print_interval == 0:
            now_lr = optimizer.state_dict()['param_groups'][0]['lr']
            msg = f'train: epoch {epoch}, iter:{iter}, loss:{np.mean(loss_list):.4f}, lr:{now_lr:.6f}'

            if isinstance(out, (tuple, list)):
                parts = _ds_breakdown_for_log(out, targets, criterion)
                if "main" in parts:
                    msg += f" | main:{parts['main']:.4f}"
                if "aux4" in parts:
                    msg += f" aux4:{parts['aux4']:.4f}"
                if "aux8" in parts:
                    msg += f" aux8:{parts['aux8']:.4f}"
            print(msg);
            logger.info(msg)

    scheduler.step()


def val_one_epoch(test_loader,
                  model,
                  criterion,
                  epoch,
                  logger,
                  config):
    model.eval()
    preds, gts, loss_list = [], [], []
    ds_seen = None

    with torch.no_grad():
        for data in tqdm(test_loader):
            img, msk = data
            img = img.cuda(non_blocking=True).float()
            msk = msk.cuda(non_blocking=True).float()

            out = model(img)
            loss = criterion(out, msk)
            loss_list.append(loss.item())

            if isinstance(out, (tuple, list)):
                if ds_seen is None:
                    ds_seen = True
                out_for_metric = out[0]
            else:
                if ds_seen is None:
                    ds_seen = False
                out_for_metric = out

            gts.append(msk.squeeze(1).cpu().detach().numpy())
            preds.append(out_for_metric.squeeze(1).cpu().detach().numpy())

    tip = "val: deep supervision outputs detected (tuple)" if ds_seen else "val: single-head output detected (no tuple)"
    logger.info(tip);
    print(tip)

    if epoch % config.val_interval == 0:
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds >= config.threshold, 1, 0)
        y_true = np.where(gts >= 0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        log_info = (f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, miou: {miou}, '
                    f'f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, '
                    f'specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}')
        print(log_info)
        logger.info(log_info)
    else:
        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)


def save_overlay_contours(img_tensor, gt_np, pred_np, save_dir, file_name, threshold=0.5):
    img_np = img_tensor.cpu().numpy()
    if img_np.shape[0] == 3:
        img_np = np.transpose(img_np, (1, 2, 0))

    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    img_bgr = (img_np * 255).astype(np.uint8)

    if img_tensor.shape[0] == 3:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)

    gt_bin = (gt_np > 0.5).astype(np.uint8) * 255
    pred_bin = (pred_np > threshold).astype(np.uint8) * 255

    contours_gt, _ = cv2.findContours(gt_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_pred, _ = cv2.findContours(pred_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    line_thickness = 2
    cv2.drawContours(img_bgr, contours_gt, -1, (0, 0, 255), line_thickness)
    cv2.drawContours(img_bgr, contours_pred, -1, (255, 0, 0), line_thickness)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, file_name)
    cv2.imwrite(save_path, img_bgr)

def test_one_epoch(test_loader,
                   model,
                   criterion,
                   logger,
                   config,
                   test_data_name=None):
    model.eval()
    preds, gts, loss_list = [], [], []

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data
            img = img.cuda(non_blocking=True).float()
            msk = msk.cuda(non_blocking=True).float()

            out = model(img)

            current_device = out[0].device if isinstance(out, (tuple, list)) else out.device

            msk = msk.to(current_device)
            criterion = criterion.to(current_device)

            loss = criterion(out, msk)
            loss_list.append(loss.item())

            out_for_vis = out[0] if isinstance(out, (tuple, list)) else out

            msk_np = msk.squeeze(1).cpu().detach().numpy()
            out_np = out_for_vis.squeeze(1).cpu().detach().numpy()

            gts.append(msk_np)
            preds.append(out_np)
            save_dir = os.path.join(config.work_dir, 'outputs_contours/')

            for b in range(img.shape[0]):
                file_name = f"pred_img_{i}_{b}.png"
                if test_data_name is not None:
                    file_name = f"{test_data_name}_{file_name}"

                save_overlay_contours(
                    img_tensor=img[b],
                    gt_np=msk_np[b],
                    pred_np=out_np[b],
                    save_dir=save_dir,
                    file_name=file_name,
                    threshold=config.threshold
                )

        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds >= config.threshold, 1, 0)
        y_true = np.where(gts >= 0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info);
            logger.info(log_info)

        log_info = (f'test of best model, loss: {np.mean(loss_list):.4f}, miou: {miou}, '
                    f'f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, '
                    f'specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}')
        print(log_info);
        logger.info(log_info)

    return np.mean(loss_list)
