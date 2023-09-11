r""" testing code """
import argparse
import os
import torch.nn as nn
import numpy as np
import torch

import cv2

from skimage import filters
from PIL import Image
from model.sccnet import SCCNetwork
from common.logger import Logger, AverageMeter
from common.vis import Visualizer
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset
import torchvision.transforms.functional as FF
import torch.nn.functional as F
from torchmetrics.classification import BinaryJaccardIndex


def eigen2img_multi(eigen):
    img = eigen.numpy()
    threshs = filters.threshold_multiotsu(img)
    return img > threshs[1]


def eigen2img_adp(eigen):
    minv, maxv = eigen.min(), eigen.max()
    eigen = (eigen - minv) / (1e-6 + maxv - minv)
    uint_img = (eigen.numpy() * 255).astype('uint8')
    return cv2.adaptiveThreshold(uint_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, 9, 2)


def eigen2img_merge(eigen):
    mask = eigen2img_multi(eigen)
    detail = np.asarray(eigen2img_adp(eigen))
    return np.logical_and(mask, detail)


def test(model, dataloader, nshot, args):
    r""" Test HSNet """

    # Freeze randomness during testing for reproducibility
    utils.fix_randseed(0)
    average_meter = AverageMeter(dataloader.dataset)
    
    iou = BinaryJaccardIndex().cuda()

    for idx, batch in enumerate(dataloader):

        # 1. forward pass
        batch = utils.to_cuda(batch)
        pred_mask = model.module.predict_mask_nshot(batch, nshot=nshot)

        assert pred_mask.size() == batch['query_mask'].size()

        if args.fuse:
            b = pred_mask.size()[0]
            for i in range(b):
                img_name = batch['query_name'][i]
                file_name = os.path.join(args.eigen_path, f'{img_name}.pth')
                eigen = torch.load(file_name)

                if args.perfect:
                    best_iou = 0
                    best_eigen_mask = None
                    for j in range(4):
                        eigen_img = eigen2img_merge(eigen['eigenvectors'][j + 1].resize(64, 64))
                        eigen_mask = torch.tensor(eigen_img).int()
                        eigen_mask = F.interpolate(
                            eigen_mask.unsqueeze(0).unsqueeze(0).float(), (256, 256),
                            mode='bilinear',
                            align_corners=True).cuda().int().squeeze()
                        tiou = iou(batch['query_mask'][i], eigen_mask).cpu().item()
                        if tiou > best_iou:
                            best_iou = tiou
                            best_eigen_mask = eigen_mask

                    if best_iou > 0.1 and best_eigen_mask is not None:
                        pred_mask[i, :, :] = torch.logical_or(pred_mask[i, :, :], best_eigen_mask)
                else:
                    eigen_img = eigen2img_merge(eigen['eigenvectors'][1].resize(64, 64))
                    eigen_mask = torch.tensor(eigen_img).int()
                    eigen_mask = F.interpolate(
                        eigen_mask.unsqueeze(0).unsqueeze(0).float(), (256, 256),
                        mode='bilinear',
                        align_corners=True).cuda().int().squeeze()
                    tiou = iou(pred_mask[i, :, :], eigen_mask).cpu().item()
                    if tiou > 0.1:
                        pred_mask[i, :, :] = torch.logical_or(pred_mask[i, :, :], eigen_mask)

        if len(args.seg_path) > 0:
            b = pred_mask.size()[0]
            for i in range(b):
                img_name = batch['query_name'][i]
                cls_id = batch['class_id'][i]
                file_name = os.path.join(args.seg_path, f'{img_name}__{cls_id}.png')
                img = FF.to_pil_image(pred_mask[i, :, :].int())
                img.save(file_name)

        # 2. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask.clone(), batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
        average_meter.write_process(idx, len(dataloader), epoch=-1, write_batch_idx=1)

        # Visualize predictions
        if Visualizer.visualize:
            Visualizer.visualize_prediction_batch(batch['support_imgs'], batch['support_masks'],
                                                  batch['query_img'], batch['query_mask'],
                                                  pred_mask, batch['class_id'], idx,
                                                  area_inter[1].float() / area_union[1].float())

    # Write evaluation results
    average_meter.write_result('Test', 0)
    miou, fb_iou = average_meter.compute_iou()

    return miou, fb_iou


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='SCCNet Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='../Datasets_HSN')
    parser.add_argument('--benchmark', type=str, default='pascal', choices=['pascal', 'isaid', 'dlrsd'])
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--bsz', type=int, default=1)
    parser.add_argument('--img_size', type=int, default=400)
    parser.add_argument('--nworker', type=int, default=0)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--nshot', type=int, default=1)
    parser.add_argument('--backbone', type=str, default='resnet101', choices=['vgg16', 'resnet50', 'resnet101'])
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--use_original_imgsize', action='store_true')
    parser.add_argument('--seg_path', type=str, default='')
    parser.add_argument('--eigen_path', type=str, default='')
    parser.add_argument('--fuse', type=bool, default=False)
    parser.add_argument('--perfect', type=bool, default=False)
    args = parser.parse_args()
    Logger.initialize(args, training=False)

    # Model initialization
    model = SCCNetwork(args.backbone, args.use_original_imgsize)
    model.eval()
    Logger.log_params(model)

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())
    model = nn.DataParallel(model)
    model.to(device)

    # Load trained model
    if args.load == '': raise Exception('Pretrained model not specified.')
    model.load_state_dict(torch.load(args.load))

    # Helper classes (for testing) initialization
    Evaluator.initialize()
    Visualizer.initialize(args.visualize)
    
    # Dataset initialization
    FSSDataset.initialize(img_size=args.img_size, datapath=args.datapath, use_original_imgsize=args.use_original_imgsize)
    dataloader_test = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'test', aug=False, shot=args.nshot)

    # Test HSNet
    with torch.no_grad():
        test_miou, test_fb_iou = test(model, dataloader_test, args.nshot, args)
    Logger.info('Fold %d mIoU: %5.2f \t FB-IoU: %5.2f' % (args.fold, test_miou.item(), test_fb_iou.item()))
    Logger.info('==================== Finished Testing ====================')
