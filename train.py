r""" training (validation) code """
import argparse

import torch.optim as optim
import torch.nn as nn
import torch

from model.sccnet import SCCNetwork
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common import utils, my_optim
from data.dataset import FSSDataset


def train(epoch, model, dataloader, optimizer, training, loss_type='no', ld=1.0):
    r""" Train HSNet """
    train.count = getattr(train, 'count', 0)

    # Force randomness during training / freeze randomness during testing
    utils.fix_randseed(None) if training else utils.fix_randseed(0)
    model.module.train_mode() if training else model.module.eval()
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):
        train.count += 1
        if train.count > args.max_steps:
            break

        my_optim.adjust_learning_rate_poly(args, optimizer, train.count)

        # 1. forward pass
        batch = utils.to_cuda(batch)
        logit_mask, logit_mask2 = model(batch['query_img'], batch['support_imgs'].squeeze(1),
                                        batch['support_masks'].squeeze(1), batch['query_mask'])
        pred_mask = logit_mask.argmax(dim=1)

        # 2. Compute loss & update model parameters
        loss = model.module.compute_objective(logit_mask, batch['query_mask'])
        loss2 = model.module.compute_objective(logit_mask2, batch['query_mask'])
        loss += ld * loss2
        if loss_type == 'focal':
            loss += 0.5*model.module.compute_focal_loss(pred_mask)
        elif loss_type == 'area':
            loss += 0.5 * model.module.compute_area_loss(pred_mask, batch['query_mask'])
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 3. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask, batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())
        average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)

    # Write evaluation results
    average_meter.write_result('Training' if training else 'Validation', epoch)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()

    return avg_loss, miou, fb_iou


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='SCCNet Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='../Datasets')
    parser.add_argument('--benchmark', type=str, default='pascal', choices=['pascal', 'isaid', 'dlrsd'])
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--loss', type=str, default='no', choices=['no', 'focal', 'area'])
    parser.add_argument('--bsz', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--power', type=float, default=0.9)
    parser.add_argument('--ld', type=float, default=1.0)
    parser.add_argument('--niter', type=int, default=2000)
    parser.add_argument('--max_steps', type=int, default=50001)
    parser.add_argument('--nworker', type=int, default=8)
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--backbone', type=str, default='resnet101', choices=['vgg16', 'resnet50', 'resnet101'])
    parser.add_argument('--img_size', type=int, default=400)
    parser.add_argument('--use_original_imgsize', type=bool, default=False)
    parser.add_argument('--aug', type=bool, default=False)
    parser.add_argument('--freeze', type=bool, default=True)
    args = parser.parse_args()
    Logger.initialize(args, training=True)

    # Model initialization
    model = SCCNetwork(args.backbone, False, args.freeze)
    Logger.log_params(model)

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())
    model = nn.DataParallel(model)
    model.to(device)

    # Helper classes (for training) initialization
    optimizer = my_optim.get_finetune_optimizer(args, model)
    Evaluator.initialize()

    # Dataset initialization
    FSSDataset.initialize(img_size=args.img_size,
                          datapath=args.datapath,
                          use_original_imgsize=args.use_original_imgsize)
    dataloader_trn = FSSDataset.build_dataloader(args.benchmark, args.bsz,
                                                 args.nworker, args.fold,
                                                 'trn', aug=args.aug)
    dataloader_val = FSSDataset.build_dataloader(args.benchmark, args.bsz,
                                                 args.nworker, args.fold,
                                                 'val')

    # Train HSNet
    best_val_miou = float('-inf')
    best_val_loss = float('inf')
    for epoch in range(args.niter):

        trn_loss, trn_miou, trn_fb_iou = train(epoch,
                                               model,
                                               dataloader_trn,
                                               optimizer,
                                               training=True,
                                               loss_type=args.loss,
                                               ld=args.ld)
        with torch.no_grad():
            val_loss, val_miou, val_fb_iou = train(epoch,
                                                   model,
                                                   dataloader_val,
                                                   optimizer,
                                                   training=False)

        # Save the best model
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            Logger.save_model_miou(model, epoch, val_miou)

        Logger.tbd_writer.add_scalars('data/loss', {'trn_loss': trn_loss, 'val_loss': val_loss}, epoch)
        Logger.tbd_writer.add_scalars('data/miou', {'trn_miou': trn_miou, 'val_miou': val_miou}, epoch)
        Logger.tbd_writer.add_scalars('data/fb_iou', {'trn_fb_iou': trn_fb_iou, 'val_fb_iou': val_fb_iou}, epoch)
        Logger.tbd_writer.flush()

        if train.count > args.max_steps:
            break
    Logger.tbd_writer.close()
    Logger.info('==================== Finished Training ====================')
