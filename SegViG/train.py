import warnings
warnings.filterwarnings('ignore')
import argparse
import time
import yaml
import os
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime

import torch
import torch.nn as nn
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from timm.data import resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.models import create_model, resume_checkpoint
from timm.utils import *
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler

from data.myloader_csv_seg_patch import create_loader
import model as model_module
from evals import overall_result
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, cohen_kappa_score, classification_report
from coral_pytorch.losses import CornLoss
from coral_pytorch.dataset import corn_label_from_logits
import wandb
from utils import ranking_contrastive_loss
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')

config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='SegViG Training')

parser.add_argument('--training_mech', type=str, default='vig',
                    help='vig | vig_pano_feat | vig_pano_pred')
parser.add_argument('--data_dir', type=str, default=None,
                    help='root directory of images')
parser.add_argument('--meta_dir', type=str, default='/path/to/meta_data',
                    help='directory containing train.csv / test.csv and segment data')
parser.add_argument('--model', default='vig_b_224_gelu', type=str, metavar='MODEL')
parser.add_argument('--pretrained', action='store_true', default=False)
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH')
parser.add_argument('--resume', default='', type=str, metavar='PATH')
parser.add_argument('--no-resume-opt', action='store_true', default=False)
parser.add_argument('--num-classes', type=int, default=4, metavar='N')
parser.add_argument('--gp', default=None, type=str, metavar='POOL')
parser.add_argument('--img-size', type=int, default=None, metavar='N')
parser.add_argument('--crop-pct', default=None, type=float, metavar='N')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME')
parser.add_argument('-b', '--batch-size', type=int, default=64, metavar='N')
parser.add_argument('-vb', '--validation-batch-size-multiplier', type=int, default=1, metavar='N')

parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER')
parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M')
parser.add_argument('--weight-decay', type=float, default=0.05)
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM')

parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER')
parser.add_argument('--lr', type=float, default=2e-3, metavar='LR')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV')
parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT')
parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N')
parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR')
parser.add_argument('--epochs', type=int, default=50, metavar='N')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N')
parser.add_argument('--decay-epochs', type=float, default=30, metavar='N')
parser.add_argument('--warmup-epochs', type=int, default=20, metavar='N')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE')

parser.add_argument('--no-aug', action='store_true', default=False)
parser.add_argument('--repeated-aug', action='store_true')
parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT')
parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO')
parser.add_argument('--hflip', type=float, default=0.5)
parser.add_argument('--vflip', type=float, default=0.)
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT')
parser.add_argument('--aa', type=str, default=None, metavar='NAME')
parser.add_argument('--aug-splits', type=int, default=0)
parser.add_argument('--jsd', action='store_true', default=False)
parser.add_argument('--corn_loss', action='store_true', default=False)
parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT')
parser.add_argument('--remode', type=str, default='pixel')
parser.add_argument('--recount', type=int, default=1)
parser.add_argument('--resplit', action='store_true', default=False)
parser.add_argument('--mixup', type=float, default=0.0)
parser.add_argument('--cutmix', type=float, default=0.0)
parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None)
parser.add_argument('--mixup-prob', type=float, default=1.0)
parser.add_argument('--mixup-switch-prob', type=float, default=0.5)
parser.add_argument('--mixup-mode', type=str, default='batch')
parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N')
parser.add_argument('--smoothing', type=float, default=0.0)
parser.add_argument('--train-interpolation', type=str, default='random')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT')
parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT')

parser.add_argument('--bn-tf', action='store_true', default=False)
parser.add_argument('--bn-momentum', type=float, default=None)
parser.add_argument('--bn-eps', type=float, default=None)
parser.add_argument('--sync-bn', action='store_true')
parser.add_argument('--dist-bn', type=str, default='')
parser.add_argument('--split-bn', action='store_true')

parser.add_argument('--model-ema', action='store_true', default=False)
parser.add_argument('--model-ema-force-cpu', action='store_true', default=False)
parser.add_argument('--model-ema-decay', type=float, default=0.99996)

parser.add_argument('--seed', type=int, default=42, metavar='S')
parser.add_argument('--log-interval', type=int, default=50, metavar='N')
parser.add_argument('--recovery-interval', type=int, default=0, metavar='N')
parser.add_argument('-j', '--workers', type=int, default=4, metavar='N')
parser.add_argument('--num-gpu', type=int, default=1)
parser.add_argument('--save-images', action='store_true', default=False)
parser.add_argument('--amp', action='store_true', default=False)
parser.add_argument('--apex-amp', action='store_true', default=False)
parser.add_argument('--native-amp', action='store_true', default=False)
parser.add_argument('--channels-last', action='store_true', default=False)
parser.add_argument('--pin-mem', action='store_true', default=False)
parser.add_argument('--no-prefetcher', action='store_true', default=False)
parser.add_argument('--output', default='', type=str, metavar='PATH')
parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC')
parser.add_argument('--tta', type=int, default=0, metavar='N')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--use-multi-epochs-loader', action='store_true', default=False)
parser.add_argument("--init_method", default='env://', type=str)
parser.add_argument("--train_url", type=str)

parser.add_argument('--attn_ratio', type=float, default=1.)
parser.add_argument("--pretrain_path", default=None, type=str)
parser.add_argument("--evaluate", action='store_true', default=False)
parser.add_argument("--num_images_per_buffer", type=int, default=8)
parser.add_argument("--min_images_per_buffer", type=int, default=10)
parser.add_argument("--use_segmentation_edge", nargs="?", const=False, default=False)
parser.add_argument('--contrastive_loss', action='store_true', default=False)
parser.add_argument('--wandb', action='store_true', default=False)
parser.add_argument("--device_num", type=int, default=0)
parser.add_argument('--contrast_alpha', type=float, default=0.1)
parser.add_argument('--weighted_sampler', action='store_true', default=False)
parser.add_argument("--num_knn", type=int, default=18)
parser.add_argument('--overlap', action='store_true', default=False)
parser.add_argument('--w', type=float, default=1.0)
parser.add_argument('--tau', type=float, default=0.2)
parser.add_argument('--e', type=float, default=0.01)
parser.add_argument("--exp_type", default=None, type=str)


def _parse_args():
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    args = parser.parse_args(remaining)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def print_train_info(args):
    print(f"Training Mechanism          : {args.training_mech}")
    print(f"Batch Size                  : {args.batch_size}")
    print(f"Model                       : {args.model}")
    print(f"Pretrained                  : {args.pretrained}")
    print(f"Pretrain Path               : {args.pretrain_path}")
    print(f"Epochs                      : {args.epochs}")
    print(f"Meta Directory              : {args.meta_dir}")
    print(f"Use Segmentation Edge       : {args.use_segmentation_edge}")
    print(f"Corn Loss                   : {args.corn_loss}")
    print(f"Contrastive Loss            : {args.contrastive_loss}")
    print(f"Contrastive Loss weight     : {args.contrast_alpha}")
    print(f"Weighted Sampler            : {args.weighted_sampler}")
    print(f"Repeated Aug                : {args.repeated_aug}")
    print(f"Overlap Edge                : {args.overlap}")


def main():
    setup_default_logging()
    args, args_text = _parse_args()

    torch.cuda.set_device(args.device_num)

    if args.wandb:
        if args.corn_loss:
            save_name = f'{args.training_mech}_corn_seg_edge({args.use_segmentation_edge})'
        else:
            save_name = f'{args.training_mech}_ce_seg_edge({args.use_segmentation_edge})'
        if args.contrastive_loss:
            save_name = f'{save_name}_contrastive_{args.contrast_alpha}'
        if args.smoothing == 0:
            save_name = f'{save_name}_nosmoothing'
        if args.exp_type is not None:
            save_name = f'{args.exp_type}_{save_name}'
        wandb_log = wandb.init(project='SegViG', name=save_name, config=args)
    else:
        wandb_log = None

    print_train_info(args)
    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        if args.distributed and args.num_gpu > 1:
            _logger.warning('Using more than one GPU per process in distributed mode is not allowed.')
            args.num_gpu = 1

    args.device = 'cuda:%d' % args.device_num
    args.world_size = 1
    args.rank = 0
    if args.distributed:
        args.num_gpu = 1
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.rank = int(os.environ['RANK'])
        torch.distributed.init_process_group(
            backend='nccl', init_method=args.init_method,
            rank=args.rank, world_size=args.world_size)
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
    assert args.rank >= 0

    torch.manual_seed(args.seed + args.rank)

    num_class = args.num_classes - 1 if args.corn_loss else args.num_classes
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=num_class,
        drop_rate=args.drop,
        drop_connect_rate=args.drop_connect,
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_tf=args.bn_tf,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        checkpoint_path=args.initial_checkpoint,
        use_segmentation_edge=args.use_segmentation_edge,
        num_knn=args.num_knn,
    )

    if args.pretrain_path is not None:
        print('Loading:', args.pretrain_path)
        model_dict = model.state_dict()
        checkpoint = torch.load(args.pretrain_path)
        modified_state_dict = {
            k: v for k, v in checkpoint.items()
            if k in model_dict and model_dict[k].size() == v.size()
        }
        model_dict.update(modified_state_dict)
        model.load_state_dict(model_dict)
        print('Pretrained weights loaded.')

    if args.local_rank == 0:
        _logger.info('Model %s created, param count: %d' %
                     (args.model, sum([m.numel() for m in model.parameters()])))

    data_config = resolve_data_config(vars(args), model=model, verbose=args.local_rank == 0)

    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    use_amp = None
    if args.amp:
        if has_apex:
            args.apex_amp = True
        elif has_native_amp:
            args.native_amp = True
    if args.apex_amp and has_apex:
        use_amp = 'apex'
    elif args.native_amp and has_native_amp:
        use_amp = 'native'
    elif args.apex_amp or args.native_amp:
        _logger.warning("Neither APEX nor native Torch AMP is available, using float32.")

    if args.num_gpu > 1:
        if use_amp == 'apex':
            use_amp = None
        model = nn.DataParallel(model, device_ids=list(range(args.num_gpu))).cuda(args.device)
    else:
        model.cuda(args.device)
        if args.channels_last:
            model = model.to(memory_format=torch.channels_last)

    optimizer = create_optimizer(args, model)

    amp_autocast = suppress
    loss_scaler = None
    if use_amp == 'apex':
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
    elif use_amp == 'native':
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()

    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            model, args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=args.local_rank == 0)

    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume=args.resume)

    if args.distributed:
        if args.sync_bn:
            try:
                if has_apex and use_amp != 'native':
                    model = convert_syncbn_model(model)
                else:
                    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            except Exception as e:
                _logger.error('Failed to enable Synchronized BatchNorm.')
        if has_apex and use_amp != 'native':
            model = ApexDDP(model, delay_allreduce=True)
        else:
            model = NativeDDP(model, device_ids=[args.local_rank])

    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    if args.start_epoch is not None:
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    if args.local_rank == 0:
        _logger.info('Scheduled epochs: {}'.format(num_epochs))

    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)
        if args.prefetcher:
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']

    buffer_wise = args.training_mech in ['vig_pano_feat', 'vig_pano_pred']

    loader_train = create_loader(
        data_dir=args.data_dir,
        csv_file=os.path.join(args.meta_dir, 'train.csv'),
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        use_multi_epochs_loader=args.use_multi_epochs_loader,
        repeated_aug=args.repeated_aug,
        buffer_wise=buffer_wise,
        num_images_per_buffer=args.num_images_per_buffer,
        min_images_per_buffer=args.min_images_per_buffer,
        vis=False,
        weighted=args.weighted_sampler,
        overlap=args.overlap,
    )

    loader_eval = create_loader(
        data_dir=args.data_dir,
        csv_file=os.path.join(args.meta_dir, 'test.csv'),
        input_size=data_config['input_size'],
        batch_size=args.validation_batch_size_multiplier * args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
        buffer_wise=buffer_wise,
        num_images_per_buffer=args.num_images_per_buffer,
        min_images_per_buffer=args.min_images_per_buffer,
        vis=False,
        overlap=args.overlap,
    )

    if args.jsd:
        assert num_aug_splits > 1
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing).cuda(args.device)
    elif mixup_active:
        train_loss_fn = SoftTargetCrossEntropy().cuda(args.device)
    elif args.corn_loss:
        train_loss_fn = CornLoss(num_classes=args.num_classes).cuda(args.device)
    elif args.smoothing:
        train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing).cuda(args.device)
    else:
        train_loss_fn = nn.CrossEntropyLoss().cuda(args.device)

    validate_loss_fn = (CornLoss(num_classes=args.num_classes) if args.corn_loss
                        else nn.CrossEntropyLoss()).cuda(args.device)

    if args.evaluate:
        eval_metrics = validate(model, loader_eval, validate_loss_fn, args,
                                amp_autocast=amp_autocast, return_overall_result=True)
        print('Evaluation Result:', eval_metrics)
        return

    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    best_overall_result = None
    saver = None
    output_dir = ''
    if args.local_rank == 0:
        output_base = args.output if args.output else './output'
        exp_name = '-'.join([
            datetime.now().strftime("%Y%m%d-%H%M%S"),
            args.model,
            str(data_config['input_size'][-1])
        ])
        output_dir = get_outdir(output_base, 'train', exp_name)
        decreasing = eval_metric in ['loss', 'mse', 'mae']
        saver = CheckpointSaver(
            model=model, optimizer=optimizer, args=args, model_ema=model_ema, amp_scaler=loss_scaler,
            checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=decreasing)
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    try:
        for epoch in range(start_epoch, num_epochs):
            if args.distributed:
                loader_train.sampler.set_epoch(epoch)
            start = time.time()

            train_metrics = train_epoch(
                epoch, model, loader_train, optimizer, train_loss_fn, args,
                lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir,
                amp_autocast=amp_autocast, loss_scaler=loss_scaler,
                model_ema=model_ema, mixup_fn=mixup_fn, wandb_log=wandb_log)

            print('epoch time:', time.time() - start)

            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

            eval_metrics, overall_result_dict = validate(
                model, loader_eval, validate_loss_fn, args,
                amp_autocast=amp_autocast, return_overall_result=True)

            if model_ema is not None and not args.model_ema_force_cpu:
                if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                    distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')
                ema_eval_metrics, _ = validate(
                    model_ema.ema, loader_eval, validate_loss_fn, args,
                    amp_autocast=amp_autocast, log_suffix=' (EMA)')
                eval_metrics = ema_eval_metrics

            if lr_scheduler is not None:
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            update_summary(epoch, train_metrics, eval_metrics,
                           os.path.join(output_dir, 'summary.csv'),
                           write_header=best_metric is None)

            if saver is not None:
                save_metric = overall_result_dict[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)
                if best_epoch == epoch:
                    best_overall_result = overall_result_dict

    except KeyboardInterrupt:
        pass

    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))
        _logger.info('*** Best Overall Result: {}'.format(best_overall_result))

    if wandb_log is not None:
        wandb_log.log({
            'test_mae': best_overall_result['mae'],
            'test_mse': best_overall_result['mse'],
            'test_kohen_quad': best_overall_result['kohen_quad'],
            'test_acc': best_overall_result['top1'],
            'test_macro-f1': best_overall_result['f1'],
            'spearman_rank': best_overall_result['spearman_rank'],
        })
        plt.figure(figsize=(6, 6))
        sns.heatmap(best_overall_result['confusion_matrix'], annot=True, fmt="d",
                    cmap="Blues",
                    xticklabels=list(range(args.num_classes)),
                    yticklabels=list(range(args.num_classes)))
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        wandb_log.log({"confusion_matrix_image": wandb.Image(plt)})
        plt.close()
        wandb_log.finish()


def train_epoch(epoch, model, loader, optimizer, loss_fn, args,
                lr_scheduler=None, saver=None, output_dir='', amp_autocast=suppress,
                loss_scaler=None, model_ema=None, mixup_fn=None, wandb_log=None):

    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()
    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)

    for batch_idx, (input, targets) in enumerate(loader):
        seg_result, target = targets
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        if not args.prefetcher:
            input = input.cuda(args.device)
            seg_result = seg_result.cuda(args.device)
            target = target.cuda(args.device)
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        with amp_autocast():
            if args.use_segmentation_edge:
                output = model(input, train_mech=args.training_mech,
                               b_size=args.batch_size, n_image=args.num_images_per_buffer,
                               seg_result=seg_result)
            else:
                output = model(input, train_mech=args.training_mech,
                               b_size=args.batch_size, n_image=args.num_images_per_buffer)

            loss = loss_fn(output, target)

            if args.contrastive_loss:
                feature = model.get_feature()
                if args.corn_loss:
                    pred = corn_label_from_logits(output)
                else:
                    _, pred = torch.max(output, 1)
                loss_contrast = ranking_contrastive_loss(
                    feature, target, pred, w=args.w, weights=1, t=args.tau, e=args.e)
                loss = loss + loss_contrast * args.contrast_alpha

        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(loss, optimizer, clip_grad=args.clip_grad,
                        parameters=model.parameters(), create_graph=second_order)
        else:
            loss.backward(create_graph=second_order)
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)
        num_updates += 1

        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)
            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), input.size(0))
            if args.local_rank == 0:
                _logger.info(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                    'LR: {lr:.3e}  '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch, batch_idx, len(loader),
                        100. * batch_idx / last_idx,
                        loss=losses_m, batch_time=batch_time_m,
                        rate=input.size(0) * args.world_size / batch_time_m.val,
                        lr=lr, data_time=data_time_m))

        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()

    if wandb_log is not None:
        wandb_log.log({'epoch': epoch, 'train_loss': losses_m.avg})

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg)])


def validate(model, loader, loss_fn, args, amp_autocast=suppress,
             log_suffix='', return_overall_result=False, wandb_log=None):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    f1_m = AverageMeter()
    kohen_m = AverageMeter()

    model.eval()
    end = time.time()
    last_idx = len(loader) - 1

    with torch.no_grad():
        pred_list = []
        lbl_list = []
        for batch_idx, (input, targets) in enumerate(loader):
            seg_result, target = targets
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.cuda(args.device)
                seg_result = seg_result.cuda(args.device)
                target = target.cuda(args.device)
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                if args.use_segmentation_edge:
                    output = model(input, train_mech=args.training_mech,
                                   b_size=args.batch_size, n_image=args.num_images_per_buffer,
                                   seg_result=seg_result)
                else:
                    output = model(input, train_mech=args.training_mech,
                                   b_size=args.batch_size, n_image=args.num_images_per_buffer)

            if isinstance(output, (tuple, list)):
                output = output[0]

            if args.corn_loss:
                pred = corn_label_from_logits(output)
            else:
                _, pred = torch.max(output, 1)

            pred_list.append(pred)
            lbl_list.append(target)

            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]

            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            f1 = f1_score(target.cpu(), pred.cpu(), average='macro')
            kohen_quad = cohen_kappa_score(target.cpu(), pred.cpu(), weights='quadratic')

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()
            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))
            f1_m.update(f1.item(), output.size(0))
            kohen_m.update(kohen_quad.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                _logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'F1: {f1.val:>7.4f} ({f1.avg:>7.4f})  '
                    'Kohen Quad: {kohen_quad.val:>7.4f} ({kohen_quad.avg:>7.4f})'.format(
                        log_name, batch_idx, last_idx,
                        loss=losses_m, top1=top1_m, f1=f1_m, kohen_quad=kohen_m))

    metrics = OrderedDict([
        ('loss', losses_m.avg), ('top1', top1_m.avg),
        ('top5', top5_m.avg), ('f1', f1_m.avg), ('kohen_quad', kohen_m.avg)
    ])

    mae, mse, kohen_quad, spearman_rank, acc, f1, classif_report, confusion_mat = \
        overall_result(torch.cat(lbl_list), torch.cat(pred_list))

    overall_result_dict = {
        'mae': mae, 'mse': mse, 'kohen_quad': kohen_quad,
        'spearman_rank': spearman_rank, 'top1': acc, 'f1': f1,
        'classif_report': classif_report, 'confusion_matrix': confusion_mat,
    }
    metrics.update({k: v for k, v in overall_result_dict.items()
                    if k not in ('classif_report', 'confusion_matrix')})

    if args.local_rank == 0:
        _logger.info(
            f"Overall Results:{log_suffix} "
            f"Loss: {metrics['loss']:.4f}, Acc@1: {metrics['top1']:.4f}, "
            f"F1: {metrics['f1']:.4f}, Kohen: {metrics['kohen_quad']:.4f}, "
            f"MAE: {metrics['mae']:.4f}, MSE: {metrics['mse']:.4f}, "
            f"Spearman: {metrics['spearman_rank']:.4f}"
        )

    return metrics, overall_result_dict


if __name__ == '__main__':
    main()
