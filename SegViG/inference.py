#!/usr/bin/env python
import argparse
import os
import yaml
import torch
from contextlib import suppress
from timm.data import resolve_data_config
from timm.models import create_model
from data.myloader_csv_seg_patch import create_loader
from coral_pytorch.dataset import corn_label_from_logits
import model as model_module

from tqdm import tqdm
import time
import pandas as pd
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="SegViG Inference")
    p.add_argument('--checkpoint', type=str, required=True,
                   help='path to checkpoint (.pth or .tar)')
    p.add_argument('--data_dir', type=str, required=True,
                   help='root directory of images')
    p.add_argument('--meta_dir', type=str, required=True,
                   help='directory containing CSV files and segment data')
    p.add_argument('--csv_file', type=str, default='test.csv',
                   help='CSV file name under meta_dir')
    p.add_argument('--batch_size', type=int, default=None,
                   help='override batch size from args.yaml')
    p.add_argument('--device', type=str, default=None,
                   help='override device from args.yaml')
    p.add_argument('--output_preds', type=str, default='preds.csv',
                   help='output CSV file name for predictions')
    return p.parse_args()


def load_yaml_config(ckpt_path):
    folder = os.path.dirname(ckpt_path)
    yaml_path = os.path.join(folder, 'args.yaml')
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"No args.yaml found in {folder}")
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    cli = parse_args()
    cfg = load_yaml_config(cli.checkpoint)

    if cli.batch_size is not None:
        cfg['batch_size'] = cli.batch_size
    if cli.device is not None:
        cfg['device'] = cli.device

    device = torch.device(cfg.get('device', 'cuda:0') if torch.cuda.is_available() else 'cpu')

    num_class = cfg['num_classes'] - (1 if cfg.get('corn_loss', False) else 0)
    model = create_model(
        cfg['model'],
        pretrained=False,
        num_classes=num_class,
        use_segmentation_edge=cfg['use_segmentation_edge'],
        num_knn=cfg['num_knn'],
    )
    ck = torch.load(cli.checkpoint, map_location=device)
    state = ck.get('state_dict', ck)
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()

    data_cfg = resolve_data_config(cfg, model=model)
    loader = create_loader(
        data_dir=cli.data_dir,
        csv_file=os.path.join(cli.meta_dir, cli.csv_file),
        input_size=data_cfg['input_size'],
        batch_size=cfg['batch_size'],
        is_training=False,
        use_prefetcher=False,
        interpolation=data_cfg['interpolation'],
        mean=data_cfg['mean'],
        std=data_cfg['std'],
        num_workers=cfg.get('workers', 4),
        distributed=False,
        crop_pct=data_cfg['crop_pct'],
        pin_memory=True,
        buffer_wise=False,
        num_images_per_buffer=1,
        min_images_per_buffer=1,
        vis=False,
        overlap=cfg.get('overlap', False),
    )

    use_amp = cfg.get('amp', False)
    amp_autocast = torch.cuda.amp.autocast if use_amp else suppress

    preds = []
    times = []
    total_start = time.time()

    with torch.no_grad():
        for imgs, (seg, _) in tqdm(loader, desc="Inference", unit="batch"):
            t0 = time.time()
            imgs = imgs.to(device)
            n_image = cfg.get('num_images_per_buffer', 1)
            b_size = cfg['batch_size']
            with amp_autocast():
                if cfg.get('use_segmentation_edge', False):
                    seg = seg.to(device)
                    out = model(imgs, train_mech=cfg['training_mech'],
                                b_size=b_size, n_image=n_image, seg_result=seg)
                else:
                    out = model(imgs, train_mech=cfg['training_mech'],
                                b_size=b_size, n_image=n_image)

            logits = out[0] if isinstance(out, (tuple, list)) else out

            if cfg.get('corn_loss', False):
                batch_pred = corn_label_from_logits(logits).cpu().tolist()
            else:
                batch_pred = logits.argmax(1).cpu().tolist()

            times.append(time.time() - t0)
            preds.extend(batch_pred)

    total_time = time.time() - total_start
    fps = len(preds) / total_time
    print(f"\nTotal samples: {len(preds)}, Total time: {total_time:.2f}s, FPS: {fps:.1f}")

    csv_path = os.path.join(cli.meta_dir, cli.csv_file)
    df = pd.read_csv(csv_path)
    df['inference_result'] = preds
    out_csv = os.path.splitext(cli.csv_file)[0] + '_with_preds.csv'
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    label_col = 'avg_q7' if 'avg_q7' in df.columns else None
    if label_col:
        labels = df[label_col].values
        pred_arr = np.array(preds) + 1
        mae = np.mean(np.abs(pred_arr - labels))
        mse = np.mean((pred_arr - labels) ** 2)
        rmse = np.sqrt(mse)
        abs_diff = np.abs(pred_arr - labels)
        print(f"\n=== Evaluation Metrics (n={len(labels)}) ===")
        print(f"MAE : {mae:.4f}")
        print(f"MSE : {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        for k in [0, 1, 2, 3]:
            p = (abs_diff <= k).mean()
            print(f"P(|diff|<={k}): {p:.4f}  ({int(p * len(labels))}/{len(labels)})")


if __name__ == '__main__':
    main()
