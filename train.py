import os
import time
import datetime
from typing import Union, List
import torch
from torch.utils import data
import matplotlib.pyplot as plt

# ============================================================
# 【重要】每次跑不同实验，改这个名字！
# 例如： "baseline", "se_only", "cbam_only", "ours"
EXP_NAME = "ours"
# ============================================================

# --- 1. 导入双开关模型 ---
from u2net_pp import u2net_improved


# --- 导入其他工具  ---
from train_utils import train_one_epoch, evaluate, get_params_groups, create_lr_scheduler
from my_dataset import DUTSDataset
import transforms as T


class SODPresetTrain:
    def __init__(self, base_size: Union[int, List[int]], crop_size: int,
                 hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Resize(base_size, resize_mask=True),
            T.RandomCrop(crop_size),
            T.RandomHorizontalFlip(hflip_prob),
            T.Normalize(mean=mean, std=std)
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


class SODPresetEval:
    def __init__(self, base_size: Union[int, List[int]], mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Resize(base_size, resize_mask=False),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size

    # --- 自动创建对应的保存文件夹 ---
    save_dir = f"save_weights_{EXP_NAME}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 日志文件也加上实验名字，防止混淆
    results_file = f"results_{EXP_NAME}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"

    train_dataset = DUTSDataset(args.data_path, train=True, transforms=SODPresetTrain([512, 512], crop_size=512))
    val_dataset = DUTSDataset(args.data_path, train=False, transforms=SODPresetEval([512, 512]))

    num_workers = 0
    train_data_loader = data.DataLoader(train_dataset,
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        shuffle=True,
                                        pin_memory=True,
                                        collate_fn=train_dataset.collate_fn)

    val_data_loader = data.DataLoader(val_dataset,
                                      batch_size=1,
                                      num_workers=num_workers,
                                      pin_memory=True,
                                      collate_fn=val_dataset.collate_fn)

    # --- 2. 使用你的模型 ---
    print(f"正在启动实验: {EXP_NAME}")
    model = u2net_improved(out_ch=1)
    model.to(device)

    params_group = get_params_groups(model, weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(params_group, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_data_loader), args.epochs,
                                       warmup=True, warmup_epochs=2)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # Resume 逻辑保持不变
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    current_mae, current_f1 = 1.0, 0.0
    start_time = time.time()
    train_losses = []

    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = train_one_epoch(model, optimizer, train_data_loader, device, epoch,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)
        train_losses.append(mean_loss)

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        if epoch % args.eval_interval == 0 or epoch == args.epochs - 1:
            mae_metric, f1_metric = evaluate(model, val_data_loader, device=device)
            mae_info, f1_info = mae_metric.compute(), f1_metric.compute()
            print(f"[epoch: {epoch}] val_MAE: {mae_info:.3f} val_maxF1: {f1_info:.3f}")
            with open(results_file, "a") as f:
                write_info = f"[epoch: {epoch}] train_loss: {mean_loss:.4f} lr: {lr:.6f} " \
                             f"MAE: {mae_info:.3f} maxF1: {f1_info:.3f} \n"
                f.write(write_info)

            # 保存到对应的文件夹
            if current_mae >= mae_info and current_f1 <= f1_info:
                torch.save(save_file, os.path.join(save_dir, "model_best.pth"))

        # 只保留最近10个epoch
        last_epoch_path = os.path.join(save_dir, f"model_{epoch - 10}.pth")
        if os.path.exists(last_epoch_path):
            os.remove(last_epoch_path)

        torch.save(save_file, os.path.join(save_dir, f"model_{epoch}.pth"))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch u2net training")
    # 请确认这里的数据路径是正确的！
    parser.add_argument("--data-path", default=r"C:\Users\Downloads\Lightweight-Annotation", help="DUTS root")
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=4, type=int)  # 建议设为 8 或 12
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float, metavar='W', dest='weight_decay')
    parser.add_argument("--epochs", default=400, type=int, metavar="N")  # 跑400轮
    parser.add_argument("--eval-interval", default=10, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--print-freq', default=50, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N')
    parser.add_argument("--amp", default=True, type=bool, help="Use mixed precision")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
