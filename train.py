import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils_box import *
from loss import YoloLoss
from VisRetNet.models import RMT_S
from  VOC_dataset import VOCDataset
from config import *


model = RMT_S(num_classes=5).to(DEVICE)

optimizer = optim.Adam(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)
loss_fn = YoloLoss()

if LOAD_MODEL:
    load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

train_dataset = VOCDataset(
    "dataset/train.csv",
    transform=transform,
    img_dir=IMG_DIR,
    label_dir=LABEL_DIR,
)

test_dataset = VOCDataset(
    "dataset/test.csv", transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR,
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    # num_workers=NUM_WORKERS,
    # pin_memory=PIN_MEMORY,
    # shuffle=True,
    # drop_last=True,
)

# test_loader = DataLoader(
#     dataset=test_dataset,
#     batch_size=BATCH_SIZE,
#     # num_workers=NUM_WORKERS,
#     # pin_memory=PIN_MEMORY,
#     # shuffle=True,
#     # drop_last=True,
# )

for epoch in range(EPOCHS):
    # for x, y in train_loader:
    #    x = x.to(DEVICE)
    #    for idx in range(8):
    #        bboxes = cellboxes_to_boxes(model(x))
    #        bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
    #        plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes)

    #    import sys
    #    sys.exit()

    pred_boxes, target_boxes = get_bboxes(
        train_loader, model, iou_threshold=0.5, threshold=0.4
    )

    mean_avg_prec = mean_average_precision(
        pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
    )
    print(f"epoch: {epoch} Train mAP: {mean_avg_prec}")

    #if mean_avg_prec > 0.9:
    #    checkpoint = {
    #        "state_dict": model.state_dict(),
    #        "optimizer": optimizer.state_dict(),
    #    }
    #    save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
    #    import time
    #    time.sleep(10)

    train_fn(train_loader, model, optimizer, loss_fn)
    test_fn(train_loader, model, loss_fn)
