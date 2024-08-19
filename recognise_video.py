import torch
import cv2
import numpy as np
import yaml
from pathlib import Path
from models.yolo import Model
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

class Opt:
    def __init__(self, img_size, batch_size):
        self.img_size = img_size
        self.batch_size = batch_size
        self.rect = True
        self.single_cls = False
        self.bucketing = False
        self.prefix = None

def recognize(data,
              cfg=None,
              weight=None,
              video_path=None,
              img_size=352,
              conf_thres=0.5,
              iou_thres=0.6,
              augment=False,
              model=None):
    device = select_device('')
    
    # Load dataset config
    if isinstance(data, str):
        with open(data) as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
    
    # Extract class names
    class_names = data.get('names', [])
    nc = int(data.get('nc', 3))
    if len(class_names) == 0:
        class_names = [f'class_{i}' for i in range(nc)]

    # Load model
    model = Model(cfg, ch=3, nc=nc)
    state_dict = torch.load(weight, map_location='cpu')['model']
    
    model_state_dict = model.state_dict()
    for k, v in state_dict.items():
        if k in model_state_dict and model_state_dict[k].shape == v.shape:
            model_state_dict[k] = v
    model.load_state_dict(model_state_dict, strict=False)
    
    model.to(device).eval()

    half = device.type != 'cpu'
    if half:
        model.half()

    opt = Opt(img_size, batch_size=1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = Path(video_path).stem + '_result.avi'
    out = cv2.VideoWriter(out_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (frame_width, frame_height))

    print("Starting video processing...")

    batch_frames = []
    frame_count = 0
    batch_size = 8  # Process 8 frames at a time to speed up

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        batch_frames.append(frame)
        if len(batch_frames) >= batch_size:
            # Process batch
            frames = np.stack(batch_frames)
            batch_frames = []

            imgs = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (img_size, img_size)) for f in frames]
            imgs = np.stack(imgs)
            imgs = np.transpose(imgs, (0, 3, 1, 2))  # Change to (N, C, H, W)
            imgs = torch.from_numpy(imgs).float() / 255.0
            imgs = imgs.to(device, non_blocking=True)
            imgs = imgs.half() if half else imgs.float()

            with torch.no_grad():
                out_tensor, _ = model(imgs, augment=augment)
                out_tensor = non_max_suppression(out_tensor, conf_thres=conf_thres, iou_thres=iou_thres)

            for i, img0 in enumerate(frames):
                if out_tensor[i] is not None:
                    pred = out_tensor[i]
                    pred[:, :4] = scale_coords(imgs.shape[2:], pred[:, :4], img0.shape[:2]).round()
                    for *xyxy, conf, cls in pred:
                        label = f'{class_names[int(cls)]} {conf:.2f}'
                        xyxy = [int(x) for x in xyxy]
                        cv2.rectangle(img0, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                        cv2.putText(img0, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                out.write(img0)
                
            frame_count += len(frames)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Result saved to {out_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='Path to model config')
    parser.add_argument('--weight', type=str, help='Path to model weights')
    parser.add_argument('--data', type=str, help='Path to dataset config')
    parser.add_argument('--video', type=str, help='Path to input video')
    parser.add_argument('--img_size', type=int, default=352, help='Image size')
    parser.add_argument('--conf_thres', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.6, help='IOU threshold for NMS')
    args = parser.parse_args()

    recognize(data=args.data,
              cfg=args.cfg,
              weight=args.weight,
              video_path=args.video,
              img_size=args.img_size,
              conf_thres=args.conf_thres,
              iou_thres=args.iou_thres)
