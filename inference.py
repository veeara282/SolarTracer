import os
import torch
from torchvision import transforms
import argparse

from torchvision.transforms.functional import to_pil_image
from multi_resolution_segmentation import MultiResolutionSegmentation
from utils import SegmentationDataset, threshold
from tqdm import trange

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate the model on the test set')
    parser.add_argument('model', metavar='model.pt', default='model.pt')
    parser.add_argument('dataset', default='./NY_dataset/eval/')
    parser.add_argument('-o', '--out', default='./output/')
    return parser.parse_args()

def main():
    args = parse_args()

    # Read model file; do inference on CPU
    model = MultiResolutionSegmentation.from_save_file(args.model).cpu()
    model.eval()
    print(f"Loaded model from {args.model}")

    # Read files from test set
    dataset = SegmentationDataset(args.dataset, transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), # This time, convert directly to FloatTensor
        transforms.Lambda(lambda x: torch.unsqueeze(x, 0)) # Add batch dimension -> (N, C, H, W)
    ]))
    print(f"Loaded dataset from {args.dataset}")
    os.makedirs(args.out, exist_ok=True, mode=0o775)
    for i in trange(len(dataset), desc="Inference on test set"):
        # Do inference on the current image
        img, _ = dataset[i]
        pred_cam = torch.squeeze(model(img, return_cam=True), 0)
        pred_mask = threshold(pred_cam) * 255
        # Get the filename and extension
        img_path, _ = dataset.samples[i]
        _, img_filename = os.path.split(img_path)
        img_file, img_ext = os.path.splitext(img_filename)
        # Convert outputs to PIL images
        with to_pil_image(pred_cam) as cam_img:
            cam_path = os.path.join(args.out, f"{img_file}_cam{img_ext}")
            cam_img.save(cam_path)
        with to_pil_image(pred_mask, mode='1') as mask_img:
            mask_path = os.path.join(args.out, f"{img_file}_mask{img_ext}")
            mask_img.save(mask_path)

if __name__ == '__main__':
    main()
