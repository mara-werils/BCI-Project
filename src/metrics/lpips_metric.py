import lpips
import torch

class LPIPSMetric:
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu
        self.loss_fn = lpips.LPIPS(net='alex')
        if self.use_gpu and torch.cuda.is_available():
            self.loss_fn.cuda()

    def calculate(self, img1, img2):
        """
        Calculate LPIPS distance between two images.
        img1, img2: Tensor or numpy array. 
                    If tensor, shape (N, 3, H, W), range [-1, 1].
                    If numpy, shape (H, W, 3) or (H, W), range [0, 255] or [0, 1]. 
                    We assume they are properly preprocessed tensors here as per this project's standard.
        """
        if self.use_gpu and torch.cuda.is_available():
            if not img1.is_cuda:
                img1 = img1.cuda()
            if not img2.is_cuda:
                img2 = img2.cuda()
        
        # Ensure range is [-1, 1] if implied. 
        # But commonly model outputs are [-1, 1] in pix2pix.
        
        with torch.no_grad():
            d = self.loss_fn(img1, img2)
        return d.item()
