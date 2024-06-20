import random
import torch

class hideandseek(torch.nn.Module):
    """Vertically flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, grid, has_prob=0.3, p=0.5):
        super().__init__()
        self.gird=grid
        self.has_prob = has_prob # hiding each patch probability
        self.p = p # Hide and Seek probability

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.
        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """

        if self.p < torch.rand(1):
            return img
            
        patch_w = self.gird[0]
        patch_h = self.gird[1]
        wd = img.size(-2)
        ht = img.size(-1)

        h_grid_sizes=int(ht/patch_h)
        w_grid_sizes=int(wd/patch_w)

        # hide the patches
        for x in range(0,wd,w_grid_sizes):
            for y in range(0,ht,h_grid_sizes):
                x_end = min(wd, x+w_grid_sizes)  
                y_end = min(ht, y+h_grid_sizes)
                if random.random() <=  self.has_prob:
                    img[:,:,x:x_end,y:y_end]=0
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"