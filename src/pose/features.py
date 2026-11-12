"""
Dense (patch-level) DINOv3 features for the training-free 6D pose stage.
"""
import torch
import torch.nn.functional as F

from src.pose.geometry import PATCH


class DenseDINOv3:
    patch_size = PATCH

    def __init__(
        self,
        repo_dir="./dinov3_repo",
        model_name="dinov3_vitl16",
        weights="./dinov3_repo/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        layers=(23,),
        device="cuda",
        chunk_size=32,
        amp=True,
    ):
        self.model_name = model_name
        self.layers = tuple(int(layer) for layer in layers)  # 0-based block indices
        self.device = torch.device(device)
        self.chunk_size = chunk_size
        self.amp = amp
        model = torch.hub.load(repo_dir, model_name, source="local", weights=weights)
        # eval mode is required: in train mode the RoPE coordinates are jittered
        self.model = model.eval().to(self.device).requires_grad_(False)

    @property
    def name(self):
        return f"{self.model_name}_L" + "-".join(str(layer) for layer in self.layers)

    @property
    def dim(self):
        return self.model.embed_dim * len(self.layers)

    @torch.no_grad()
    def extract(self, images):
        """
        images: (B, 3, S, S) float, ImageNet-normalised and masked, S a multiple of 16.
        Returns (B, N, C) float32 patch features (row-major patches), L2-normalised.
        """
        B, _, H, W = images.shape
        num_patches = (H // PATCH) * (W // PATCH)
        if B == 0:
            return torch.zeros(
                (0, num_patches, self.dim), dtype=torch.float32, device=self.device
            )
        use_amp = self.amp and self.device.type == "cuda"
        outputs = []
        chunk = max(1, self.chunk_size * 448 * 448 // (H * W))  # fewer images per chunk for large crops
        for start in range(0, B, chunk):
            x = images[start : start + chunk].to(
                self.device, dtype=torch.float32
            )
            with torch.autocast(
                device_type=self.device.type, dtype=torch.bfloat16, enabled=use_amp
            ):
                layer_feats = self.model.get_intermediate_layers(
                    x, n=list(self.layers), reshape=False, norm=True
                )
            feats = torch.cat(
                [F.normalize(feat.float(), dim=-1) for feat in layer_feats], dim=-1
            )
            outputs.append(F.normalize(feats, dim=-1))
        return torch.cat(outputs, dim=0)
