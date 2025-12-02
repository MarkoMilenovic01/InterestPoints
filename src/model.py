# ======================================================================
# SuperPoint-style Detector (PyTorch)
# Reshape/Permute decoder (NO PixelShuffle)
# ======================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# 1. Basic ResNet Block
# ======================================================================

# ======================================================================
# 1. Basic ResNet Block
#    - Two 3×3 convolutions
#    - BatchNorm after each conv
#    - Skip connection
#    - 1×1 projection if input/output channels differ
# ======================================================================

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.proj = nn.Conv2d(in_channels, out_channels, 1) \
                    if in_channels != out_channels else None

    def forward(self, x):
        identity = x  

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.proj is not None:
            identity = self.proj(identity)

        out = out + identity     
        return F.relu(out)       



# ======================================================================
# 2. Encoder
#
# Input:  (B, 1, 240, 320)  grayscale image
# Output: (B, 128, 30, 40)  feature map (downsampled by factor 8)
#
# Structure:
#   Stage 1: 2x ResNet(1→64), MaxPool → output size (B, 64, H/2, W/2)
#   Stage 2: 2x ResNet(64→64), MaxPool → output size (B, 64, H/4, W/4)
#   Stage 3: 2x ResNet(64→128), MaxPool → output size (B,128,H/8,W/8)
#   Stage 4: 2x ResNet(128→128)        → output size (B,128,H/8,W/8)
# ======================================================================

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        # ---------------------------
        # Stage 1 (Downsample → H/2)
        # ---------------------------
        self.s1b1 = ResNetBlock(1, 64)
        self.s1b2 = ResNetBlock(64, 64)
        self.pool1 = nn.MaxPool2d(2)  # (H,W) → (H/2, W/2)

        # ---------------------------
        # Stage 2 (Downsample → H/4)
        # ---------------------------
        self.s2b1 = ResNetBlock(64, 64)
        self.s2b2 = ResNetBlock(64, 64)
        self.pool2 = nn.MaxPool2d(2)  # (H/2,W/2) → (H/4, W/4)

        # ---------------------------
        # Stage 3 (Downsample → H/8)
        # ---------------------------
        self.s3b1 = ResNetBlock(64, 128)
        self.s3b2 = ResNetBlock(128, 128)
        self.pool3 = nn.MaxPool2d(2)  # (H/4,W/4) → (H/8, W/8)

        # ---------------------------
        # Stage 4 (No downsample)
        # ---------------------------
        self.s4b1 = ResNetBlock(128, 128)
        self.s4b2 = ResNetBlock(128, 128)

    def forward(self, x):
        # Input: x ∈ (B, 1, H, W) - Grayscale image
        # Expect: H,W divisible by 8 (e.g. 240×320)

        # ----- Stage 1 -----
        x = self.s1b1(x)
        x = self.s1b2(x)
        x = self.pool1(x)
        # Size → (B, 64, H/2, W/2)

        # ----- Stage 2 -----
        x = self.s2b1(x)
        x = self.s2b2(x)
        x = self.pool2(x)
        # Size → (B, 64, H/4, W/4)

        # ----- Stage 3 -----
        x = self.s3b1(x)
        x = self.s3b2(x)
        x = self.pool3(x)
        # Size → (B, 128, H/8, W/8)

        # ----- Stage 4 -----
        x = self.s4b1(x)
        x = self.s4b2(x)
        # Final size → (B, 128, H/8, W/8)

        return x




# ======================================================================
# 3. Detector Head
# Input:  (B, 128, H/8, W/8) - feature map from encoder
# Output: (B, 65,  H/8, W/8)   ← raw logits (NO softmax here!)
# ======================================================================

class DetectorHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(256)

        self.out_conv = nn.Conv2d(256, 65, kernel_size=1)

    def forward(self, x):
        # x is the encoder output: (B, 128, H/8, W/8)

        x = F.relu(self.bn1(self.conv1(x)))

        # final logits: (B, 65, H/8, W/8)
        x = self.out_conv(x)

        return x



# ======================================================================
# 4. Decoder (RESHAPE → PERMUTE → RESHAPE)
# ======================================================================

class LogitsToProbMap(nn.Module):
    """Convert (B,65,H/8,W/8) logits → (B,1,H,W) probability map."""

    def forward(self, logits):
        B, C, Hc, Wc = logits.shape
        assert C == 65

        # softmax
        probs = F.softmax(logits, dim=1)

        # drop dustbin
        probs = probs[:, :-1, :, :]   # → (B,64,Hc,Wc)

        # Step 1: reshape channels → (8×8)
        probs = probs.view(B, 8, 8, Hc, Wc)   # (B, 8, 8, Hc, Wc)

        # Step 2: reorder to interleave correctly
        probs = probs.permute(0, 3, 1, 4, 2)
        # now shape = (B, Hc, 8, Wc, 8)

        # Step 3: final reshape
        prob_map = probs.reshape(B, 1, Hc * 8, Wc * 8)

        return prob_map



# ======================================================================
# 5. Full SuperPoint Detector
# ======================================================================

class SuperPointDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.head    = DetectorHead()
        self.decode  = LogitsToProbMap()

    def forward(self, x):
        features = self.encoder(x)
        logits   = self.head(features)

        if self.training:
            return logits

        prob_map = self.decode(logits)
        return logits, prob_map



def test_decoder_correctness():
    model = SuperPointDetector().eval()

    H, W = 256, 256
    Hc, Wc = H // 8, W // 8

    # Create fake logits
    logits = torch.zeros(1, 65, Hc, Wc)

    cell_x = 10   # which coarse cell in X direction
    cell_y = 5    # which coarse cell in Y direction
    sub_x = 3     # which subpixel inside 8x8
    sub_y = 6     # which subpixel inside 8x8

    channel_index = sub_y * 8 + sub_x  # flatten (8×8) into 0..63

    logits[0, channel_index, cell_y, cell_x] = 20

    _, prob = model(torch.randn(1,1,H,W))  # we ignore this part
    prob = model.decode(logits)

    # Expected location in final image
    px = cell_x * 8 + sub_x
    py = cell_y * 8 + sub_y

    print("Max location =", torch.argmax(prob).item())
    print("Expected index =", py * W + px)


def test_forward_pass():
    model = SuperPointDetector().eval()

    x = torch.randn(1, 1, 256, 256)
    logits, prob = model(x)

    print("Encoder output:", model.encoder(x).shape)
    print("Logits shape:", logits.shape)
    print("Prob shape:", prob.shape)



# ======================================================================
# MAIN TEST
# ======================================================================

if __name__ == "__main__":
    model = SuperPointDetector()
    model.eval()

    x = torch.randn(1, 1, 256, 256)

    logits, prob_map = model(x)

    print("\n===== SHAPE CHECK =====")
    print("Input:     ", x.shape)
    print("Logits:    ", logits.shape)      # (1,65,32,32)
    print("Prob map:  ", prob_map.shape)    # (1,1,256,256)
    print("=======================\n")

    test_decoder_correctness()
    test_forward_pass()

    model = SuperPointDetector()
    x = torch.randn(2,1,256,256)
    logits = model(x)

    loss = logits.mean()
    loss.backward()
    print("Gradient OK")

    