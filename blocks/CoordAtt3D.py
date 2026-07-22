import torch
import torch.nn as nn

class h_swish(nn.Module):
    """
    Hard Swish activation function utilized for computationally 
    efficient non-linear transformations without exponential operations.
    """
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return x * self.relu(x + 3) / 6

class CoordAtt3D(nn.Module):
    def __init__(self, inp, reduction=32, dropout=0.1):
        super(CoordAtt3D, self).__init__()
        # Ensure a minimum number of reduction channels to prevent 
        # information bottleneck collapse in shallow layers.
        mip = max(8, inp // reduction)
        
        # Adaptive pooling across the three Cartesian axes (Depth, Height, Width)
        self.pool_d = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.pool_h = nn.AdaptiveAvgPool3d((1, None, 1))
        self.pool_w = nn.AdaptiveAvgPool3d((1, 1, None))
        
        # Shared 1x1x1 convolution for multi-axis feature aggregation
        self.conv1 = nn.Conv3d(inp, mip, kernel_size=1, stride=1, padding=0)
        
        # GroupNorm is strictly required here; BatchNorm3d will fail 
        # catastrophically with a batch size of 1.
        self.norm1 = nn.GroupNorm(1, mip)
        self.act = h_swish()
        
        # Output transformations for each distinct coordinate axis
        self.conv_d = nn.Conv3d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_h = nn.Conv3d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv3d(mip, inp, kernel_size=1, stride=1, padding=0)
        
        # Dropout added to mitigate the train/val loss gap (overfitting)
        self.dropout = nn.Dropout3d(p=dropout)

    def forward(self, x):
        identity = x
        b, c, d, h, w = x.size()
        
        # Extract coordinate tensors and permute for concatenation
        # x_d: (b, c, d, 1, 1) -> permute -> (b, c, 1, 1, d)
        x_d = self.pool_d(x).permute(0, 1, 3, 4, 2)
        # x_h: (b, c, 1, h, 1) -> permute -> (b, c, 1, 1, h)
        x_h = self.pool_h(x).permute(0, 1, 2, 4, 3)
        # x_w: (b, c, 1, 1, w)
        x_w = self.pool_w(x)
        
        # Concatenate along the spatial dimension (dim=4)
        # Shape becomes: (b, c, 1, 1, d + h + w)
        y = torch.cat([x_d, x_h, x_w], dim=4)
        
        # Process the aggregated spatial coordinates
        y = self.conv1(y)
        y = self.norm1(y)
        y = self.act(y)
        
        # Split the intermediate feature map back into individual axes
        x_d, x_h, x_w = torch.split(y, [d, h, w], dim=4)
        
        # Permute back to the original anatomical orientations
        x_d = x_d.permute(0, 1, 4, 2, 3) # (b, mip, d, 1, 1)
        x_h = x_h.permute(0, 1, 2, 4, 3) # (b, mip, 1, h, 1)
        
        # Generate the attention weights via 1x1x1 convs and sigmoid scaling
        a_d = self.conv_d(x_d).sigmoid()
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        
        # Re-weight the original input tensor and apply regularization
        out = identity * a_d * a_h * a_w
        out = self.dropout(out)
        
        return out