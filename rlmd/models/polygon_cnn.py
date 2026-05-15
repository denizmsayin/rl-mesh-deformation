import torch
import torch.nn as nn
import torch.nn.functional as F


def circular_pad_polygon(x, num_verts, pad):
    """
    Circular-pad each item's valid vertex run along the sequence dim.

    For batch item b with n = num_verts[b] valid vertices, the output positions
    pad..pad+n-1 hold the original vertex features and the surrounding `pad`
    positions on each side wrap around that item's n-vertex cycle. Positions
    past pad+n-1 (inside the batch's zero-padding region) are set to zero.

    Args:
        x: (B, C, N_max) float.
        num_verts: (B,) long.
        pad: int — left/right pad in vertices.

    Returns:
        (B, C, N_max + 2*pad) float.
    """
    _, C, N_max = x.shape
    device = x.device
    ext_len = N_max + 2 * pad

    pos = torch.arange(ext_len, device=device)[None, :] - pad  # (1, ext_len) in [-pad, N_max+pad-1]
    n = num_verts[:, None]  # (B, 1)

    in_region = pos < n + pad  # (B, ext_len); pos >= -pad always holds
    # torch's `%` uses Python-style modulo, so negative pos wraps to the tail.
    n_safe = n.clamp(min=1)
    src_idx = torch.where(in_region, pos % n_safe, torch.zeros_like(pos))

    idx = src_idx.unsqueeze(1).expand(-1, C, -1)  # (B, C, ext_len)
    x_ext = torch.gather(x, 2, idx)
    return x_ext * in_region.unsqueeze(1).to(x.dtype)


def check_sequential_l(L, num_verts):
    """
    Assert L[b, i] == (i, (i+1) mod num_verts[b]) for all valid i.

    Raises ValueError if L does not encode the canonical sequential-cycle
    ordering of vertices around each closed polygon.
    """
    B, M_max, _ = L.shape
    device = L.device
    idx = torch.arange(M_max, device=device)[None, :].expand(B, -1)
    n = num_verts[:, None]
    valid = idx < n
    n_safe = n.clamp(min=1)
    expected_start = idx
    expected_end = (idx + 1) % n_safe

    starts_ok = (~valid) | (L[..., 0] == expected_start)
    ends_ok = (~valid) | (L[..., 1] == expected_end)
    if not (bool(starts_ok.all()) and bool(ends_ok.all())):
        raise ValueError(
            "L does not encode sequential vertex ordering; "
            "PolygonCNN requires L[i] == (i, (i+1) mod n) for i < n."
        )


class PolygonCNN(nn.Module):
    """
    1D CNN for point-wise feature extraction on closed 2D polygons.

    Vertices must be listed sequentially around each polygon; convolutions wrap
    circularly across each item's valid `num_verts` vertices. The line tensor L
    is not used by the convolution itself, but `forward(..., check_l=True)`
    verifies the sequentiality assumption.

    Args:
        in_channels: input feature dim per vertex (2 for raw XY).
        hidden_channels: tuple of channel widths for hidden conv layers.
        out_channels: output feature dim per vertex.
        kernel_size: odd int.
        layernorm: if True, apply LayerNorm over the channel dim per vertex
            after each conv (except the last).
    """
    def __init__(self, in_channels=2, hidden_channels=(64, 64, 128),
                 out_channels=128, kernel_size=5, layernorm=True):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for symmetric circular padding")
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2

        dims = [in_channels, *hidden_channels, out_channels]
        self.convs = nn.ModuleList([
            nn.Conv1d(dims[i], dims[i + 1], kernel_size, padding=0)
            for i in range(len(dims) - 1)
        ])
        if layernorm:
            self.norms = nn.ModuleList([
                nn.LayerNorm(dims[i + 1]) for i in range(len(dims) - 2)
            ])
        else:
            self.norms = None

    def forward(self, V, L, num_verts, check_l=False):
        """
        Args:
            V: (B, N_max, in_channels) float.
            L: (B, M_max, 2) long — only consulted when check_l=True.
            num_verts: (B,) long.
            check_l: if True, assert L encodes sequential ordering.

        Returns:
            (B, N_max, out_channels) float, zeroed at padded positions.
        """
        if check_l:
            check_sequential_l(L, num_verts)

        x = V.transpose(1, 2)  # (B, C_in, N_max)
        last = len(self.convs) - 1
        for i, conv in enumerate(self.convs):
            x = circular_pad_polygon(x, num_verts, self.pad)
            x = conv(x)
            if i < last:
                if self.norms is not None:
                    x = self.norms[i](x.transpose(1, 2)).transpose(1, 2)
                x = F.relu(x)

        N_max = x.shape[-1]
        mask = (torch.arange(N_max, device=x.device)[None, :] < num_verts[:, None])
        x = x * mask.unsqueeze(1).to(x.dtype)
        return x.transpose(1, 2)  # (B, N_max, C_out)
