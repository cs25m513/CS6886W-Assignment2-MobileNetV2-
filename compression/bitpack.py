"""From-scratch arbitrary bit-width (1..8 bit) tensor packing and unpacking into uint8 buffers.

PyTorch lacks native sub-byte tensor dtypes (e.g., int2, int3, int4, int6). Storing quantized
sub-byte integers inside torch.int8 or torch.float32 tensors wastes RAM/VRAM.
This module packs N unsigned b-bit integers [0, 2^b - 1] into ceil(N * b / 8) contiguous
uint8 bytes in RAM/VRAM using pure vectorized tensor bitwise arithmetic.
"""

import math
from typing import Tuple
import torch


def packed_nbytes(numel: int, bits: int) -> int:
    """Returns exact number of uint8 bytes required to pack `numel` elements of `bits` width."""
    assert 1 <= bits <= 8, f"Bit-width must be in [1, 8], got {bits}"
    return math.ceil(numel * bits / 8)


def pack_bits(q_tensor: torch.Tensor, bits: int) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """Packs an unsigned integer tensor with values in [0, 2^bits - 1] into a 1D uint8 tensor.

    Args:
        q_tensor: Tensor of unsigned integer values in [0, (1 << bits) - 1].
        bits: Number of bits per element (1 <= bits <= 8).

    Returns:
        packed: 1D torch.uint8 tensor of length ceil(numel * bits / 8).
        orig_shape: Original shape tuple needed for unpacking.
    """
    assert 1 <= bits <= 8, f"bits must be in [1, 8], got {bits}"
    orig_shape = tuple(q_tensor.shape)
    flat = q_tensor.reshape(-1).to(torch.uint8)
    numel = flat.numel()

    if bits == 8:
        return flat.contiguous(), orig_shape

    if bits == 4:
        # Fast path: 2 elements per byte
        if numel % 2 != 0:
            flat = torch.cat([flat, torch.zeros(1, dtype=torch.uint8, device=flat.device)])
        pairs = flat.view(-1, 2)
        packed = ((pairs[:, 0] & 0x0F) << 4) | (pairs[:, 1] & 0x0F)
        return packed.contiguous(), orig_shape

    if bits == 2:
        # Fast path: 4 elements per byte
        rem = numel % 4
        if rem != 0:
            pad = 4 - rem
            flat = torch.cat([flat, torch.zeros(pad, dtype=torch.uint8, device=flat.device)])
        quads = flat.view(-1, 4)
        packed = (
            ((quads[:, 0] & 0x03) << 6)
            | ((quads[:, 1] & 0x03) << 4)
            | ((quads[:, 2] & 0x03) << 2)
            | (quads[:, 3] & 0x03)
        )
        return packed.contiguous(), orig_shape

    # General vectorized bit-stream packing for any bits in {1, 3, 5, 6, 7}
    # Pack groups of 8 elements into `bits` bytes (since lcm(bits, 8) divides 8 * bits)
    rem = numel % 8
    if rem != 0:
        pad = 8 - rem
        flat = torch.cat([flat, torch.zeros(pad, dtype=torch.uint8, device=flat.device)])

    groups = flat.view(-1, 8).to(torch.int64)  # (G, 8)
    mask = (1 << bits) - 1
    acc = torch.zeros(groups.shape[0], dtype=torch.int64, device=flat.device)
    for i in range(8):
        acc |= (groups[:, i] & mask) << (i * bits)

    # Extract `bits` bytes per group
    out_bytes = []
    for b_idx in range(bits):
        out_bytes.append(((acc >> (b_idx * 8)) & 0xFF).to(torch.uint8))
    packed = torch.stack(out_bytes, dim=1).reshape(-1)

    total_bytes = packed_nbytes(numel, bits)
    return packed[:total_bytes].contiguous(), orig_shape


def unpack_bits(
    packed: torch.Tensor, orig_shape: Tuple[int, ...], bits: int, out_dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Unpacks a 1D uint8 bit-packed tensor back to `orig_shape` with dtype `out_dtype`."""
    assert 1 <= bits <= 8, f"bits must be in [1, 8], got {bits}"
    numel = math.prod(orig_shape)

    if bits == 8:
        return packed[:numel].to(out_dtype).view(orig_shape)

    if bits == 4:
        high = (packed >> 4) & 0x0F
        low = packed & 0x0F
        unpacked = torch.stack([high, low], dim=1).reshape(-1)[:numel]
        return unpacked.to(out_dtype).view(orig_shape)

    if bits == 2:
        b0 = (packed >> 6) & 0x03
        b1 = (packed >> 4) & 0x03
        b2 = (packed >> 2) & 0x03
        b3 = packed & 0x03
        unpacked = torch.stack([b0, b1, b2, b3], dim=1).reshape(-1)[:numel]
        return unpacked.to(out_dtype).view(orig_shape)

    # General vectorized unpacking for bits in {1, 3, 5, 6, 7}
    num_groups = math.ceil(numel / 8)
    req_bytes = num_groups * bits
    if packed.numel() < req_bytes:
        pad_len = req_bytes - packed.numel()
        packed = torch.cat([packed, torch.zeros(pad_len, dtype=torch.uint8, device=packed.device)])

    byte_groups = packed[:req_bytes].view(num_groups, bits).to(torch.int64)
    acc = torch.zeros(num_groups, dtype=torch.int64, device=packed.device)
    for b_idx in range(bits):
        acc |= byte_groups[:, b_idx] << (b_idx * 8)

    mask = (1 << bits) - 1
    elems = []
    for i in range(8):
        elems.append((acc >> (i * bits)) & mask)
    unpacked = torch.stack(elems, dim=1).reshape(-1)[:numel]
    return unpacked.to(out_dtype).view(orig_shape)
