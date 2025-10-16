import pytest
import torch

from dinov2.models.vision_transformer import DinoVisionTransformer


def make_vit_mp(depth=8, embed_dim=64, num_heads=4, devices=None):
    return DinoVisionTransformer(
        img_size=64,
        patch_size=16,
        in_chans=3,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=2.0,
        block_chunks=0,
        mp_devices=devices,
    )


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires >= 4 CUDA devices")
def test_mp_device_placement():
    devices = [torch.device(f"cuda:{i}") for i in range(4)]
    model = make_vit_mp(depth=8, devices=devices)

    assert getattr(model, "model_parallel", False), "model_parallel flag not set"
    assert hasattr(model, "_block_devices") and len(model._block_devices) == model.n_blocks

    # stage split mapping correctness
    for stage_idx, (s, e) in enumerate(model.stage_splits):
        for i in range(s, e):
            assert model._block_devices[i] == devices[stage_idx]

    # module placements
    first_block_param_dev = next(model.blocks[0].parameters()).device
    last_block_param_dev = next(model.blocks[-1].parameters()).device
    assert first_block_param_dev == devices[0]
    assert last_block_param_dev == devices[-1]
    assert next(model.patch_embed.parameters()).device == devices[0]
    assert next(model.norm.parameters()).device == devices[-1]


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires >= 4 CUDA devices")
def test_mp_forward_and_backward_single_input():
    devices = [torch.device(f"cuda:{i}") for i in range(4)]
    model = make_vit_mp(depth=8, devices=devices)
    model.train()

    x = torch.randn(2, 3, 64, 64)  # CPU input, model will move across devices
    out = model(x, is_training=True)

    assert isinstance(out, dict)
    assert out["x_norm_clstoken"].device == devices[-1]
    assert out["x_prenorm"].device == devices[-1]

    loss = out["x_norm_clstoken"].float().pow(2).mean()
    loss.backward()

    # Ensure grads flowed to early and late blocks
    assert any(p.grad is not None for p in model.blocks[0].parameters())
    assert any(p.grad is not None for p in model.blocks[-1].parameters())


def test_non_mp_forward_cpu():
    model = DinoVisionTransformer(
        img_size=32,
        patch_size=16,
        in_chans=3,
        embed_dim=32,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        block_chunks=0,
    )
    model.eval()

    x = torch.randn(1, 3, 32, 32)
    out = model(x, is_training=True)
    assert isinstance(out, dict)
    assert out["x_norm_clstoken"].shape == (1, 32)

    y = model(x)  # inference path
    assert y.shape == (1, 32)
