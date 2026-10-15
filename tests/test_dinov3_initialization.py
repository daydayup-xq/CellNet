"""Checkpoint mapping must be complete and must leave the CellNet stem untouched."""
import tempfile
import unittest
from pathlib import Path
import torch
from safetensors.torch import save_file
from cellnet.encoder import CellNetEncoder
from cellnet.initialization import load_dinov3_backbone


class DinoInitializationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        self.model = CellNetEncoder(dict(embedding_dim=32, attention_heads=4, depth=2,
            tokens_per_cell=4, readout_layers=2, patch_size=16, image_size=[32, 32],
            vocabulary_size=10, zero_expression_floor=0.02, drop_path_rate=0.1))
        self.state = {}
        for k, v in self.model.state_dict().items():
            if not k.startswith('patch_embed.'):
                self.state[k.replace('.ls1.gamma', '.gamma_1').replace('.ls2.gamma', '.gamma_2')] = torch.full_like(v, 0.25)
        self.state['patch_embed.proj.weight'] = torch.ones(32, 3, 16, 16)
        self.state['patch_embed.proj.bias'] = torch.ones(32)
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'model.safetensors'

    def tearDown(self):
        self.tmp.cleanup()

    def test_timm_layerscale_mapping_and_stem_preservation(self):
        before = {k: v.clone() for k, v in self.model.patch_embed.state_dict().items()}
        save_file(self.state, str(self.path))
        load_dinov3_backbone(self.model, self.path)
        for k, v in self.model.state_dict().items():
            if not k.startswith('patch_embed.'):
                torch.testing.assert_close(v, torch.full_like(v, .25))
        for k, v in self.model.patch_embed.state_dict().items():
            torch.testing.assert_close(v, before[k], rtol=0, atol=0)

    def test_incomplete_checkpoint_fails_before_mutation(self):
        self.state.pop('blocks.1.attn.qkv.weight')
        before = self.model.cls_token.detach().clone()
        save_file(self.state, str(self.path))
        with self.assertRaisesRegex(ValueError, 'missing='):
            load_dinov3_backbone(self.model, self.path)
        torch.testing.assert_close(self.model.cls_token, before, rtol=0, atol=0)

    def test_nonzero_qkv_bias_rejected(self):
        self.state['layer.0.attention.q_proj.bias'] = torch.ones(32)
        save_file(self.state, str(self.path))
        with self.assertRaisesRegex(ValueError, 'zero DINOv3 QKV biases'):
            load_dinov3_backbone(self.model, self.path)

    def test_wrong_architecture_shape_rejected(self):
        self.state['reg_token'] = torch.ones(1, 4, 64)
        save_file(self.state, str(self.path))
        with self.assertRaisesRegex(ValueError, 'shape_mismatch='):
            load_dinov3_backbone(self.model, self.path)


if __name__ == '__main__':
    unittest.main()
