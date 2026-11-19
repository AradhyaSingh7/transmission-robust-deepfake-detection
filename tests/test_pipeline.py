"""
Unit tests for Dual-Stream Architecture and Transformations
"""
import unittest
import torch
import numpy as np
from PIL import Image

from src.models.branches import SpatialBranch, FrequencyBranch
from src.models.cbam import CBAM
from src.models.dynamic_gate import DynamicGatedFusion, GatedHybridDetector
from src.data.transforms import DCTTransform, JPEGCompress, ResizeCompress


class TestPipeline(unittest.TestCase):

    def setUp(self):
        # Create a synthetic test RGB image
        self.img = Image.fromarray(np.uint8(np.random.rand(256, 256, 3) * 255))
        self.batch_size = 2

    def test_dct_transform_shape(self):
        transform = DCTTransform(channels='rgb')
        tensor = transform(self.img)
        self.assertEqual(tensor.shape, (3, 256, 256))
        self.assertTrue(torch.all(tensor >= 0.0) and torch.all(tensor <= 1.0))

    def test_jpeg_transform(self):
        compress = JPEGCompress(quality=50)
        out_img = compress(self.img)
        self.assertEqual(out_img.size, (256, 256))

    def test_dynamic_gate_routing(self):
        gate = DynamicGatedFusion(feature_dim=2048)
        f_s = torch.randn(self.batch_size, 2048)
        f_f = torch.randn(self.batch_size, 2048)
        fused, alpha = gate(f_s, f_f)

        self.assertEqual(fused.shape, (self.batch_size, 4096))
        self.assertEqual(alpha.shape, (self.batch_size, 1))
        self.assertTrue(torch.all(alpha >= 0.0) and torch.all(alpha <= 1.0))

    def test_detector_forward_pass(self):
        detector = GatedHybridDetector(feature_dim=2048, pretrained=False)
        s_in = torch.randn(self.batch_size, 3, 256, 256)
        f_in = torch.randn(self.batch_size, 3, 256, 256)
        prob, alpha = detector(s_in, f_in)

        self.assertEqual(prob.shape, (self.batch_size, 1))
        self.assertEqual(alpha.shape, (self.batch_size, 1))
        self.assertTrue(torch.all(prob >= 0.0) and torch.all(prob <= 1.0))


if __name__ == '__main__':
    unittest.main()
