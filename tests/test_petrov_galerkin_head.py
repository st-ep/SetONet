import unittest

import torch

from Models.SetONet import SetONet


class TestPetrovGalerkinHead(unittest.TestCase):
    def _make_model(self) -> SetONet:
        model = SetONet(
            input_size_src=1,
            output_size_src=1,
            input_size_tgt=1,
            output_size_tgt=1,
            p=8,
            phi_hidden_size=32,
            rho_hidden_size=32,
            trunk_hidden_size=32,
            n_trunk_layers=3,
            activation_fn=torch.nn.ReLU,
            use_deeponet_bias=False,
            phi_output_size=16,
            use_positional_encoding=False,
            pos_encoding_type="skip",
            aggregation_type="mean",
            branch_head_type="petrov_attention",
        )
        return model.double()

    def test_permutation_invariance(self) -> None:
        torch.manual_seed(0)
        model = self._make_model()
        model.eval()

        batch_size, n_sensors, n_points = 2, 17, 11
        xs = torch.randn(batch_size, n_sensors, 1, dtype=torch.float64)
        us = torch.randn(batch_size, n_sensors, 1, dtype=torch.float64)
        ys = torch.randn(batch_size, n_points, 1, dtype=torch.float64)
        mask = torch.ones(batch_size, n_sensors, dtype=torch.bool)

        with torch.no_grad():
            out1 = model(xs, us, ys, sensor_mask=mask)
            perm = torch.randperm(n_sensors)
            out2 = model(xs[:, perm], us[:, perm], ys, sensor_mask=mask[:, perm])

        self.assertLess((out1 - out2).abs().max().item(), 1e-6)

    def test_padding_invariance(self) -> None:
        torch.manual_seed(0)
        model = self._make_model()
        model.eval()

        batch_size, n_valid, n_total, n_points = 2, 7, 13, 9
        xs_valid = torch.randn(batch_size, n_valid, 1, dtype=torch.float64)
        us_valid = torch.randn(batch_size, n_valid, 1, dtype=torch.float64)
        ys = torch.randn(batch_size, n_points, 1, dtype=torch.float64)

        xs_padded = torch.zeros(batch_size, n_total, 1, dtype=torch.float64)
        us_padded = torch.zeros(batch_size, n_total, 1, dtype=torch.float64)
        xs_padded[:, :n_valid] = xs_valid
        us_padded[:, :n_valid] = us_valid

        mask = torch.zeros(batch_size, n_total, dtype=torch.bool)
        mask[:, :n_valid] = True

        with torch.no_grad():
            out_valid = model(xs_valid, us_valid, ys, sensor_mask=None)
            out_padded = model(xs_padded, us_padded, ys, sensor_mask=mask)

        self.assertLess((out_valid - out_padded).abs().max().item(), 1e-6)


if __name__ == "__main__":
    unittest.main()

