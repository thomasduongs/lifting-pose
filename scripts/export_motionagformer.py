#!/usr/bin/env python3
"""One-time export of official MotionAGFormer-XS H36M weights; no Python at runtime.

Use the upstream source at commit 4756fd1eb7cc73f0e991f091ff2280e030ab85f3.
Requires torch, timm, einops, PyYAML, onnx, and onnxruntime.
"""
import argparse
import inspect
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('models/motionagformer-xs.onnx'))
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    import yaml
    from model.MotionAGFormer import MotionAGFormer
    torch.set_num_threads(2)
    config = yaml.safe_load((args.source / 'configs/h36m/MotionAGFormer-xsmall.yaml').read_text())
    allowed = inspect.signature(MotionAGFormer).parameters
    kwargs = {key: value for key, value in config.items() if key in allowed}
    kwargs['act_layer'] = torch.nn.GELU
    model = MotionAGFormer(**kwargs).eval()
    # The official checkpoint stores a NumPy scalar validation metric.
    with torch.serialization.safe_globals([
        (np._core.multiarray.scalar, 'numpy.core.multiarray.scalar'),
        np.dtype, np.dtypes.Float64DType, np.dtypes.Float32DType,
    ]):
        weights = torch.load(args.checkpoint, map_location='cpu', weights_only=True)['model']
    weights = {key.removeprefix('module.'): value for key, value in weights.items()}
    model.load_state_dict(weights, strict=True)
    torch.manual_seed(7)
    sample = torch.randn(1, 27, 17, 3)
    sample[..., 2] = torch.rand(1, 27, 17)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        expected = model(sample).numpy()
        torch.onnx.export(model, sample, args.output, input_names=['poses2d'],
                          output_names=['poses3d'], opset_version=17, dynamo=False)
    exported = onnx.load(str(args.output))
    onnx.checker.check_model(exported)
    session = ort.InferenceSession(str(args.output), providers=['CPUExecutionProvider'])
    actual = session.run(None, {'poses2d': sample.numpy()})[0]
    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-4)
    print(f'Exported and verified {args.output}: input/output [1,27,17,3], max error {np.max(np.abs(actual-expected)):.6g}')
    # Save validation fixtures beside the ONNX file for a cross-runtime check.
    sample.numpy().tofile(args.output.with_suffix('.input.f32'))
    expected.tofile(args.output.with_suffix('.expected.f32'))


if __name__ == '__main__':
    main()
