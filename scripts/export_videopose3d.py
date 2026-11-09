#!/usr/bin/env python3
"""One-time ONNX export of the official VideoPose3D COCO checkpoint.

The Rust benchmark uses OpenCV, not Python, for inference. Supply the upstream
VideoPose3D source and pretrained_h36m_detectron_coco.bin. Upstream code/weights
retain their CC BY-NC license; this helper does not bundle either.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('models/videopose3d.onnx'))
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from common.model import TemporalModel, TemporalModelOptimized1f

    torch.set_num_threads(2)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    weights = checkpoint['model_pos']
    channels = weights['expand_conv.weight'].shape[0]
    model = TemporalModelOptimized1f(17, 2, 17, [3] * 5, channels=channels).eval()
    reference = TemporalModel(17, 2, 17, [3] * 5, channels=channels).eval()
    model.load_state_dict(weights, strict=True)
    reference.load_state_dict(weights, strict=True)
    torch.manual_seed(7)
    sample = torch.randn(1, 243, 17, 2) * 0.3
    with torch.no_grad():
        expected = reference(sample).numpy()
        optimized = model(sample).numpy()
    np.testing.assert_allclose(optimized, expected, rtol=1e-4, atol=1e-5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, sample, args.output, input_names=['poses2d'],
                      output_names=['poses3d'], opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(str(args.output)))
    session = ort.InferenceSession(str(args.output), providers=['CPUExecutionProvider'])
    actual = session.run(None, {'poses2d': sample.numpy()})[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
    sample.numpy().tofile(args.output.with_suffix('.input.f32'))
    expected.tofile(args.output.with_suffix('.expected.f32'))
    metadata = {
        'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        'input_shape': list(sample.shape), 'output_shape': list(expected.shape),
        'max_reference_error': float(np.abs(actual - expected).max()),
        'source': str(args.source.resolve()),
        'source_model_sha256': hashlib.sha256((args.source / 'common/model.py').read_bytes()).hexdigest(),
    }
    args.output.with_suffix('.export.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata))


if __name__ == '__main__':
    main()
