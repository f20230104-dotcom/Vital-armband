#!/usr/bin/env python3
"""
verify_sim_windows.py - Score windows captured by tools/host_sim.cpp with the int8 TFLite model.

This closes the loop between the C++ firmware logic and the trained network: the windows are
exactly what motion_task hands to TFLite Micro (same ring buffer, same impact alignment, same
units), so the probabilities printed here are what the ESP32 will compute (up to rounding).

    c++ -std=c++17 -O2 -Iinclude tools/host_sim.cpp src/drivers/*.cpp -o host_sim   (see host_sim.cpp)
    ./host_sim 480 windows.csv
    python ml_training/verify_sim_windows.py windows.csv
"""
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

MODEL = Path(__file__).resolve().parent / "models" / "fall_cnn_int8.tflite"


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    X = np.loadtxt(sys.argv[1], delimiter=",", dtype=np.float32, ndmin=2).reshape(-1, 128, 6)
    it = tf.lite.Interpreter(model_path=str(MODEL), experimental_op_resolver_type=
                             tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    it.allocate_tensors()
    ind, outd = it.get_input_details()[0], it.get_output_details()[0]
    s_in, z_in = ind["quantization"]
    s_out, z_out = outd["quantization"]
    for i, x in enumerate(X):
        q = np.clip(np.round(x / s_in + z_in), -128, 127).astype(np.int8)[None]
        it.set_tensor(ind["index"], q)
        it.invoke()
        p = (it.get_tensor(outd["index"]).astype(np.float32)[0, 0] - z_out) * s_out
        mag = np.linalg.norm(x[:, :3], axis=1)
        print(f"window {i}: p(fall)={p:.3f}  min|a| pre-impact={mag[34:64].min():.2f} g  "
              f"peak={mag[59:74].max():.2f} g  -> {'FALL' if p >= 0.5 else 'ADL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
