# Tools moved

The Jetson thermal toolkit that accompanied paper 7 now lives in its own
repository, so it can be versioned, issue-tracked and installed independently
of this papers repo:

**https://github.com/Hi5808/jetson-tools**

```bash
git clone https://github.com/Hi5808/jetson-tools.git
cd jetson-tools
./jetson-fan-curve.sh --show     # read-only, safe on an unfamiliar board
```

Contents: `jetson-fan-curve.sh` (inspect/set fan curves in real degrees C,
converting to the TMARGIN encoding internally), `jetson-soak.sh` (sustained
soak with drift analysis), `gpu_burn.cu` (tensor core + DRAM + FP32 load) and
`jetson_thermal_lib.sh` (runtime platform detection).

The raw measurement data for paper 7 remains here under `../data/`.
