# PowerSense

A Deep Reinforcement Learning Framework for Power-Aware Throughput Optimization in High-Density Manufacturing Test Environments.

## Overview

PowerSense jointly optimizes real-time power monitoring, demand forecasting, and dynamic test slot allocation to maximize throughput within hard power safety limits in manufacturing test environments.

The framework consists of three modules:

1. **Graph Attention Network (GAT)** — models the hierarchical electrical distribution topology to predict power consumption at arbitrary granularity
2. **Temporal Fusion Transformer (TFT)** — multi-horizon power demand forecasting incorporating product specifications and production schedules
3. **Proximal Policy Optimization (PPO)** — learns dynamic slot allocation policies via a high-fidelity digital twin

## Repository Structure

```
powersense/
├── README.md
├── LICENSE
├── benchmark/
│   ├── powerbench_spec.yaml      # Full benchmark specification
│   └── generate_benchmark.py     # Synthetic data generation
├── requirements.txt
└── configs/
    └── default_hyperparams.yaml  # All hyperparameters from the paper
```

## PowerBench Benchmark

PowerBench is a synthetic benchmark modeling a realistic high-volume manufacturing test facility. All parameters are calibrated to published industrial data:

- **Power profiles**: Calibrated to [SPECpower_ssj2008](https://www.spec.org/power_ssj2008/results/) published results (900+ server configurations)
- **Facility topology**: Follows NEC/NFPA 70 electrical distribution design practices
- **Variability**: Coefficient of variation (σ/μ = 0.05–0.15) matches published data center measurements (Barroso et al., 2019)

| Parameter | Value |
|-----------|-------|
| Total test slots | 5,000 |
| Transformers (L1) | 8 |
| Distribution panels (L2) | 48 |
| Branch circuits (L3) | 384 |
| Total facility capacity | 4.5 MW |
| Product types | 12 |
| Avg. power per UUT | 0.5–1.2 kW |
| Test duration | 2–8 hours |
| Test phases per product | 3–6 |
| Peak-to-average power ratio | 1.4–2.1 |
| Production rate | 800–1,200 UUTs/day |
| Decision epoch interval | 5 minutes |

## Quick Start

```bash
pip install -r requirements.txt

# Generate benchmark data
python benchmark/generate_benchmark.py --output data/ --days 30

# Generate with custom parameters
python benchmark/generate_benchmark.py --output data/ --days 30 --slots 2000 --seed 42
```

## Hyperparameters

All hyperparameters are specified in `configs/default_hyperparams.yaml` and match the paper exactly.

| Module | Parameter | Value |
|--------|-----------|-------|
| GAT | Layers | 3 |
| GAT | Hidden dim | 64 |
| GAT | Attention heads | 4 |
| GAT | Learning rate | 1e-3 |
| TFT | Attention heads | 4 |
| TFT | Hidden size | 128 |
| TFT | LSTM layers | 2 |
| PPO | MLP layers | 3 |
| PPO | Hidden units | 256 |
| PPO | Learning rate | 3e-4 |
| PPO | Batch size | 2,048 |
| PPO | Clip ratio | 0.2 |
| PPO | Discount factor | 0.99 |
| PPO | Training steps | 5M |

## Requirements

- Python >= 3.9
- PyTorch >= 2.1
- PyTorch Geometric >= 2.4
- NumPy >= 1.24
- PyYAML >= 6.0

## Citation

If you use PowerSense or PowerBench in your research, please cite:

```bibtex
@article{marimuthu2026powersense,
  title={PowerSense: A Deep Reinforcement Learning Framework for Power-Aware 
         Throughput Optimization in High-Density Manufacturing Test Environments},
  author={Marimuthu, Gopalakrishnan},
  journal={IEEE Transactions on Industrial Informatics},
  year={2026},
  note={Under review}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
