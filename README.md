# 🔋 g1-bms-yolo-benchmark

Benchmark and energy consumption analysis of CPU and GPU inference on the Unitree G1 EDU humanoid robot using ROS2 and YOLO-based computer vision workloads.

---

# 🤖 Overview

This repository contains the experimental framework used to evaluate:

- 🔋 Power consumption
- ⚡ Energy expenditure
- 🚀 Inference latency
- 📈 Throughput (FPS)
- 🔋 Battery behavior

during AI inference execution on the Unitree G1 EDU humanoid robot.

The experiments compare three execution scenarios:

1. ⚙️ Baseline (robot powered on without inference)
2. 🖥️ CPU-based inference
3. ⚡ GPU-based inference using CUDA acceleration

The study focuses on the energetic impact of embedded AI workloads on humanoid robotic platforms with limited battery capacity.

---

# 🦾 Hardware Platform

## Robot
- Unitree G1 EDU Humanoid Robot

## Embedded Computing Platform
- NVIDIA Jetson Orin NX

## Sensors
- RGB camera
- Internal Battery Management System (BMS)

---

# 💻 Software Stack

## Operating System
- Ubuntu Linux

## Middleware
- ROS2

## Deep Learning Frameworks
- PyTorch 2.4.1 (CPU execution)
- NVIDIA PyTorch 2.1.0a0 for JetPack (GPU execution)

## AI Model
- YOLO26 Extra Large (`yolo26x.pt`)

---

# 📊 Collected Metrics

The ROS2-based BMS monitor records:

- Voltage (V)
- Current (A)
- Power (W)
- Battery State of Charge (SOC)
- Temperature (°C)

## Sampling Configuration

| Parameter | Value |
|---|---|
| Sampling Rate | ~20 Hz |
| Samples per Scenario | 10,000 |
| Acquisition Duration | ~494 s |

---

# ⚙️ Experimental Scenarios

## ⚙️ Baseline
Robot powered on without inference workload.

## 🖥️ CPU Inference
Inference executed exclusively on CPU.

## ⚡ GPU Inference
Inference executed using CUDA acceleration on the Jetson Orin NX GPU.

---

# 📁 Repository Structure

```text
.
├── data/               # Raw experimental data
├── scripts/            # Benchmark and analysis scripts
├── ros2_ws/            # ROS2 workspace
├── figures/            # Experimental figures and plots
├── results/            # Processed results
└── README.md
````

---

# 🎯 Main Objectives

* Quantify energy overhead caused by AI inference
* Compare CPU and GPU efficiency
* Evaluate latency and throughput trade-offs
* Support research in embedded AI for humanoid robotics

---

# 📊 Experimental Results

## ⚙️ Baseline Scenario

### 🔋 Power and Battery Metrics

| Metric          | Value     |
| --------------- | --------- |
| Average Voltage | 46.54 V   |
| Average Current | 2.01 A    |
| Average Power   | 93.57 W   |
| Total Energy    | 12.84 Wh  |
| SOC Variation   | 34% → 31% |
| Battery Temp 1  | 38.47 °C  |
| Battery Temp 2  | 36.23 °C  |

### 📌 Notes

* Represents idle robot energy consumption.
* Used as reference for overhead calculations.
* Demonstrates stable baseline operation.

---

## 🖥️ CPU Inference Results

### 🔋 Power and Battery Metrics

| Metric          | Value     |
| --------------- | --------- |
| Average Voltage | 47.36 V   |
| Average Current | 2.13 A    |
| Average Power   | 100.63 W  |
| Total Energy    | 13.81 Wh  |
| SOC Variation   | 41% → 37% |
| Battery Temp 1  | 37.77 °C  |
| Battery Temp 2  | 35.00 °C  |

### 🚀 Inference Performance

| Metric                   | Value                |
| ------------------------ | -------------------- |
| Model                    | `yolo26x.pt`         |
| Average Latency          | 4367.87 ms           |
| Throughput               | 0.2 FPS              |
| Median Latency           | 4364.18 ms           |
| Min / Max Latency        | 4303.89 / 4492.37 ms |
| Standard Deviation       | 31.05 ms             |
| Coefficient of Variation | 0.71%                |

### 📌 Notes

* CPU inference was computationally impractical for real-time perception.
* Latency exceeded 4 seconds per frame.
* Energy overhead remained relatively small compared to baseline execution.

---

## ⚡ GPU Inference Results (CUDA)

### 🔋 Power and Battery Metrics

| Metric          | Value     |
| --------------- | --------- |
| Average Voltage | 52.61 V   |
| Average Current | 2.28 A    |
| Average Power   | 119.74 W  |
| Total Energy    | 16.43 Wh  |
| SOC Variation   | 90% → 87% |
| Battery Temp 1  | 33.10 °C  |
| Battery Temp 2  | 28.94 °C  |

### 🚀 Inference Performance

| Metric                   | Value             |
| ------------------------ | ----------------- |
| Model                    | `yolo26x.pt`      |
| Average Latency          | 106.6 ms          |
| Throughput               | 9.4 FPS           |
| Median Latency           | 106.15 ms         |
| Min / Max Latency        | 101.59 / 116.6 ms |
| Standard Deviation       | 2.13 ms           |
| Coefficient of Variation | 2.0%              |
| GPU Memory Usage         | 265.5 MB          |

### 📌 Notes

* CUDA acceleration enabled near real-time inference.
* GPU execution achieved approximately 41× speedup over CPU execution.
* Despite higher power consumption, GPU execution delivered substantially better energy efficiency.

---

# 📈 Comparative Analysis

## 🔋 Energy Consumption

| Scenario    | Average Power | Energy Consumption | Overhead vs Baseline |
| ----------- | ------------- | ------------------ | -------------------- |
| ⚙️ Baseline | 93.57 W       | 12.84 Wh           | —                    |
| 🖥️ CPU     | 100.63 W      | 13.81 Wh           | +7.56%               |
| ⚡ GPU       | 119.74 W      | 16.43 Wh           | +27.96%              |

## 🚀 Inference Performance

| Scenario | FPS     | Mean Latency |
| -------- | ------- | ------------ |
| 🖥️ CPU  | 0.2 FPS | 4367.87 ms   |
| ⚡ GPU    | 9.4 FPS | 106.6 ms     |

## ⚡ Energy Efficiency

| Scenario | FPS/W        |
| -------- | ------------ |
| 🖥️ CPU  | 2.28 mFPS/W  |
| ⚡ GPU    | 78.33 mFPS/W |

---

# 🧠 Key Findings

✅ GPU inference enabled practical real-time perception.

✅ CPU execution was energetically cheaper but computationally impractical.

✅ CUDA acceleration increased power consumption by approximately 27.96%.

✅ GPU execution achieved approximately 41× performance improvement.

✅ The Jetson Orin NX demonstrated favorable energy-performance trade-offs for embedded humanoid robotics.

✅ The ROS2-based BMS monitoring framework successfully quantified the energetic impact of AI inference workloads.

---

# 📚 Citation

If you use this repository in academic work, please cite:

```bibtex
@misc{souza2026g1benchmark,
  title={Energy and Performance Characterization of CPU and GPU Inference on the Unitree G1 EDU Humanoid Robot},
  author={Vitor Amadeu Souza},
  year={2026},
  note={GitHub repository}
}
```

---

# 📄 License

This project is released under the MIT License.

---

# 👨‍💻 Contact

Vitor Amadeu Souza
vitor.souza@ime.eb.br
