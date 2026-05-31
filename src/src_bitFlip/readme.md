# WiMAX PGDBF Decoder Project

## Overview
I have implemented a **Probabilistic Gradient Descent Bit Flipping (PGDBF)** decoder for the WiMAX 5/6 rate LDPC code. My implementation focuses on solving the "trapping set" problem common in deterministic bit-flipping algorithms.

## Key Technical Features
* **PGDBF Algorithm:** I introduced an 80% probabilistic flipping rule to help the decoder escape local minima.
* **Energy Tolerance:** I implemented a 0.5 energy threshold, allowing the decoder to group and flip multiple bits simultaneously.
* **Endurance:** I configured the decoder for up to 500 iterations to ensure robustness in high-noise channels.

## How to Run
### 1. Requirements
Ensure you have `numpy`, `pandas`, and `matplotlib` installed:
pip install numpy pandas matplotlib

### 2. Decoding a Signal
To run the primary decoder and test a single frame:

Bash
python bitFlip.py
### 3. Generating Statistics
To run the BER simulation and generate the simulation_results.csv data:

Bash
python generate_stats.py
### 4. Visualizing Performance
To generate the BER waterfall graph (ber_curve.png):

Bash
python plot_ber.py