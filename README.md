# Adversarial Robustness in Network Intrusion Detection

This repository contains the source code for an empirical analysis of adversarial attacks and defenses on a neural network-based Network Intrusion Detection System (NIDS).

## Overview

The project demonstrates:
1.  The vulnerability of a standard Multi-Layer Perceptron (MLP) model to white-box attacks (PGD).
2.  The effectiveness of Adversarial Training as a defense mechanism.
3.  A comprehensive evaluation of model robustness under varying attack strengths.

## Project Structure

-   `/data`: Holds the NSL-KDD dataset files.
-   `/src`: Contains the main experiment logic (`main.py`) and utility functions (`utils.py`).
-   `/results`: The output directory for generated figures and CSV data.
-   `/notebooks`: Contains scripts to generate all figures from the results.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone [your-repo-link]
    cd your-project-name
    ```

2.  **Create and activate the Conda environment:**
    ```bash
    conda env create -f environment.yml
    conda activate adversarial-ids
    ```

3.  **Download the dataset:**
    -   Download the NSL-KDD dataset (`KDDTrain+.txt` and `KDDTest+.txt`).
    -   Place both files inside the `/data` directory.

## Usage

1.  **Run the main experiment:**
    This will train the models, perform the robustness analysis, and save the results to a CSV file in the `/results` directory.
    ```bash
    python src/main.py
    ```

2.  **Generate the figures:**
    Run the plotting scripts to generate the figures from the experiment. The figures will be saved in the `/results` directory.
    ```bash
    python notebooks/plot_accuracy_comparison.py
    python notebooks/plot_f1_heatmap.py
    python notebooks/plot_robustness_curve.py
    ```
