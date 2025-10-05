# Spectrum Intelligence: AI-Powered RF Fingerprinting

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Frontend](https://img.shields.io/badge/Frontend-React-blue?logo=react)](https://react.dev/)
[![Backend](https://img.shields.io/badge/Backend-FastAPI-green?logo=fastapi)](https://fastapi.tiangolo.com/)
[![AI](https://img.shields.io/badge/AI-TensorFlow-orange?logo=tensorflow)](https://www.tensorflow.org/)

**A cloud-native platform for real-time radio frequency (RF) signal classification and anomaly detection using deep learning.**

---

## Live Demo

**Experience Spectrum Intelligence live:** [https://rf-fingerprinting.vercel.app/](https://rf-fingerprinting.vercel.app/)

![Spectrum Intelligence UI](https://i.imgur.com/your-screenshot-url.png)

---

## Project Overview

In an increasingly crowded wireless landscape, distinguishing between legitimate and potentially malicious signals is a critical security challenge. Spectrum Intelligence is an AI-powered web service that addresses this by providing instant analysis of RF signals.

The system uses a **dual-model AI core** to perform two-in-one diagnostics:
1.  **RF Fingerprinting:** A Convolutional Neural Network (CNN) classifies known signal types based on their unique modulation characteristics.
2.  **Zero-Day Threat Detection:** An unsupervised Autoencoder model detects anomalies by identifying signals that deviate from expected patterns.

This prototype demonstrates a robust, end-to-end solution, from its intuitive user interface to its powerful, cloud-hosted AI backend. The models were trained on the industry-standard **RadioML 2016.10a** dataset, enabling them to handle real-world signal variations.

### Key Features

-   **Dual-Model AI Analysis:** Combines classification of known signals with unsupervised anomaly detection for comprehensive threat analysis.
-   **Explainable AI (XAI) Visualization:** Features a "Reconstruction Error Plot" that visually pinpoints the exact location of anomalous behavior within a signal, explaining *why* it was flagged.
-   **Cloud-Native Platform:** A fully web-based service, allowing operators to analyze signals from any device with a browser, without needing specialized local software.
-   **Fast Demo Samples:** Includes pre-loaded, optimized sample files to allow for quick and easy demonstration of the system's capabilities.

---

## Technology Stack & Architecture

The project is built on a modern, decoupled architecture, ensuring scalability and ease of maintenance.

-   **Frontend:**
    -   **Framework:** React
    -   **Build Tool:** Vite
    -   **Styling:** Tailwind CSS
    -   **Visualization:** Plotly.js
    -   **Deployment:** Vercel

-   **Backend:**
    -   **Language:** Python
    -   **Framework:** FastAPI
    -   **AI/ML Library:** TensorFlow (Keras)
    -   **Deployment:** Render

### Architecture Diagram

```mermaid
graph TD
    subgraph "User"
        A[<fa:fa-user> User in Browser]
    end

    subgraph "Frontend (Hosted on Vercel)"
        B[React Web Application]
        C[UI: File Upload & Sample Selection]
        D[Plotly.js Visualization]
    end

    subgraph "Backend (Hosted on Render)"
        E[FastAPI Service]
        subgraph "AI Core (TensorFlow/Keras)"
            F[Anomaly Detector <br> (Autoencoder)]
            G[Classifier <br> (CNN)]
            H[Intelligence Synthesis <br> - Calculates Error <br> - Overrides Classification]
        end
    end

    A -- "Interacts with" --> B
    B -- "User selects/uploads .npy file" --> C
    C -- "HTTPS POST Request" --> E
    E --> F & G
    F & G --> H
    H -- "Processed JSON Result" --> E
    E -- "HTTPS JSON Response" --> D
    D -- "Displays Plots & Results" --> B
