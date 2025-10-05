# Spectrum Intelligence: AI-Powered RF Fingerprinting

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Frontend](https://img.shields.io/badge/Frontend-React-blue?logo=react)](https://react.dev/)
[![Backend](https://img.shields.io/badge/Backend-FastAPI-green?logo=fastapi)](https://fastapi.tiangolo.com/)
[![AI](https://img.shields.io/badge/AI-TensorFlow-orange?logo=tensorflow)](https://www.tensorflow.org/)

**A cloud-native platform for real-time radio frequency (RF) signal classification and anomaly detection using deep learning.**

---

## Live Demo

**Experience Spectrum Intelligence live:** [https://rf-fingerprinting.vercel.app/](https://rf-fingerprinting.vercel.app/)

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

## Technology Stack

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

---

## Architecture Diagram Link

[https://lucid.app/lucidchart/4d156ecd-11b6-48f2-a61e-88e65ae2bf05/edit?view_items=pePW5jqhf95l%2CpePW540d6_Mw%2CpePWGPR7Oa1S%2CpePWH2NulFd9%2CpePWBe3LywiN%2CpePWdfhz5VVs%2CpePWWZjk7nvz%2CpePWmtuC7sVu%2CpePWo8Vv8f6c%2CpePWhV4n0J_i%2CpePWu1lQ~YzI%2CpePWPvd~B](https://lucid.app/lucidchart/4d156ecd-11b6-48f2-a61e-88e65ae2bf05/edit?view_items=pePW5jqhf95l%2CpePW540d6_Mw%2CpePWGPR7Oa1S%2CpePWH2NulFd9%2CpePWBe3LywiN%2CpePWdfhz5VVs%2CpePWWZjk7nvz%2CpePWmtuC7sVu%2CpePWo8Vv8f6c%2CpePWhV4n0J_i%2CpePWu1lQ~YzI%2CpePWPvd~B)

---

 ## How to Use the Live Demo

1.  **Navigate** to the live application.
2.  **Select a Sample:** In the "Test with a Fast Demo Sample" card, click one of the buttons (e.g., "Load Stealth Demo").
3.  **Analyze:** Click the "Analyze Signal" button and wait for the results to appear on the right.
4.  **Interpret the Results:**
    * The **Analysis Complete** card shows the final classification and whether an anomaly was detected.
    * The **Reconstruction Error Analysis** plot visually confirms the findings. A clean signal will have very low yellow bars, while an anomalous signal will have prominent yellow spikes.

---

## Local Development Setup

### Prerequisites
- Node.js and npm
- Python 3.11+ and pip

### Backend Setup
1.  Navigate to the `backend` directory:
    ```sh
    cd backend
    ```
2.  Install the required Python packages:
    ```sh
    pip install -r requirements.txt
    ```
3.  Run the FastAPI server:
    ```sh
    uvicorn main:app --reload
    ```
    The backend will be running at `http://127.0.0.1:8000`.

### Frontend Setup
1.  Navigate to the `frontend` directory:
    ```sh
    cd frontend
    ```
2.  Install the Node.js dependencies:
    ```sh
    npm install
    ```
3.  Run the development server:
    ```sh
    npm run dev
    ```
    The frontend will be available at `http://localhost:5173` (or another port if 5173 is busy).

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---


