# Kalman-Flight-Tracker
A Python-based simulation visualizing the Kalman Filter for high-precision object tracking and navigation amidst noisy sensor data.

# Kalman Flight Tracker

An interactive Streamlit application that provides a simulation of the Kalman filter for high-precision aircraft navigation. This project demonstrates how the algorithm takes erratic, noisy sensor readings (simulating GPS) and produces a smooth, reliable estimate of an object's true trajectory.

![Kalman Filter Demo GIF](static/demo.mp4)

---

## Features

* **Interactive Simulation:** Adjust parameters like GPS noise and flight instability to see how the filter performs under different conditions.
* **High-Quality Animation:** Generates a smooth, high-performance GIF of the flight path on the fly, providing a clear and compelling visualization.
* **Dynamic S-Path Route:** The simulation follows a complex, S-shaped flight path over San Francisco to better showcase the filter's tracking capabilities.
* **Intuitive Explanations:** A side-by-side view explains the core concepts of the Kalman filter, starting with a simple intuition, then providing the underlying mathematical model.
* **Polished & Professional UI:** A clean, modern interface designed for a great user experience.

---

## 🛠️ Getting Started

To get this application running on your local machine, follow these steps.

### Prerequisites

Ensure you have Python 3.8+ installed.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/Kalman-Flight-Tracker.git](https://github.com/YOUR_USERNAME/Kalman-Flight-Tracker.git)
    cd Kalman-Flight-Tracker
    ```

2.  **Create a `requirements.txt` file** with the following content:
    ```
    streamlit
    numpy
    matplotlib
    contextily
    geopandas
    shapely
    streamlit_js_eval
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

---

## ▶️ How to Run the App

Once the installation is complete, you can launch the Streamlit application.

1.  Make sure you are in the root directory of the project (`Kalman-Flight-Tracker`).
2.  Run the following command in your terminal:
    ```bash
    streamlit run app.py
    ```

Your web browser will automatically open with the interactive application. Adjust the sliders in the sidebar and click "Generate Flight Simulation" to see the Kalman filter in action!

---

## 💡 Inspiration

This project was inspired by a visit to the **Databricks + AI Summit**, where a Joby Aviation eVTOL was on display. Receiving a postcard from one of their test flights sparked my curiosity to explore and visualize the data science behind autonomous navigation.