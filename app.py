import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import contextily as ctx
import geopandas as gpd
from shapely.geometry import Point
import base64
from streamlit_js_eval import streamlit_js_eval

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Kalman Filter Flight Simulation")

# --- Theme Detection & Configuration ---
try:
    is_dark_mode = streamlit_js_eval(js_expressions="window.matchMedia('(prefers-color-scheme: dark)').matches", key="theme")
except Exception:
    is_dark_mode = False

if is_dark_mode:
    THEME = {
        "bg_app": "#0E1117", "bg_card": "#1F2937", "text_primary": "#FAFAFA",
        "text_secondary": "#D1D5DB", "border": "#374151", "accent": "#3B82F6",
        "map_provider": ctx.providers.CartoDB.DarkMatter, "plot_bg": "#1F2937",
        "plot_text": "white", "c_ideal": "#60A5FA", "c_noise": "#FBBF24",
        "c_kalman": "#818CF8", "c_aircraft": "#34D399"
    }
    THEME_NAME = 'dark'
else:
    THEME = {
        "bg_app": "#F0F2F6", "bg_card": "#FFFFFF", "text_primary": "#0D1B2A",
        "text_secondary": "#495057", "border": "#E5E7EB", "accent": "#007BFF",
        "map_provider": ctx.providers.CartoDB.Positron, "plot_bg": "white",
        "plot_text": "black", "c_ideal": "#3B82F6", "c_noise": "#F97316",
        "c_kalman": "#6366F1", "c_aircraft": "#14B8A6"
    }
    THEME_NAME = 'light'

# --- Custom CSS ---
CSS = f"""
[data-testid="stAppViewContainer"] {{ background-color: {THEME['bg_app']}; }}
h1, h2, h3, h4, h5, h6 {{ color: {THEME['text_primary']}; }}
.stMarkdown, .stSlider label {{ color: {THEME['text_secondary']}; }}
section[data-testid="stSidebar"] + div [data-testid="stVerticalBlock"] {{
    background-color: {THEME['bg_card']}; border-radius: 12px;
    padding: 2rem; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border: 1px solid {THEME['border']};
}}
h2 {{ border-bottom: 2px solid {THEME['border']}; padding-bottom: 10px; }}
[data-testid="stSidebar"] {{ background-color: {THEME['bg_card']}; border-right: 1px solid {THEME['border']}; }}
.stButton>button {{ background-color: {THEME['accent']}; color: white; border: none; }}
.stButton>button:hover {{ filter: brightness(85%); }}
"""
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


# --- Core GIF Generation Function (with contrast fixes) ---
@st.cache_data(show_spinner=False)
def generate_animation_gif(gps_noise, flight_instability, theme_name, theme_colors):
    """Generates a GIF that matches the provided theme, with all text colors correctly set."""
    dt = 1.0; num_steps = 90
    F = np.array([[1, dt, 0,  0], [0,  1, 0,  0], [0,  0, 1, dt], [0,  0, 0,  1]])
    H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
    Q = np.eye(4) * flight_instability**2; R = np.eye(2) * gps_noise**2
    
    start_p, turn1_p, turn2_p, end_p = Point(-122.47, 37.805), Point(-122.42, 37.81), Point(-122.44, 37.78), Point(-122.41, 37.77)
    points_m = gpd.GeoDataFrame(geometry=[start_p, turn1_p, turn2_p, end_p], crs="EPSG:4326").to_crs(epsg=3857).geometry
    start_m, turn1_m, turn2_m, end_m = points_m[0], points_m[1], points_m[2], points_m[3]

    true_states = np.zeros((4, num_steps)); turn_step_1, turn_step_2 = 30, 60
    vel1_x, vel1_y = (turn1_m.x - start_m.x) / (turn_step_1 * dt), (turn1_m.y - start_m.y) / (turn_step_1 * dt)
    true_states[:, 0] = [start_m.x, vel1_x, start_m.y, vel1_y]
    vel2_x, vel2_y = (turn2_m.x - turn1_m.x) / ((turn_step_2 - turn_step_1) * dt), (turn2_m.y - turn1_m.y) / ((turn_step_2 - turn_step_1) * dt)
    vel3_x, vel3_y = (end_m.x - turn2_m.x) / ((num_steps - turn_step_2) * dt), (end_m.y - turn2_m.y) / ((num_steps - turn_step_2) * dt)
    for k in range(1, num_steps):
        if k == turn_step_1: true_states[1, k-1], true_states[3, k-1] = vel2_x, vel2_y
        if k == turn_step_2: true_states[1, k-1], true_states[3, k-1] = vel3_x, vel3_y
        true_states[:, k] = F @ true_states[:, k-1] + np.random.multivariate_normal(np.zeros(4), Q)
    measurements = np.zeros((2, num_steps))
    for k in range(num_steps): measurements[:, k] = H @ true_states[:, k] + np.random.multivariate_normal(np.zeros(2), R)
    
    x_hat = np.array([start_m.x, vel1_x, start_m.y, vel1_y]); P = np.eye(4) * 500.0
    kalman_estimates = np.zeros((4, num_steps))
    for k in range(num_steps):
        if k == turn_step_1: x_hat[1], x_hat[3] = vel2_x, vel2_y
        if k == turn_step_2: x_hat[1], x_hat[3] = vel3_x, vel3_y
        x_hat_pred = F @ x_hat; P_pred = F @ P @ F.T + Q; y_k = measurements[:, k] - H @ x_hat_pred; S_k = H @ P_pred @ H.T + R
        K_k = P_pred @ H.T @ np.linalg.inv(S_k); x_hat = x_hat_pred + K_k @ y_k; P = (np.eye(4) - K_k @ H) @ P_pred
        kalman_estimates[:, k] = x_hat

    fig, ax = plt.subplots(figsize=(12, 12)); fig.patch.set_facecolor(theme_colors['plot_bg'])
    bounds = gpd.GeoDataFrame(geometry=[Point(-122.50, 37.76), Point(-122.38, 37.82)], crs="EPSG:4326").to_crs(epsg=3857).total_bounds
    ax.set_xlim(bounds[0], bounds[2]); ax.set_ylim(bounds[1], bounds[3]); ax.set_xticks([]); ax.set_yticks([])
    ctx.add_basemap(ax, crs='EPSG:3857', source=theme_colors['map_provider'])
    ax.set_title("Kalman Filter - Predicting Flight Location", fontsize=20, weight='bold', color=theme_colors['plot_text'])
    
    ax.plot(true_states[0, :], true_states[2, :], '--', color=theme_colors['c_ideal'], linewidth=1.5, alpha=0.8, label="Ideal Flight Plan")
    measurement_trail, = ax.plot([], [], 'X', markersize=10, color=theme_colors['c_noise'], alpha=0.7, label="Recent GPS Signals", linestyle='None')
    kalman_path, = ax.plot([], [], '-', linewidth=4, color=theme_colors['c_kalman'], label="Kalman Filter Estimate")
    current_pos, = ax.plot([], [], 'o', markersize=14, color=theme_colors['c_aircraft'], zorder=10, label="Current Position (eVTOL)", markeredgecolor=theme_colors['plot_bg'], markeredgewidth=1.5)
    
    props = dict(boxstyle='round,pad=0.5', facecolor=theme_colors['bg_card'], alpha=0.9, edgecolor=theme_colors['border'])
    # FIX: Explicitly set text color for the info box
    ax.text(0.02, 0.98, "Powered by a State-Space Model", transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props, color=theme_colors['text_primary'])
    predict_blinker = ax.text(0.98, 0.98, "PREDICTING", transform=ax.transAxes, fontsize=12, fontweight='bold', ha='right', va='top')
    update_blinker = ax.text(0.98, 0.94, "UPDATING", transform=ax.transAxes, fontsize=12, fontweight='bold', ha='right', va='top')
    legend = ax.legend(loc='lower left', fontsize='medium')
    # FIX: Explicitly set text color for the legend
    plt.setp(legend.get_texts(), color=theme_colors['plot_text'])

    def animate(i):
        start_index = max(0, i - 10); measurement_trail.set_data(measurements[0, start_index:i+1], measurements[1, start_index:i+1])
        kalman_path.set_data(kalman_estimates[0, :i+1], kalman_estimates[2, :i+1])
        pos = (kalman_estimates[0, i], kalman_estimates[2, i]); current_pos.set_data([pos[0]], [pos[1]])
        if i % 2 == 0: predict_blinker.set_color(theme_colors['c_aircraft']); update_blinker.set_color('#6B7280') # Faded gray
        else: predict_blinker.set_color('#6B7280'); update_blinker.set_color(theme_colors['c_ideal'])
        return kalman_path, current_pos, predict_blinker, update_blinker, measurement_trail

    anim = FuncAnimation(fig, animate, frames=num_steps, interval=80, blit=False)
    gif_path = f"kalman_simulation_{theme_name}.gif"
    anim.save(gif_path, writer='pillow', fps=15, dpi=120)
    plt.close(fig)
    return gif_path

# --- Main App UI ---
st.title("Kalman Filter Flight Simulation")
st.sidebar.header("Simulation Parameters")
r_std_dev = st.sidebar.slider("GPS Noise (Inaccuracy)", min_value=1.0, max_value=50.0, value=25.0, step=1.0, help="Higher values mean more scattered GPS readings.")
q_std_dev = st.sidebar.slider("Flight Instability", min_value=0.0, max_value=5.0, value=0.7, step=0.1, help="Simulates unpredictable factors like wind gusts.")
generate_button = st.sidebar.button("Generate Flight Simulation")

col1, col2 = st.columns([2, 1.2])

with col1:
    st.header("Navigation Visualization")
    if generate_button:
        with st.spinner("Generating high-quality animation... This may take a moment."):
            gif_path = generate_animation_gif(r_std_dev, q_std_dev, THEME_NAME, THEME)
            file_ = open(gif_path, "rb"); contents = file_.read()
            data_url = base64.b64encode(contents).decode("utf-8"); file_.close()
            st.image(f"data:image/gif;base64,{data_url}", use_column_width=True)
    else:
        st.info("Adjust parameters in the sidebar and click 'Generate' to begin.")

with col2:
    st.header("How It Works")
    
    # --- RESTRUCTURED EXPLANATION ---
    st.subheader("The Intuition")
    st.markdown("""
    Imagine you're trying to track a frisbee in a gusty wind while your vision is a bit blurry.
    
    1.  **Predict:** Based on the frisbee's last known direction and speed, you predict where it will be a split-second later. This is your "best guess" based on physics.
    
    2.  **Update:** You then get a quick, blurry glimpse of the frisbee (your "noisy measurement").
    
    The Kalman Filter is a mathematical way to optimally **blend your prediction with your blurry glimpse**. If you're very confident in your prediction, you'll mostly ignore the blurry view. But if the glimpse is clearer, you'll use it to correct your prediction. This **Predict-Update** cycle happens continuously, resulting in a remarkably smooth and accurate track of the frisbee's true location.
    """)

    st.subheader("The Technical Details: State-Space Model")
    st.markdown("The filter uses two core equations to achieve this:")
    
    st.markdown("##### 1. The State Equation (Prediction Model)")
    st.markdown("Describes the physics of motion. It answers: *'Where will the aircraft be next, based on its current velocity and direction?'*")
    st.latex(r'''x_k = F_k x_{k-1} + w_k''')
    
    st.markdown("##### 2. The Measurement Equation (Update Model)")
    st.markdown("Relates the state to a sensor reading. It answers: *'What does the noisy GPS signal tell me about the aircraft's state?'*")
    st.latex(r'''z_k = H_k x_k + v_k''')
    
    st.info("The filter's brilliance is its ability to use the uncertainty of both the prediction ($w_k$) and the measurement ($v_k$) to calculate the optimal blend at every step.")