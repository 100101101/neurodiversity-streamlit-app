import streamlit as st
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib # For loading your trained model and threshold
# os module might be needed if model files are in a subdirectory
# import os

st.set_page_config(layout="wide")

# --- Configuration & Thresholds ---
# For Tapping Task
TAPPING_THRESHOLD_T = 24299.9

# --- EMG Model Configuration (MUST MATCH YOUR TRAINING SCRIPT) ---
WIN_MS     = 250   # window length in milliseconds
STRIDE_MS  = 125   # overlap in milliseconds
FS         = 1000  # sampling rate in Hz
win_pts    = int(WIN_MS/1000 * FS)
stride_pts = int(STRIDE_MS/1000 * FS)
EXPECTED_FEATURES = 5 # RMS, MAV, ZC, WL, SSC - from your window_features function

# --- Initialize Session State (Common and Mode-Specific) ---
# Common
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "Rhythmic Tapping Test" # Default mode

# Tapping Task Specific State (remains unchanged)
if "tap_times" not in st.session_state:
    st.session_state.tap_times = []
if "tap_test_started" not in st.session_state:
    st.session_state.tap_test_started = False
if "tap_results_calculated" not in st.session_state:
    st.session_state.tap_results_calculated = False
if "tap_intervals" not in st.session_state:
    st.session_state.tap_intervals = np.array([])
if "tap_var_iti" not in st.session_state:
    st.session_state.tap_var_iti = 0.0
if "tap_mean_iti" not in st.session_state:
    st.session_state.tap_mean_iti = 0.0
if "tap_std_iti" not in st.session_state:
    st.session_state.tap_std_iti = 0.0

# EMG Analysis Specific State - MODIFIED
if "emg_prediction_label" not in st.session_state: # Renamed for clarity
    st.session_state.emg_prediction_label = None
if "emg_subject_probability" not in st.session_state: # Store the aggregated probability
    st.session_state.emg_subject_probability = None
# st.session_state.extracted_emg_features can be removed or repurposed if needed
# For now, we'll focus on the direct model output.
if "uploaded_emg_signal_for_plot" not in st.session_state:
    st.session_state.uploaded_emg_signal_for_plot = None

# --- Load Model and Threshold (once per session) ---
@st.cache_resource # Caches the loaded model for efficiency
def load_emg_model_assets():
    try:
        # Ensure these files are in the same directory as this Streamlit script
        pipeline = joblib.load('emg_neurodiversity_pipeline.joblib')
        threshold = joblib.load('emg_neurodiversity_threshold.joblib')
        return pipeline, threshold
    except FileNotFoundError:
        st.error("CRITICAL ERROR: Model or threshold file not found. Make sure 'emg_neurodiversity_pipeline.joblib' and 'emg_neurodiversity_threshold.joblib' are in the app's root directory.")
        return None, None
    except Exception as e:
        st.error(f"Error loading EMG model assets: {e}")
        return None, None

model_pipeline, optimal_emg_threshold = load_emg_model_assets()


# --- Helper Functions ---
# Tapping Test Helper Function (remains unchanged)
def calculate_iti_stats_from_times(times_ms):
    if len(times_ms) < 2:
        return np.array([]), np.nan, np.nan, np.nan
    intervals_ms = np.diff(times_ms)
    intervals_filtered_ms = intervals_ms[(intervals_ms > 50) & (intervals_ms < 5000)]
    if len(intervals_filtered_ms) < 2:
        mean_iti_ms = np.mean(intervals_ms) if len(intervals_ms) > 0 else np.nan
        std_iti_ms = np.std(intervals_ms) if len(intervals_ms) > 0 else np.nan
        return intervals_ms, np.nan, mean_iti_ms, std_iti_ms
    var_iti_ms2 = float(np.var(intervals_filtered_ms))
    mean_iti_ms = float(np.mean(intervals_filtered_ms))
    std_iti_ms = float(np.std(intervals_filtered_ms))
    return intervals_ms, var_iti_ms2, mean_iti_ms, std_iti_ms

# calculate_emg_stats_from_series IS NO LONGER NEEDED FOR PREDICTION,
# as we use window_features now. Keep if used for other display purposes,
# otherwise, it can be removed. For this integration, we'll remove it to avoid confusion.

# Feature Extraction for EMG (MUST BE IDENTICAL TO YOUR TRAINING SCRIPT)
def emg_window_features(sig, win, step): # Renamed for clarity within Streamlit context
    feats = []
    # Ensure signal is a NumPy array for consistent processing
    if not isinstance(sig, np.ndarray):
        sig = np.array(sig, dtype=float) # Ensure it's float for calculations

    if len(sig) < win: # Not enough data for even one window
        return np.array(feats)

    for start in range(0, len(sig) - win + 1, step):
        w_slice = sig[start:start+win]
        # Ensure w_slice is a numpy array before calculations
        w = np.array(w_slice, dtype=float) # Convert slice to numpy array of floats

        rms = np.sqrt(np.mean(w**2))
        mav = np.mean(np.abs(w))
        zc  = ((w[:-1]*w[1:]) < 0).sum()
        wl  = np.sum(np.abs(np.diff(w)))
        # For SSC, ensure enough points after diff for the second diff
        if len(w) > 2:
            ssc = np.sum(((np.diff(w[:-1]) * np.diff(w[1:])) < 0).astype(int))
        else:
            ssc = 0 # Or np.nan, handle accordingly if it can be nan
        feats.append([rms, mav, zc, wl, ssc])
    return np.array(feats)


# --- Main App Layout ---
st.title("Neurodiversity Pattern Explorer")

# --- Mode Selection in Sidebar ---
st.sidebar.header("Select Analysis Mode")
app_mode_options = ("Rhythmic Tapping Test", "EMG File Analysis")
# Get current index for the radio button, default to 0 if mode is not in options (e.g. after a code change)
current_mode_index = app_mode_options.index(st.session_state.app_mode) if st.session_state.app_mode in app_mode_options else 0

app_mode = st.sidebar.radio(
    "Choose a tool:",
    app_mode_options,
    index=current_mode_index, # Set the default based on session state
    key="app_mode_selector"
)

if st.session_state.app_mode != app_mode:
    st.session_state.app_mode = app_mode
    st.rerun()

# ==============================================================================
# --- MODE 1: Rhythmic Tapping Test (remains largely unchanged) ---
# ==============================================================================
if st.session_state.app_mode == "Rhythmic Tapping Test":
    st.header("Rhythmic Tapping Test")
    st.markdown("""
    **Instructions:**
    Use one finger; keep your wrist still; tap at a comfortable, even pace.
    When ready, click **Start Tapping Test / Reset**, then tap the **space-bar** 30 times using the text box below (press Space then Enter for each tap).
    """)

    if st.button("Start Tapping Test / Reset", key="reset_tapping_button"):
        st.session_state.tap_times = []
        st.session_state.tap_test_started = True
        st.session_state.tap_results_calculated = False
        st.rerun()

    if st.session_state.tap_test_started and not st.session_state.tap_results_calculated:
        idx = len(st.session_state.tap_times)
        if idx < 30:
            input_key = f"tap_input_{idx}"
            user_input = st.text_input(f"Tap #{idx+1}/30: Press Space then Enter", key=input_key, value="", max_chars=1)
            if user_input:
                if user_input.endswith(" "):
                    current_time_ms = time.time() * 1000
                    last_time_ms = st.session_state.tap_times[-1] if st.session_state.tap_times else 0
                    if current_time_ms - last_time_ms > 50:
                        st.session_state.tap_times.append(current_time_ms)
                        st.rerun()
                else:
                    st.warning("Please press Spacebar (then Enter).")
        if len(st.session_state.tap_times) == 30:
            raw_intervals, var_iti, mean_iti, std_iti = calculate_iti_stats_from_times(st.session_state.tap_times)
            st.session_state.tap_intervals = raw_intervals
            st.session_state.tap_var_iti = var_iti
            st.session_state.tap_mean_iti = mean_iti
            st.session_state.tap_std_iti = std_iti
            st.session_state.tap_results_calculated = True
            st.session_state.tap_test_started = False
            st.rerun()

    if st.session_state.tap_results_calculated:
        st.subheader("Tapping Test Results")
        if pd.isna(st.session_state.tap_var_iti):
            st.warning("Could not calculate ITI variance (e.g., too few valid taps or all taps were identical).")
        else:
            st.write(f"Measured ITI Variance: **{st.session_state.tap_var_iti:.1f} ms²**")
            if st.session_state.tap_var_iti > TAPPING_THRESHOLD_T:
                st.error("🔴 Neurodivergent-likely")
            else:
                st.success("🟢 Neurotypical-likely")
        st.info("⚠️ **Disclaimer:** This is an experimental tool...")

        st.subheader("Tapping Consistency Chart")

        if len(st.session_state.tap_intervals) > 0:
            interval_df = pd.DataFrame({
                'Tap Interval Number': range(1, len(st.session_state.tap_intervals) + 1),
                'Time Between Taps (ms)': st.session_state.tap_intervals
            })
            plot_intervals = interval_df['Time Between Taps (ms)'].copy()
            q_low = plot_intervals.quantile(0.01)
            q_hi  = plot_intervals.quantile(0.99)
            plot_intervals_clipped = plot_intervals.clip(q_low, q_hi)
            chart_df = pd.DataFrame({'Interval (ms)': plot_intervals_clipped})
            chart_df.index = interval_df['Tap Interval Number']
            st.line_chart(chart_df)
        else:
            st.write("Not enough data to plot intervals.")
        # ... (Your existing interval statistics display) ...
        st.markdown("**Interval Statistics:**")
        if not pd.isna(st.session_state.tap_mean_iti):
            st.write(f"- Average Interval (Mean ITI): {st.session_state.tap_mean_iti:.1f} ms")
            # ... (rest of stats display)
        else:
            st.write("Interval statistics could not be calculated.")


# ==============================================================================
# --- MODE 2: EMG File Analysis --- MODIFIED FOR REAL MODEL ---
# ==============================================================================
elif st.session_state.app_mode == "EMG File Analysis":
    st.header("EMG File Analysis")
    st.markdown("""
    Upload a CSV file containing a single column of raw EMG signal values.
    The app will extract features from sliding windows, use a trained Random Forest model
    to predict probabilities for each window, average these probabilities for a subject-level score,
    and then classify the subject based on an optimized threshold.
    """)

    uploaded_emg_file = st.file_uploader("Upload your EMG data CSV file", type=["csv"], key="emg_file_uploader_real_model")

    if uploaded_emg_file is not None:
        # Reset previous EMG analysis results on new upload
        st.session_state.emg_prediction_label = None
        st.session_state.emg_subject_probability = None
        st.session_state.uploaded_emg_signal_for_plot = None
        
        if model_pipeline is None or optimal_emg_threshold is None:
            st.error("The EMG prediction model is not loaded. Cannot proceed.")
        else:
            try:
                df_uploaded = pd.read_csv(uploaded_emg_file)
                if df_uploaded.empty:
                    st.error("The uploaded CSV file is empty.")
                else:
                    st.write("Uploaded EMG data preview (first 5 rows):")
                    st.dataframe(df_uploaded.head())

                    emg_column_name = 'EMG'
                    if emg_column_name not in df_uploaded.columns:
                        if len(df_uploaded.columns) > 0:
                            emg_column_name = df_uploaded.columns[0]
                            st.info(f"No 'EMG' column found. Using the first column: '{emg_column_name}' for EMG data.")
                        else:
                            st.error("The uploaded CSV file has no columns.")
                            st.stop()
                    
                    emg_signal_raw = df_uploaded[emg_column_name].copy()
                    emg_signal_numeric = pd.to_numeric(emg_signal_raw, errors='coerce').dropna()
                    st.session_state.uploaded_emg_signal_for_plot = emg_signal_numeric # Store for plotting

                    if emg_signal_numeric.empty or len(emg_signal_numeric) < win_pts:
                        st.error(f"Not enough valid numeric data points in the EMG signal (need at least {win_pts} for one window). Found {len(emg_signal_numeric)} valid points after cleaning.")
                    else:
                        # 1. Windowing and Feature Extraction
                        subject_window_features = emg_window_features(emg_signal_numeric.values, win_pts, stride_pts)

                        if subject_window_features.size == 0 or subject_window_features.shape[1] != EXPECTED_FEATURES:
                            st.error(f"Could not extract valid window features. Ensure signal is long enough and clean. Expected {EXPECTED_FEATURES} features per window.")
                        else:
                            # 2. Predict probabilities using the loaded pipeline
                            window_probabilities = model_pipeline.predict_proba(subject_window_features)[:, 1]

                            # 3. Aggregate to subject-level probability
                            st.session_state.emg_subject_probability = np.mean(window_probabilities)

                            # 4. Classify based on the optimal threshold
                            prediction_value = 1 if st.session_state.emg_subject_probability >= optimal_emg_threshold else 0
                            st.session_state.emg_prediction_label = "Neurodivergent" if prediction_value == 1 else "Neurotypical"
            
            except pd.errors.EmptyDataError:
                st.error("The uploaded CSV file is empty.")
            except Exception as e:
                st.error(f"An error occurred while processing the EMG file: {e}")
                st.error("Please ensure the CSV file has a single column of numerical EMG data, optionally with a header like 'EMG'.")

    # --- Display EMG Analysis Results Section (Using Real Model) ---
    if st.session_state.emg_prediction_label is not None: # Check if prediction was made
        st.markdown("---")
        st.subheader("EMG Analysis Prediction:")
        
        display_prob = st.session_state.emg_subject_probability if st.session_state.emg_subject_probability is not None else "N/A"
        
        if st.session_state.emg_prediction_label == "Neurodivergent":
            st.error(f"🔴 Predicted Group: **Neurodivergent**")
        elif st.session_state.emg_prediction_label == "Neurotypical":
            st.success(f"🟢 Predicted Group: **Neurotypical**")
        else: # Error message
            st.warning(f"Prediction Status: {st.session_state.emg_prediction_label}")

        if isinstance(display_prob, float):
             st.write(f"Subject's Average Window Probability Score: **{display_prob:.4f}**")
             st.write(f"(Using probability threshold: {optimal_emg_threshold:.4f}. Scores >= threshold are classified as Neurodivergent)")
        else:
             st.write(f"Subject's Average Window Probability Score: {display_prob}")


    if st.session_state.uploaded_emg_signal_for_plot is not None and not st.session_state.uploaded_emg_signal_for_plot.empty:
        st.markdown("**Visualization of Your Uploaded EMG Signal:**")
        try:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(st.session_state.uploaded_emg_signal_for_plot.values) 
            ax.set_title("Uploaded EMG Data")
            ax.set_xlabel("Sample Number")
            ax.set_ylabel("EMG Amplitude")
            ax.grid(True)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Could not plot the EMG signal: {e}")

    st.markdown("---")
    st.info(f"""
        ⚠️ **Important Disclaimer for EMG Analysis:**
        This prediction is based on a Random Forest model trained on a research dataset (N=43 subjects)
        using 5 features (RMS, MAV, ZC, WL, SSC) extracted from {WIN_MS}ms sliding windows.
        The model aggregates window predictions to a subject-level score, which is then compared against
        an optimized threshold ({optimal_emg_threshold:.4f}) to make the final classification.
        Reported subject-level accuracy on the test data for this model configuration was approximately 97.7%.

        This tool is for **experimental and illustrative purposes only** and is **NOT a medical diagnosis**.
        Neurodiversity is complex. Please consult with qualified professionals for any diagnostic concerns.
        """)