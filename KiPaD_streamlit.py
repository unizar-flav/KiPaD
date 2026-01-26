"""
KiPaD - Kinetic Parameters Determination
Streamlit Application

A tool for determining kinetic parameters from time-resolved spectroscopic data
using Singular Value Decomposition (SVD) and non-linear fitting.
"""

import csv
import io
import zipfile
from datetime import datetime
import tempfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.linalg import svd
from scipy.stats import probplot
import streamlit as st

from functions.general import argLeastSquares, procesa
from functions.specific import (
    read_spectra,
    slice_dataset,
    scree_plot_with_fit,
    entropy_selection,
    broken_stick_method,
    matrix_approximation,
    deriv_conc,
    Model_spectra,
)


# Set page config
st.set_page_config(
    page_title="KiPaD",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and Description
st.title("KiPaD - Kinetic Parameters Determination")
st.markdown("*Determination of kinetic parameters from time-resolved spectroscopic data*")


# ==================== HELPER FUNCTIONS ====================

def show_footer():
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    by Mario Asensio Franco

    with contributions from:
    - Sergio Boneta Martínez
    - José Carlos Ciria Coscolluela
    - Milagros Medina Trullenque

    GPLv3 © 2024-2026 \\
    @ Universidad de Zaragoza
    """)
    
@st.cache_data
def cached_svd(data_array):
    """Cached wrapper for SVD calculation"""
    return svd(data_array, full_matrices=False)

def create_plotly_2d_plot(df, title, x_axis, y_axis, legend_title):
    """Creates a 2D Plotly plot from the given DataFrame."""
    # Handle duplicate indices by averaging them
    if df.index.duplicated().any():
        df = df.groupby(df.index).mean()

    fig = go.Figure()

    # Generate a rainbow colorscale
    n_lines = len(df.columns)
    colors = [f'hsl({h}, 70%, 50%)' for h in np.linspace(0, 300, n_lines)]

    for idx, col in enumerate(df.columns):
        fig.add_trace(go.Scatter(
            x=df.index.astype(float),
            y=df[col].values,
            mode='lines',
            name=str(col),
            line=dict(color=colors[idx], width=2),
            visible=True  # Show all traces by default
        ))

    fig.update_layout(
        title=title,
        xaxis_title=x_axis,
        yaxis_title=y_axis,
        legend_title=legend_title,
        height=600,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02
        )
    )

    return fig


def create_plotly_comparison_plot(df1, df2, title, x_axis, y_axis, legend_title, df1_label, df2_label):
    """Creates a Plotly plot comparing two DataFrames."""
    # Handle duplicate indices by averaging them
    if df1.index.duplicated().any():
        df1 = df1.groupby(df1.index).mean()
    if df2.index.duplicated().any():
        df2 = df2.groupby(df2.index).mean()

    fig = go.Figure()

    # Colors for the two dataframes
    df1_color = '#1f77b4'  # Blue
    df2_color = '#ff7f0e'  # Orange

    for col in df1.columns:
        # Add df1 trace
        fig.add_trace(go.Scatter(
            x=df1.index.astype(float),
            y=df1[col].values,
            mode='lines',
            name=f'{df1_label} - {col}',
            line=dict(color=df1_color, width=2),
            visible='legendonly'
        ))
        # Add df2 trace
        fig.add_trace(go.Scatter(
            x=df2.index.astype(float),
            y=df2[col].values,
            mode='lines',
            name=f'{df2_label} - {col}',
            line=dict(color=df2_color, width=2, dash='dash'),
            visible='legendonly'
        ))

    # Make first pair visible
    if len(fig.data) >= 2:
        fig.data[0].visible = True
        fig.data[1].visible = True

    fig.update_layout(
        title=title,
        xaxis_title=x_axis,
        yaxis_title=y_axis,
        legend_title=legend_title,
        height=600,
        showlegend=True
    )

    return fig


def create_plotly_qq_plot(residuals_dict, title):
    """Creates a QQ plot using Plotly for one or multiple series."""
    fig = go.Figure()

    # Handle backward compatibility: if a Series is passed, convert to dict
    if isinstance(residuals_dict, pd.Series):
        residuals_dict = {"Residuals": residuals_dict}

    # Generate colors using a colormap
    colors = px.colors.qualitative.Plotly

    all_theoretical = []

    for idx, (label, residuals_series) in enumerate(residuals_dict.items()):
        residuals = residuals_series.dropna().values
        (qq_theoretical, qq_residuals), (slope, intercept, _) = probplot(residuals, dist="norm")

        all_theoretical.extend(qq_theoretical)
        color = colors[idx % len(colors)]

        # Add scatter for QQ plot
        fig.add_trace(go.Scatter(
            x=qq_theoretical,
            y=qq_residuals,
            mode='markers',
            name=str(label),
            marker=dict(color=color, size=8)
        ))

    # Add single theoretical line based on all data
    if all_theoretical:
        x_line = [min(all_theoretical), max(all_theoretical)]
        y_line = x_line  # Standard normal: y = x

        fig.add_trace(go.Scatter(
            x=x_line,
            y=y_line,
            mode='lines',
            name='Theoretical',
            line=dict(color='black', width=2, dash='dash')
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Theoretical Quantiles",
        yaxis_title="Residual Quantiles",
        height=500,
        showlegend=True
    )

    return fig


def create_scree_plot_plotly(singular_values, threshold):
    """Creates a scree plot using Plotly with linear fit analysis."""
    from sklearn.linear_model import LinearRegression

    n_values = len(singular_values)
    SSVs = 0
    X_final, y_final_pred = None, None

    # Iterate through singular values, trying linear fits
    for i in range(2, n_values + 1):
        X = np.arange(1, i + 1).reshape(-1, 1)
        y = singular_values[:i]

        model = LinearRegression().fit(X, y)
        r_squared = model.score(X, y)

        if r_squared < threshold:
            SSVs = i - 1
            break
        else:
            SSVs = i
        X_final = np.arange(1, SSVs + 1).reshape(-1, 1)
        y_final_pred = model.predict(X_final)

    fig = go.Figure()

    # Plot singular values
    indices = np.arange(1, n_values + 1)
    fig.add_trace(go.Scatter(
        x=indices,
        y=singular_values,
        mode='markers',
        name='Singular Values',
        marker=dict(color='blue', size=10)
    ))

    # Plot linear fit
    if X_final is not None and y_final_pred is not None:
        fig.add_trace(go.Scatter(
            x=X_final.flatten(),
            y=y_final_pred,
            mode='lines',
            name='Linear Fit',
            line=dict(color='red', width=2, dash='dash')
        ))

    # Add vertical line at cutoff
    fig.add_vline(x=SSVs, line_dash="dash", line_color="green",
                  annotation_text=f"SSVs = {SSVs}", annotation_position="top right")

    fig.update_layout(
        title="Scree Plot with Linear Fit",
        xaxis_title="Singular Value Index",
        yaxis_title="Singular Values",
        height=500
    )

    return fig, SSVs


def create_broken_stick_plot_plotly(singular_values):
    """Creates a broken stick plot using Plotly."""
    k = len(singular_values)

    # Calculate broken stick values
    broken_stick = np.zeros(k)
    for i in range(1, k + 1):
        broken_stick[i - 1] = (1 / k) * np.sum([1 / j for j in range(i, k + 1)])

    # Normalize singular values
    singular_values_squared_n = singular_values**2 / np.sum(singular_values**2)

    fig = go.Figure()

    indices = np.arange(1, k + 1)

    # Plot singular values
    fig.add_trace(go.Scatter(
        x=indices,
        y=singular_values_squared_n,
        mode='lines+markers',
        name='Singular Values',
        line=dict(color='blue', width=2),
        marker=dict(size=8)
    ))

    # Plot broken stick values
    fig.add_trace(go.Scatter(
        x=indices,
        y=broken_stick,
        mode='lines+markers',
        name='Broken Stick',
        line=dict(color='red', width=2, dash='dash'),
        marker=dict(size=8)
    ))

    # Determine SSVs
    SSVs = 0
    for i in range(k):
        if singular_values_squared_n[i] > broken_stick[i]:
            SSVs += 1
        else:
            break

    fig.update_layout(
        title="Broken Stick Model vs Singular Values",
        xaxis_title="Index",
        yaxis_title="Proportion of Variance",
        height=500
    )

    return fig, SSVs


# ==================== SIDEBAR & DATA LOADING ====================

st.sidebar.header("1. Upload Spectra")
uploaded_files = st.sidebar.file_uploader(
    "Upload CSV files",
    help="Upload one or multiple time-resolved spectra CSV files.",
    accept_multiple_files=True,
    type=['csv']
)

if not uploaded_files:
    st.info("Please upload spectra CSV files in the sidebar to begin.")
    show_footer()
    st.stop()

# Initialize session state
if 'datos_org' not in st.session_state:
    st.session_state.datos_org = None
if 'datos' not in st.session_state:
    st.session_state.datos = None
if 'file_name' not in st.session_state:
    st.session_state.file_name = None
if 'datos_approx_df' not in st.session_state:
    st.session_state.datos_approx_df = None
if 'sol' not in st.session_state:
    st.session_state.sol = None
if 'Model' not in st.session_state:
    st.session_state.Model = None

with st.spinner("Reading files..."):
    # Save uploaded files temporarily
    temp_files = []
    for uploaded_file in uploaded_files:
        # Check if file is already closed (if rerun)
        try:
            uploaded_file.seek(0)
            content = uploaded_file.read().decode('utf-8')
        except ValueError:
            # Re-upload or error handling if file pointer closed
            content = ""
            
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='w') as tmp:
            tmp.write(content)
            temp_files.append(tmp.name)
            # Reset pointer
            try:
                uploaded_file.seek(0)
            except:
                pass
    
    # Read spectra
    datos_org, file_name = read_spectra(temp_files)

st.session_state.datos_org = datos_org
st.session_state.file_name = file_name
st.success(f"Successfully loaded data: {file_name}")

# ==================== DATA PRE-PROCESSING (SIDEBAR) ====================

st.sidebar.header("2. Pre-processing")
st.sidebar.subheader("Dataset Slicing")

with st.sidebar.expander("Slice Dataset Options", expanded=False):
    do_slice = st.toggle("Perform dataset slicing", value=False)
    
    # Display data info if available
    if st.session_state.datos_org is not None:
        min_t = float(st.session_state.datos_org.index.min())
        max_t = float(st.session_state.datos_org.index.max())
        min_w = float(st.session_state.datos_org.columns.min())
        max_w = float(st.session_state.datos_org.columns.max())
        st.info(f"Data Range:\nTime: {min_t:.2e} to {max_t:.2e} s\nWave: {min_w:.1f} to {max_w:.1f} nm")
    else:
        min_t = max_t = min_w = max_w = None

    st.caption("Time Range (s)")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        t_start = st.number_input("Start", value=None, format="%.6f", key="t_start_input",
                                   min_value=min_t, max_value=max_t)
    with col_t2:
        t_end = st.number_input("End", value=None, format="%.6f", key="t_end_input",
                                 min_value=min_t, max_value=max_t)

    st.caption("Wavelength Range (nm)")
    col_w1, col_w2 = st.columns(2)
    with col_w1:
        wave_start = st.number_input("Start", value=None, format="%.2f", key="w_start_input",
                                      min_value=min_w, max_value=max_w)
    with col_w2:
        wave_end = st.number_input("End", value=None, format="%.2f", key="w_end_input",
                                    min_value=min_w, max_value=max_w)

if do_slice:
    st.session_state.datos = slice_dataset(
        st.session_state.datos_org.copy(),
        t_start, t_end, wave_start, wave_end
    )
    st.sidebar.success(f"Sliced data shape: {st.session_state.datos.shape}")
else:
    st.session_state.datos = st.session_state.datos_org.copy()

datos = st.session_state.datos

# ==================== SPECTRA PLOTS ====================

st.header("Spectra Visualization")

tab1, tab2 = st.tabs(["Wavelength Plot", "Time Plot"])

with tab1:
    df_transposed = datos.T
    fig_wave = create_plotly_2d_plot(
        df_transposed,
        title=f"Absorbance vs Wavelength // {file_name}",
        x_axis="Wavelength (nm)",
        y_axis="Absorbance",
        legend_title="Time (s)"
    )
    st.plotly_chart(fig_wave, use_container_width=True)

with tab2:
    fig_time = create_plotly_2d_plot(
        datos,
        title=f"Absorbance vs Time // {file_name}",
        x_axis="Time (s)",
        y_axis="Absorbance",
        legend_title="Wavelength (nm)"
    )
    st.plotly_chart(fig_time, use_container_width=True)

# ==================== SVD ANALYSIS ====================

st.header("Singular Value Decomposition (SVD)")

st.sidebar.header("3. SVD Analysis")

scree_plot_th = st.sidebar.slider(
    "Scree Plot Threshold",
    min_value=0.5,
    max_value=0.99,
    value=0.9,
    step=0.01,
    help="R² threshold for the scree plot linear fit method"
)

entropy_threshold = st.sidebar.slider(
    "Entropy Threshold",
    min_value=0.5,
    max_value=0.99,
    value=0.9,
    step=0.01,
    help="Cumulative entropy threshold for entropy-based selection"
)

# Perform SVD (Cached)
datos_array = datos.to_numpy()
Times = datos.index
Wavelengths = datos.columns

U, Sigma, Vt = cached_svd(datos_array)

# Determine SSVs using different methods
fig_scree, n_significant_scree_plot = create_scree_plot_plotly(Sigma, scree_plot_th)
n_significant_entropy = entropy_selection(Sigma, entropy_threshold)
fig_broken, n_significant_broken_stick = create_broken_stick_plot_plotly(Sigma)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Scree Plot Method", n_significant_scree_plot)
with col2:
    st.metric("Entropy Method", n_significant_entropy)
with col3:
    st.metric("Broken Stick Method", n_significant_broken_stick)

tab1, tab2 = st.tabs(["Scree Plot", "Broken Stick Plot"])

with tab1:
    st.plotly_chart(fig_scree, use_container_width=True)

with tab2:
    st.plotly_chart(fig_broken, use_container_width=True)

with st.expander("View SVD Matrices", expanded=False):
    st.subheader("Singular Values (Σ)")
    st.dataframe(pd.DataFrame(Sigma, columns=["Singular Values"]))

# ==================== MATRIX APPROXIMATION ====================

st.header("Dimensionality Reduction")

st.sidebar.header("4. Matrix Approximation")

do_approximation = st.sidebar.toggle("Perform Matrix Approximation", value=True)

SSVs = st.sidebar.number_input(
    "Number of SSVs",
    min_value=1,
    max_value=min(20, len(Sigma)) if len(Sigma) > 0 else 1,
    value=min(3, len(Sigma)) if len(Sigma) > 0 else 1,
    step=1,
    help="Number of significant singular values to use for approximation"
)

if do_approximation:
    datos_approx = matrix_approximation(datos_array, SSVs)
    datos_approx_df = pd.DataFrame(datos_approx, index=Times, columns=Wavelengths)
    st.session_state.datos_approx_df = datos_approx_df
    st.success(f"Matrix approximation performed using {SSVs} singular values.")
else:
    st.session_state.datos_approx_df = datos.copy()
    st.info("Matrix approximation was not performed. Using original data.")

# Approximated spectra plots
with st.expander("View Approximated Spectra", expanded=False):
    datos_approx_df = st.session_state.datos_approx_df

    tab1, tab2 = st.tabs(["Wavelength Plot (Approx)", "Time Plot (Approx)"])

    with tab1:
        df_approx_transposed = datos_approx_df.T
        fig_wave_approx = create_plotly_2d_plot(
            df_approx_transposed,
            title=f"Approximated Absorbance vs Wavelength // {file_name}",
            x_axis="Wavelength (nm)",
            y_axis="Absorbance",
            legend_title="Time (s)"
        )
        st.plotly_chart(fig_wave_approx, use_container_width=True)

    with tab2:
        fig_time_approx = create_plotly_2d_plot(
            datos_approx_df,
            title=f"Approximated Absorbance vs Time // {file_name}",
            x_axis="Time (s)",
            y_axis="Absorbance",
            legend_title="Wavelength (nm)"
        )
        st.plotly_chart(fig_time_approx, use_container_width=True)

# ==================== REACTION MODEL PARAMETERS ====================

st.header("Reaction Model Parameters")

st.sidebar.header("5. Reaction Model")

n_species = st.sidebar.slider(
    "Number of Species",
    min_value=2,
    max_value=4,
    value=3,
    step=1
)

pathlength = st.sidebar.number_input(
    "Pathlength (cm)",
    min_value=0.01,
    max_value=10.0,
    value=1.0,
    step=0.1
)

Lower_bound = st.sidebar.checkbox("Apply Lower Bound to Spectra", value=False)
if Lower_bound:
    min_value = st.sidebar.number_input("Minimum Value", value=0.0)
else:
    min_value = 0

# Initial concentrations
st.subheader("Initial Concentrations (μM)")

# Prepare data for editor
conc_init_data = []
all_species_keys = ["A0", "B0", "C0", "D0"]
for i, sp in enumerate(all_species_keys):
    if i < n_species:
        conc_init_data.append({"Species": sp, "Concentration": 0.0})

df_conc_init = pd.DataFrame(conc_init_data)

edited_conc = st.data_editor(
    df_conc_init,
    column_config={
        "Species": st.column_config.TextColumn(disabled=True),
        "Concentration": st.column_config.NumberColumn("Conc (μM)", min_value=0.0, format="%.4f")
    },
    hide_index=True,
    use_container_width=True,
    num_rows="fixed",
    key="conc_editor_widget"
)

# Extract values
A0 = B0 = C0 = D0 = 0.0
for idx, row in edited_conc.iterrows():
    sp = row['Species']
    val = row['Concentration']
    if sp == 'A0': A0 = val
    elif sp == 'B0': B0 = val
    elif sp == 'C0': C0 = val
    elif sp == 'D0': D0 = val

# Rate constants
st.subheader("Rate Constants (1/s)")
st.caption("Edit values directly or check 'Fixed' to hold parameter constant.")

# Define the structure for the data editor
if 'rate_constants_df' not in st.session_state:
    st.session_state.rate_constants_df = pd.DataFrame(
        [
            {"Parameter": "k1", "Value": 0.0, "Fixed": True},
            {"Parameter": "k_1", "Value": 0.0, "Fixed": True},
            {"Parameter": "k2", "Value": 0.0, "Fixed": True},
            {"Parameter": "k_2", "Value": 0.0, "Fixed": True},
            {"Parameter": "k3", "Value": 0.0, "Fixed": True},
            {"Parameter": "k_3", "Value": 0.0, "Fixed": True},
        ]
    )

# Clean and modern data editor
edited_df = st.data_editor(
    st.session_state.rate_constants_df,
    column_config={
        "Parameter": st.column_config.TextColumn(
            "Rate Constant", 
            disabled=True
        ),
        "Value": st.column_config.NumberColumn(
            "Value (1/s)", 
            format="%.6f", 
            min_value=0.0,
            step=0.1
        ),
        "Fixed": st.column_config.CheckboxColumn(
            "Fixed?", 
            help="Check to fix this parameter during fitting",
            default=True
        ),
    },
    hide_index=True,
    use_container_width=True,
    key="rate_constants_editor_widget"
)

# Update session state with edited values
st.session_state.rate_constants_df = edited_df

# Classify into fixed and variable
fixed_ks = {}
variable_ks = {}

for index, row in edited_df.iterrows():
    param = row["Parameter"]
    val = row["Value"]
    is_fixed = row["Fixed"]
    
    if is_fixed:
        fixed_ks[param] = val
    else:
        variable_ks[param] = val

# Initial concentrations dictionary
concentration_data = {
    'A0': A0,
    'B0': B0,
    'C0': C0,
    'D0': D0,
}
species_list = ['A0', 'B0', 'C0', 'D0'][:n_species]
initial_conc = {key: concentration_data[key] for key in species_list}

initial_ks = {**fixed_ks, **variable_ks}

# Display summary
with st.expander("Parameter Summary", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**Fixed Rate Constants:**")
        for key, value in fixed_ks.items():
            st.write(f"  {key} = {value}")
    with col2:
        st.write("**Variable Rate Constants:**")
        for key, value in variable_ks.items():
            st.write(f"  {key} = {value}")
    with col3:
        st.write("**Initial Concentrations:**")
        for key, value in initial_conc.items():
            st.write(f"  {key} = {value}")

# ==================== FITTING ====================

st.header("Fitting")

Method = st.selectbox(
    "Method for Estimating Spectroscopic Species",
    options=["Pseudo-inverse", "Explicit", "Implicit"],
    index=0,
    help="""
    **Pseudo-inverse**: Best fitting, requires reasonable first estimation of rate constants.
    **Explicit**: Use to obtain initial idea of rate constants magnitude.
    **Implicit**: Alternative implicit approach.
    """
)

run_fitting = st.button("Run Fitting", type="primary")

if run_fitting:
    if not variable_ks:
        st.warning("No variable rate constants selected for optimization. Please uncheck 'Fixed' for at least one rate constant.")
    else:
        datos_approx_df = st.session_state.datos_approx_df

        initial_params = {**initial_ks}
        initial_params_var = {**variable_ks}
        nombrParVar = list(initial_params_var.keys())

        # For Explicit method, ensure initial rate constants are not zero
        # to prevent singular matrices during concentration profile calculation
        if Method == "Explicit" or Method == "Implicit":
            for key in nombrParVar:
                if initial_params[key] == 0:
                    initial_params[key] = 0.1
                    # st.info(f"{Method} method: Auto-initializing {key} to 0.1")

        fKwargs = dict(
            t=datos_approx_df.index.values,
            f_deriv=deriv_conc,
            Conc_0=initial_conc,
            abs=datos_approx_df,
            pathlength=pathlength,
            original_data=datos,
            method=Method,
            Lower_bound=Lower_bound,
            min_value=min_value,
            fitting=True,
        )

        with st.spinner("Running optimization... This may take a while."):
            try:
                sol = procesa(
                    argLeastSquares=argLeastSquares,
                    dictParEstim=initial_params,
                    nombrParVar=nombrParVar,
                    f=Model_spectra,
                    fKwargs=fKwargs,
                    Y=datos_approx_df.values.flatten(),
                )
                st.session_state.sol = sol

                # Generate Model results
                ad_parameters = sol['parAjustados']
                Model = Model_spectra(
                    ad_parameters,
                    deriv_conc,
                    initial_conc,
                    datos_approx_df.index,
                    datos_approx_df,
                    pathlength,
                    datos,
                    Method,
                    Lower_bound,
                    min_value,
                    fitting=False,
                )
                st.session_state.Model = Model

                st.success("Fitting completed successfully!")

            except Exception as e:
                st.error(f"Fitting failed: {str(e)}")
                st.exception(e)

# ==================== RESULTS ====================

if st.session_state.sol is not None and st.session_state.Model is not None:
    st.header("Results")

    sol = st.session_state.sol
    Model = st.session_state.Model

    # Display fitted parameters
    st.subheader("Fitted Parameters")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.caption("Adjusted Rate Constants")
        res_data = []
        for key, value in sol['parAjustados'].items():
            std_key = f"{key}_std"
            std_val = sol['sdPar'].get(std_key, None)
            # Ensure std_val is None if it's an empty string or 0 if that's what was intended
            if std_val == "": 
                std_val = None
                
            row = {"Parameter": key, "Value": value, "Std. Dev.": std_val}
            res_data.append(row)
        
        df_res = pd.DataFrame(res_data)
        st.dataframe(
            df_res,
            column_config={
                "Parameter": st.column_config.TextColumn("Parameter"),
                "Value": st.column_config.NumberColumn("Value (1/s)", format="%.6e"),
                "Std. Dev.": st.column_config.NumberColumn("Std. Dev.", format="%.6e"),
            },
            hide_index=True,
            use_container_width=True
        )

    with col2:
        st.metric("R²", f"{sol['R2']:.6f}")

    # Plots
    st.subheader("Model Plots")

    df_e = Model['D_model']
    df_conc = Model['C_matrix']
    df_spectra = Model['S_matrix'].T

    tab1, tab2, tab3, tab4 = st.tabs([
        "Modelled Spectra",
        "Concentration Profile",
        "Species Spectra",
        "Residuals"
    ])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig = create_plotly_2d_plot(
                df_e.T,
                title=f"Modelled Absorbance vs Wavelength",
                x_axis="Wavelength (nm)",
                y_axis="Absorbance",
                legend_title="Time (s)"
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = create_plotly_2d_plot(
                df_e,
                title=f"Modelled Absorbance vs Time",
                x_axis="Time (s)",
                y_axis="Absorbance",
                legend_title="Wavelength (nm)"
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = create_plotly_2d_plot(
            df_conc,
            title=f"Concentration over Time",
            x_axis="Time (s)",
            y_axis="Concentration (μM)",
            legend_title="Species"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = create_plotly_2d_plot(
            df_spectra,
            title=f"Spectroscopic Species Spectra",
            x_axis="Wavelength (nm)",
            y_axis="Extinction Coefficient (1/(μM·cm))",
            legend_title="Species"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        residual_type = st.radio(
            "Residual Type",
            options=["Original - Modelled", "Denoised - Modelled"],
            horizontal=True
        )

        if residual_type == "Original - Modelled":
            df_res = Model['residuals']
            r_title = "(Original - Modelled)"
        else:
            df_res = Model['residuals_denoised']
            r_title = "(Denoised - Modelled)"

        col1, col2 = st.columns(2)
        with col1:
            fig = create_plotly_2d_plot(
                df_res.T,
                title=f"Residuals vs Wavelength {r_title}",
                x_axis="Wavelength (nm)",
                y_axis="Absorbance",
                legend_title="Time (s)"
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = create_plotly_2d_plot(
                df_res,
                title=f"Residuals vs Time {r_title}",
                x_axis="Time (s)",
                y_axis="Absorbance",
                legend_title="Wavelength (nm)"
            )
            st.plotly_chart(fig, use_container_width=True)

    # Comparison plots
    st.subheader("Experimental vs Modelled Comparison")

    comparison_data = st.radio(
        "Experimental Data to Compare",
        options=["Original", "Denoised"],
        horizontal=True
    )

    if comparison_data == "Original":
        df1 = datos
    else:
        df1 = st.session_state.datos_approx_df

    df2 = Model['D_model']

    col1, col2 = st.columns(2)
    with col1:
        fig = create_plotly_comparison_plot(
            df1.T, df2.T,
            title="Absorbance vs Wavelength",
            x_axis="Wavelength (nm)",
            y_axis="Absorbance",
            legend_title="Time (s)",
            df1_label=comparison_data,
            df2_label="Model"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = create_plotly_comparison_plot(
            df1, df2,
            title="Absorbance vs Time",
            x_axis="Time (s)",
            y_axis="Absorbance",
            legend_title="Wavelength (nm)",
            df1_label=comparison_data,
            df2_label="Model"
        )
        st.plotly_chart(fig, use_container_width=True)

    # QQ Plots
    st.subheader("QQ Plots of Residuals")

    qq_residual_type = st.radio(
        "Residual Type for QQ Plot",
        options=["Original - Modelled", "Denoised - Modelled"],
        horizontal=True,
        key="qq_residual"
    )

    if qq_residual_type == "Original - Modelled":
        residuals = Model['residuals']
    else:
        residuals = Model['residuals_denoised']

    # Handle duplicate indices by averaging
    if residuals.index.duplicated().any():
        residuals_clean = residuals.groupby(residuals.index).mean()
    else:
        residuals_clean = residuals

    # Select a series for QQ plot
    selected_cols = st.multiselect(
        "Select Wavelength(s) for QQ Plot (Time Series)",
        options=residuals_clean.columns.tolist(),
        default=[residuals_clean.columns.tolist()[0]] if len(residuals_clean.columns) > 0 else []
    )

    col1, col2 = st.columns(2)
    with col1:
        if selected_cols:
            residuals_dict = {f"λ={col}": residuals_clean[col] for col in selected_cols}
            fig = create_plotly_qq_plot(
                residuals_dict,
                f"QQ Plot - Time Series ({len(selected_cols)} wavelength(s))"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select at least one wavelength to display the QQ plot.")

    with col2:
        selected_time = st.selectbox(
            "Select Time for QQ Plot (Wavelength Series)",
            options=residuals_clean.index.tolist()
        )
        # Get the row as a Series (transpose and select column)
        residuals_at_time = residuals_clean.loc[selected_time]
        if isinstance(residuals_at_time, pd.DataFrame):
            # If still a DataFrame (shouldn't happen after groupby), take mean
            residuals_at_time = residuals_at_time.mean()
        fig = create_plotly_qq_plot(
            residuals_at_time,
            f"QQ Plot - Wavelength Series (t = {selected_time})"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ==================== EXPORT ====================

    st.header("Export Results")

    export_name = st.text_input("Export Filename Prefix", value="spectra")

    # Create a BytesIO object to hold the zip file
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Original experimental data
            zipf.writestr('Original_experimental_data.csv', Model['D_orig'].to_csv())
            zipf.writestr('Original_experimental_data_TR.csv', Model['D_orig'].T.to_csv())

            # Denoised experimental data
            zipf.writestr('Denoised_experimental_data.csv', Model['D_approx'].to_csv())
            zipf.writestr('Denoised_experimental_data_TR.csv', Model['D_approx'].T.to_csv())

            # Modeled data
            zipf.writestr('Modelled_data.csv', Model['D_model'].to_csv())
            zipf.writestr('Modelled_data_TR.csv', Model['D_model'].T.to_csv())

            # Residuals from Original - Modeled
            zipf.writestr('Residuals_OrigMod.csv', Model['residuals'].to_csv())
            zipf.writestr('Residuals_OrigMod_TR.csv', Model['residuals'].T.to_csv())

            # Residuals from Denoised - Modeled
            zipf.writestr('Residuals_DenMod.csv', Model['residuals_denoised'].to_csv())
            zipf.writestr('Residuals_DenMod_TR.csv', Model['residuals_denoised'].T.to_csv())

            # Concentration profile
            zipf.writestr('Concentration_profile.csv', Model['C_matrix'].to_csv())

            # Spectroscopic species
            zipf.writestr('Spectroscopic_species.csv', Model['S_matrix'].to_csv())

            # Fitting results
            fitting_csv = io.StringIO()
            writer = csv.writer(fitting_csv)

            writer.writerow([''] * 7)
            writer.writerow(['n_species', n_species])
            writer.writerow(['pathlength (cm)', pathlength])
            writer.writerow([''] * 7)

            writer.writerow(['INITIAL CONCENTRATIONS:'])
            for key, value in initial_conc.items():
                writer.writerow([key, value])
            writer.writerow([''] * 7)

            writer.writerow(['INITIAL ks', '', 'ADJUSTED ks', '', 'STD ks'])
            for k_name in ['k1', 'k_1', 'k2', 'k_2', 'k3', 'k_3']:
                std_val = sol['sdPar'].get(f'{k_name}_std', '') if k_name in variable_ks else ''
                writer.writerow([
                    k_name, initial_ks[k_name], '',
                    k_name, sol['parAjustados'][k_name], '',
                    f'{k_name}_std', std_val
                ])

            writer.writerow([''] * 7)
            writer.writerow(['R2', sol['R2']])
            writer.writerow([''] * 7)

            writer.writerow(['Details'])
            writer.writerow(['cost', sol['detalles']['cost']])
            writer.writerow(['optimality', sol['detalles']['optimality']])
            writer.writerow(['nfev', sol['detalles']['nfev']])
            writer.writerow(['njev', sol['detalles']['njev']])
            writer.writerow(['status', sol['detalles']['status']])
            writer.writerow(['message', sol['detalles']['message']])
            writer.writerow(['success', sol['detalles']['success']])

            zipf.writestr('Fitting_result.csv', fitting_csv.getvalue())

    # Download button
    current_time = datetime.now().strftime("%d%m%Y%H%M%S")
    zip_filename = f"{export_name}_{current_time}.zip"

    st.download_button(
        label="Download Results ZIP",
        data=zip_buffer.getvalue(),
        file_name=zip_filename,
        mime="application/zip"
    )

show_footer()
