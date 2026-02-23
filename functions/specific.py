import bokeh.layouts
import bokeh.models
import bokeh.plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex
from scipy.linalg import svd
from sklearn.linear_model import LinearRegression

from .general import deriv_RK


def read_spectra(nombrFichs, tag="_t", skip_rows=0) -> tuple:
    """Reads multiple spectral data files and combines them into a single DataFrame."""
    df_list = []
    #Read each file into a DataFrame
    for fich in nombrFichs:
        temp_df = pd.read_csv(fich, skiprows=[skip_rows], index_col=0)
        df_list.append(temp_df)  # Append each DataFrame to the list
    # Concatenate all DataFrames into one
    df = pd.concat(df_list)
    # Sort the resulting DataFrame by index
    df = df.sort_index()
    # Get a name for the plots to follow which data was uploaded
    main = next((fich for fich in nombrFichs if tag in fich), None)
    return df, main


def create_plot(df, Title, x_axis, y_axis, Legend, width=1200, height=700) -> bokeh.layouts.column:
    """Creates a Bokeh plot from the given DataFrame."""
    # Create a figure
    p = bokeh.plotting.figure(
        title=Title,
        x_axis_label=x_axis,
        y_axis_label=y_axis,
        width=width,
        height=height,
    )

    # Define font sizes for the title, axes, and labels
    p.title.text_font_size = '20pt'
    p.xaxis.axis_label_text_font_size = '16pt'
    p.yaxis.axis_label_text_font_size = '16pt'
    p.xaxis.major_label_text_font_size = '12pt'
    p.yaxis.major_label_text_font_size = '12pt'

    # Generate a custom color palette using matplotlib's colormap
    n_lines = len(df.columns)
    cmap = plt.get_cmap("rainbow")  # Use the 'rainbow' colormap for visible spectrum
    colors = [to_hex(cmap(i)) for i in np.linspace(0, 1, n_lines)]

    indices = pd.to_numeric(df.index)

    # Plot each column as a line
    for idx, col in enumerate(df.columns):
        p.line(
            indices,
            df[col],
            legend_label=str(col),
            line_width=2,
            color=colors[idx],
        )

    # Customize the legend
    p.legend.title = Legend
    p.legend.location = "top_right"
    p.legend.click_policy = "hide"  # Allows hiding lines by clicking their labels
    p.toolbar_location = "below"
    p.legend.visible = False  # Initially hide the legend
    p.legend.label_text_font_size = '12pt'
    p.legend.title_text_font_size = '14pt'

    # Create a button to toggle the legend visibility
    button = bokeh.models.Button(label="Toggle Legend", button_type="success")

    # Custom JavaScript to toggle legend visibility
    button.js_on_click(
        bokeh.models.CustomJS(args=dict(legend=p.legend[0]),
                              code="\nlegend.visible = !legend.visible;\n"))

    # Return the plot object and button as a column layout
    return bokeh.layouts.column(p, button)


def slice_dataset(
    df,
    t_start=None,
    t_end=None,
    wave_start=None,
    wave_end=None,
) -> pd.DataFrame:
    """
    Slices a dataset row-wise and column-wise.

    Parameters:
    - df (pd.DataFrame): The input dataset.
    - t_start (float, optional): The starting value for row slicing.
    - t_end (float, optional): The ending value for row slicing.
    - wave_start (float, optional): The starting value for column slicing.
    - wave_end (float, optional): The ending value for column slicing.

    Returns:
    - pd.DataFrame: The sliced dataset.
    """
    # Ensure t_start and t_end are provided
    if t_start is not None and t_end is not None:
        # Find the index of the target time in the df.index
        row_start = (np.abs(df.index - t_start)).argmin()
        row_end = (np.abs(df.index - t_end)).argmin()
    else:
        row_start, row_end = None, None

    # Ensure wave_start and wave_end are provided
    if wave_start is not None and wave_end is not None:
        # Convert df.columns to float
        columns = df.columns.astype(float)  # Converts Index to array of floats
        col_start = (np.abs(columns - wave_start)).argmin()
        col_end = (np.abs(columns - wave_end)).argmin()
    else:
        col_start, col_end = None, None

    # Slice the rows
    if row_start is not None or row_end is not None:
        df = df.iloc[row_start:row_end]

    # Slice the columns
    if col_start is not None or col_end is not None:
        df = df.iloc[:, col_start:col_end]

    return df


# Scree Plot Method, with an elbow selection criterion based on the regression coefficient
def scree_plot_with_fit(singular_values, threshold, width=800, height=600) -> dict:
    """
    Plots scree plot of singular values and determine significant values using a linear fit.

    Parameters:
        singular_values (array-like): Array of singular values.
        threshold (float): Regression coefficient threshold (between 0 and 1) for linear fit.

    Returns:
        dict: Number of significant singular values and a Bokeh plot.
    """

    n_values = len(singular_values)
    SSVs = 0  # Number of significant singular values to keep

    # Initialize Bokeh figure
    p = bokeh.plotting.figure(
        title=" Scree Plot with Linear Fit",
        x_axis_label="Singular Value Index",
        y_axis_label="Singular Values",
        width=width,
        height=height,
    )
    p.title.text_font_size = '20pt'
    p.xaxis.axis_label_text_font_size = '16pt'
    p.yaxis.axis_label_text_font_size = '16pt'
    p.xaxis.major_label_text_font_size = '12pt'
    p.yaxis.major_label_text_font_size = '12pt'

    # Plot singular values
    indices = np.arange(1, n_values + 1)
    p.scatter(
        indices,
        singular_values,
        size=8,
        color='blue',
        legend_label="Singular Values",
    )

    # Placeholder variables for the inear fit line data
    X_final, y_final_pred = None, None

    # Iterate through singular values, trying linear fits
    for i in range(2, n_values + 1):  # Start with at least two points for linear regression
        X = np.arange(1, i + 1).reshape(-1, 1)
        y = singular_values[:i]

        # Perform linear regression
        model = LinearRegression().fit(X, y)
        r_squared = model.score(X, y)  # Get the R^2 (regression coefficient)

        # If the fit quality falls below the threshold, stop
        if r_squared < threshold:
            SSVs = i - 1
            break
        else:
            SSVs = i
        # Update the final data for the significant linear fit
        X_final = np.arange(1, SSVs + 1).reshape(-1, 1)
        y_final_pred = model.predict(X_final)

    # Plot the linear fit up to the last significant singular value
    p.line(
        X_final.flatten(),
        y_final_pred,
        line_width=2,
        color="red",
        line_dash="dashed",
        legend_label="Linear Fit",
    )

    # Add a vertical line to mark the cutoff for significant values
    cutoff_line = bokeh.models.Span(
        location=SSVs,
        dimension='height',
        line_color="green",
        line_dash="dashed",
    )
    p.add_layout(cutoff_line)

    # Add a label indicating the cutoff
    cutoff_label = bokeh.models.Label(
        x=SSVs,
        y=singular_values[SSVs - 1],
        text=f'Significant Count = {SSVs}',
        text_color='green',
        y_offset=10,
    )
    p.add_layout(cutoff_label)

    # Customize Legend and toolbar
    p.legend.title = "Legend"
    p.legend.location = "top_right"
    p.legend.click_policy = "hide"
    p.toolbar_location = "below"

    # Add a toogle button to control the legend visibility
    button = bokeh.models.Button(label="Toggle Legend", button_type="success")
    button.js_on_click(
        bokeh.models.CustomJS(args=dict(legend=p.legend[0]),
                              code="\nlegend.visible = !legend.visible;\n"))

    #Show the plot with the toggle button
    plot = bokeh.layouts.column(p, button)
    sol = {'SSVs': SSVs, "plot": plot}
    return sol


def entropy_selection(singular_values, entropy_threshold) -> int:
    """
    Entropy based method to determine the number of significant singular values.

    Parameters:
        singular_values (array-like): Array of singular values.
        entropy_threshold (float): Threshold for cumulative entropy (between 0 and 1).
    Returns:
        int: Number of significant singular values based on entropy threshold.
    """
    total_energy = np.sum(singular_values**2)

    # Calculate normalized singular values (f_j)
    f_j = singular_values**2 / total_energy

    # Calculate entropy
    entropy_val = np.sum(f_j * np.log(f_j)) / np.log(len(singular_values))
    #print(f"\t Entropy of singular values: {entropy_val:.4f}")

    # Calculate cumulative entropy for each k
    cumulative_entropy = np.zeros(len(singular_values))
    for k in range(len(singular_values)):
        ff = f_j[:k + 1]
        cumulative_entropy[k] = np.sum(ff * np.log(ff) / np.log(len(singular_values)))
    percentage = cumulative_entropy / entropy_val

    # Find the smallest index k such that cumulative entropy meets the threshold
    significant_indices = np.where(percentage >= entropy_threshold)[0]

    if len(significant_indices) == 0:
        # No significant indices found
        return 0
    else:
        # Number of significant components
        return significant_indices[0] + 1


def broken_stick_method(singular_values, width=800, height=600) -> dict:
    """
    Broken Stick Method to determine the number of significant singular values.

    Parameters:
        singular_values (array-like): Array of singular values.

    Returns:
        dict: Number of significant singular values and a Bokeh plot.
    """
    k = len(singular_values)

    # Calculate the broken stick values
    broken_stick = np.zeros(k)
    for i in range(1, k + 1):
        broken_stick[i - 1] = (1 / k) * np.sum([1 / j for j in range(i, k + 1)])

    # Normalize the squared singular values for comparison with the broken stick values
    singular_values_squared_n = singular_values**2 / np.sum(singular_values**2)

    # Initialize Bokeh figure
    p = bokeh.plotting.figure(
        title="Broken Stick Model vs Singular Values",
        x_axis_label="Index",
        y_axis_label="Proportion of Variance",
        width=width,
        height=height,
    )
    p.title.text_font_size = '20pt'
    p.xaxis.axis_label_text_font_size = '16pt'
    p.yaxis.axis_label_text_font_size = '16pt'
    p.xaxis.major_label_text_font_size = '12pt'
    p.yaxis.major_label_text_font_size = '12pt'

    # Plot the normalized singular values
    indices = np.arange(1, k + 1)
    p.line(
        indices,
        singular_values_squared_n,
        line_width=2,
        color="blue",
        legend_label="Singular Values",
    )
    p.scatter(indices, singular_values_squared_n, size=8, color="blue")

    # Plot the normalized broken stick values
    p.line(
        indices,
        broken_stick,
        line_width=2,
        line_dash="dashed",
        color="red",
        legend_label="Broken Stick",
    )
    p.scatter(indices, broken_stick, size=8, color="red")

    # Customize legend and toolbar
    p.legend.title = "Legend"
    p.legend.location = "top_right"
    p.legend.click_policy = "hide"
    p.toolbar_location = "below"

    # Determine the number of significant singular values using the broken stick rule
    SSVs = 0
    for i in range(k):
        if singular_values_squared_n[i] > broken_stick[i]:
            SSVs += 1
        else:
            break

    # Add a toggle button to control the legend visibility
    button = bokeh.models.Button(label="Toggle Legend", button_type="success")
    button.js_on_click(
        bokeh.models.CustomJS(args=dict(legend=p.legend),
                              code="\nlegend.visible = !legend.visible;\n"))

    #Show the plot with the toggle button
    plot = bokeh.layouts.column(p, button)
    sol = {'SSVs': SSVs, "plot": plot}
    return sol


def matrix_approximation(A, n) -> np.ndarray:
    """
    Approximates matrix A using the top n singular values

    Parameters:
    - A: The original matrix to approximate.
    - n: Number of significant singular values to use for approximation.

    Returns:
    - A_approx: The approximated matrix.
    """
    # Perform SVD using scipy.linalg.svd
    U, Sigma, VT = svd(A, full_matrices=False)

    # Truncate the matrices to keep only the top 'n' singular values
    U_n = U[:, :n]  # Keep the first 'n' columns of U
    Sigma_n = np.diag(
        Sigma[:n])  # Keep the first 'n' singular values (diagonal matrix)
    VT_n = VT[:n, :]  # Keep the first 'n' rows of V^T

    # Compute the approximated matrix
    A_approx = np.dot(U_n, np.dot(Sigma_n, VT_n))  # A_approx = U_n * Sigma_n * VT_n

    return A_approx


def kinetic_model_matrix(n_species, k_vals) -> np.ndarray:
    """
    Creates the ODE matrix to represent a system of species with the specified rate constants.

    Parameters:
        n_species (int): Number of species in the system.
        params (dict): Dictionary of rate constants, e.g., {'k1': value, 'k_1': value, 'k2': value, ...}.

    Returns:
        np.ndarray: Matrix that aligns with the ODEs specified for each species.
    """
    # Initialize an n_species x n_species matrix with zeros
    ode_matrix = np.zeros((n_species, n_species))

    # Populate the ODE matrix according to the specified rules
    for i in range(n_species):
        # Rate constant for reaction from species i to species i+1, if within bounds
        if i + 1 < n_species:
            ode_matrix[i, i] -= k_vals.get(f'k{i+1}', 0)  # Outflow from species i to i+1
            ode_matrix[i + 1, i] += k_vals.get(f'k{i+1}', 0)  # Inflow to species i+1 from i
        # Rate constant for reaction from species i+1 back to species i, if within bounds
        if i - 1 >= 0:
            ode_matrix[i, i] -= k_vals.get(f'k_{i}', 0)  # Outflow from species i to i-1
            ode_matrix[i - 1, i] += k_vals.get(f'k_{i}', 0)  # Inflow to species i-1 from i

    return ode_matrix


def deriv_conc(conc, t, ks_matrix) -> np.ndarray:
    """
    Calculate the concentration derivative of every species .
    The system of ODE's characterizing the reaction model is passed as a matrix with the rate constants
    as coefficients. (REVISE)

    Parameters
        conc: Array con las concentraciones de las especies.
        t : times points at which to calculate de derivative of the concentration with respect time.
        params: a dictionary that contains the rate constants.

    Returns:
        np.ndarray: vector that contains the derivative of the concentrations of each species
                    at the specified time point t. the rate of change of conecntration for each
                    species in the system at the given time t.

    """
    return np.dot(ks_matrix, conc)


def solv_conc_profile(k_vals, f_deriv, Conc_0, t) -> pd.DataFrame:
    """
    Solves the concentration profile of the reaction over time using
    a 4th-order Runge-Kutta (RK4) method, allowing for variable time steps.

    Parameters:
    - f: Function that computes the derivative (reaction model)
    - y0: Initial concentrations of the species
    - t: Array or list of time points (can have non-uniform intervals)
    - k_vals: Dictionary of reaction kinetic constants needed for the reaction model

    Returns:
    - df: DataFrame containing the cncentration profile for each species over time
    """

    # Extract Conc_0 from 'initial_conc' in params
    initial_conc = np.array(list(Conc_0.values()))

    n_steps = len(t)
    n_species = len(initial_conc)

    # Initialize the solution array to store each species' concentration at each time step
    solution = np.zeros((n_steps, n_species))
    solution[0] = initial_conc  # Initial conditions

    # We create the ODE system as matrix with the rate constants dispossed as its coefficients
    MCoef = kinetic_model_matrix(n_species, k_vals)

    # Iterate through each time step using the function deriv_RK
    for i in range(1, n_steps):
        current_t = t[i - 1]
        next_t = t[i]
        current_y = solution[i - 1]

        # Here, calculate the time intercal (delta_t) dynamically
        delta_t = next_t - current_t

        # Use deriv_RK to calculate the next step, passing `f_deriv` as the first argument
        solution[i] = current_y + delta_t * deriv_RK(f_deriv, current_y, current_t, delta_t, MCoef)

    # Generate column names based on the number of species (A, B, C, ...)
    column_names = [f"{chr(65 + j)}" for j in range(n_species)]

    # Create the DataFrame without empty columns
    df = pd.DataFrame(solution, index=t, columns=column_names)  # shape (time, species)
    return df


def species_spectra(
    k_vals,
    f_deriv,
    Conc_0,
    t,
    abs,
    pathlength,
    method,
    Lower_bound,
    min_value,
) -> pd.DataFrame:
    """Calculate the species spectra using different methods based on concentration profiles."""

    initial_conc = np.array(list(Conc_0.values()))

    n_species = len(initial_conc)

    # Extract the reaction model (fDeriv)
    #model = k_vals.get('fDeriv')

    C_profile = solv_conc_profile(k_vals, f_deriv, Conc_0, t)

    match method:
    # Calculation of the spectra explicitly with assumptions
        case "Explicit":
            # Generalized code to find max value and corresponding index
            max_indices = {}
            for col in C_profile.columns:
                # Get the index of the max value
                max_index = C_profile[col].idxmax()
                max_indices[col] = max_index

            # Use the indices found to slice the DataFrames
            indices = list(max_indices.values())
            # Calculate the concentration for the first and last species
            c_prof_0 = C_profile.loc[indices[0]]
            if isinstance(c_prof_0, pd.DataFrame): c_prof_0 = c_prof_0.mean(axis=0)
            first_species_c = c_prof_0.iloc[0]

            c_prof_last = C_profile.loc[indices[-1]]
            if isinstance(c_prof_last, pd.DataFrame): c_prof_last = c_prof_last.mean(axis=0)
            last_species_c = c_prof_last.iloc[-1]
            
            # Prevent division by zero if concentration is too small
            epsilon = 1e-10
            if np.abs(first_species_c) < epsilon: first_species_c = epsilon
            if np.abs(last_species_c) < epsilon: last_species_c = epsilon

            # Calculate the spectra for the first and last species
            abs_0 = abs.loc[indices[0]]
            if isinstance(abs_0, pd.DataFrame): abs_0 = abs_0.mean(axis=0)
            first_species_s = (abs_0 / (pathlength * first_species_c)).to_frame().T

            abs_last = abs.loc[indices[-1]]
            if isinstance(abs_last, pd.DataFrame): abs_last = abs_last.mean(axis=0)
            last_species_s = (abs_last / (pathlength * last_species_c)).to_frame().T

            # Identify the reduced indices (excluding the first and last species)
            red_indices = indices[1:-1]

            if len(red_indices) > 0:  # Ensure red_indices is not empty
                # Extract the relevant concentration data for the reduced species
                reduced_conc = C_profile.loc[red_indices].iloc[:, 1:-1]
                
                # Reshape concentrations for broadcasting
                c_first_vals = C_profile.loc[red_indices].iloc[:, 0].values
                if c_first_vals.ndim == 1: c_first_vals = c_first_vals.reshape(-1, 1)
                
                c_last_vals = C_profile.loc[red_indices].iloc[:, -1].values
                if c_last_vals.ndim == 1: c_last_vals = c_last_vals.reshape(-1, 1)

                reduced_abs = abs.loc[red_indices] - (c_first_vals * first_species_s.values) - (c_last_vals * last_species_s.values)

                # Solve the system of equations C^-1*A = E
                s_red = pd.DataFrame(np.dot(np.linalg.pinv(reduced_conc), reduced_abs),
                                    index=red_indices,
                                    columns=abs.columns)
                sol_expl = pd.concat([first_species_s, s_red, last_species_s])
            else:
                sol_expl = pd.concat([first_species_s, last_species_s])

            # Assign alphabetical names to the indices (A, B, C, ...)
            alphabet_indices = [chr(65 + i) for i in range(len(indices))]  # 65 is ASCII for 'A'
            sol_expl.index = alphabet_indices
            result = sol_expl

        # Implicit approach of the explicit approach above
        case "Implicit":
            # Generalized code to find max value and corresponding index
            max_indices = {}
            for col in C_profile.columns:
                # max_value = C_profile[col].max()
                max_index = C_profile[col].idxmax(
                )  # Get the index of the max value
                max_indices[col] = max_index

            # Use the indices found to slice the DataFrames
            indices = list(max_indices.values())
            reduced_conc = C_profile.loc[indices]
            reduced_abs = abs.loc[indices]
            # Solve the system of equations C^-1*A = E
            sol_imp = np.dot(np.linalg.pinv(reduced_conc), reduced_abs)

            # Assign alphabetical names to the indices (A, B, C, ...)
            alphabet_indices = [chr(65 + i) for i in range(len(indices))]  # 65 is ASCII for 'A'
            sol_imp = pd.DataFrame(sol_imp,
                                index=alphabet_indices,
                                columns=abs.columns)
            sol_imp = sol_imp
            result = sol_imp

        # Use of the pseudoinverse of the concentration to estimate the spectroscopic species
        case "Pseudo-inverse":
            # (extinction coefficients)
            sol_ps = np.dot(np.linalg.pinv(C_profile), abs)
            alphabet_indices = [chr(65 + i) for i in range(len(C_profile.columns))]  # 65 is ASCII for 'A'
            sol_ps = pd.DataFrame(sol_ps,
                                index=alphabet_indices,
                                columns=abs.columns)
            sol_ps = sol_ps
            result = sol_ps

        case _:
            raise ValueError("No valid method selected")

    if Lower_bound:
        result = result.clip(lower=min_value)
    else:
        result = result

    #return sol, sol_imp, sol_ps
    return result


def Model_spectra(
    k_vals,
    f_deriv,
    Conc_0,
    t,
    abs,
    pathlength,
    original_data,
    method,
    Lower_bound,
    min_value,
    fitting=True,
) -> pd.DataFrame:
    """Calculates the predicted absorbance data based on the kinetic model and species spectra."""

    n_species = len(np.array(list(Conc_0.values())))

    #Solve for concentrations
    C_matrix = solv_conc_profile(k_vals, f_deriv, Conc_0, t)

    #Construct full extinction coefficient matrix
    S_matrix = species_spectra(
        k_vals,
        f_deriv,
        Conc_0,
        t,
        abs,
        pathlength,
        method,
        Lower_bound,
        min_value,
    )

    #Use Lambert-Beer Law to calculate predicted absorbance (n_Lambda x n_t)
    D_model = pathlength * np.dot(C_matrix, S_matrix)

    D_exp = abs
    D_org = original_data

    #Compute residuals (difference between experimental abd predicted absorbance)
    residuals_approximated = D_exp - D_model
    residuals = D_org - D_model
    D_model_df = pd.DataFrame(D_model,
                              index=D_exp.index,
                              columns=D_exp.columns)

    if fitting:
        sol = D_model_df.values.flatten()
    else:
        sol = {
            "params": k_vals,
            "Conc_0": Conc_0,
            "pathlength": pathlength,
            "n_species": n_species,
            "D_orig": original_data,
            "D_approx": abs,
            "D_model": D_model_df,
            "C_matrix": C_matrix,
            "S_matrix": S_matrix,
            "residuals": residuals,
            "residuals_approximated": residuals_approximated
        }
    return sol


def create_dynamic_plot(
    df1,
    df2,
    Title,
    x_axis,
    y_axis,
    Legend,
    df1_label,
    df2_label,
    width=1200,
    height=700,
):
    # Create a figure
    p = bokeh.plotting.figure(
        title=Title,
        x_axis_label=x_axis,
        y_axis_label=y_axis,
        width=width,
        height=height,
    )

    # Define font sizes for the title, axes, and labels
    p.title.text_font_size = '20pt'
    p.xaxis.axis_label_text_font_size = '16pt'
    p.yaxis.axis_label_text_font_size = '16pt'
    p.xaxis.major_label_text_font_size = '12pt'
    p.yaxis.major_label_text_font_size = '12pt'

    # Check if the inputted columns are valid
    col_names = df1.columns
    n_lines = len(col_names)  # Total number of series

    # Generate two distinct colors for each DataFrame
    df1_color = "#1f77b4"  # A blue color for df1
    df2_color = "#ff7f0e"  # An orange color for df2

    indices = pd.to_numeric(df1.index)

    # Create a list to store the line objects (to toggle their visibility)
    lines = []

    # Plot all the lines (one for each series in df1 and df2)
    for i, col_name in enumerate(col_names):
        # Ensure col_name is a string for the legend label
        col_name_str = str(col_name)

        # For df1, use the same color for all lines
        line1 = p.line(
            indices,
            df1[col_name],
            legend_label=col_name_str,
            line_width=8,
            color=df1_color,
            line_dash="solid",
            alpha=1,
            name=f"line1_{i}",
        )
        # For df2, use the same color for all lines
        line2 = p.line(
            indices,
            df2[col_name],
            legend_label=col_name_str,
            line_width=4,
            color=df2_color,
            line_dash="dashed",
            name=f"line2_{i}",
        )
        points2 = p.scatter(
            indices,
            df2[col_name],
            marker="circle",
            size=4,
            color=df2_color,
            alpha=0.8
        )
        lines.append((line1, line2, points2))
    # Customize the legend
    p.legend.title = Legend
    p.legend.location = "top_right"
    p.legend.click_policy = "hide"  # Allows hiding lines by clicking their labels
    p.toolbar_location = "below"
    p.legend.visible = False  # Initially hide the legend
    p.legend.label_text_font_size = '12pt'
    p.legend.title_text_font_size = '14pt'

    # Create a dropdown menu for selecting series using df1 column names
    series_select = bokeh.models.Select(
        title="Select Series",
        value=str(0),
        options=[(str(i), str(col_name)) for i, col_name in enumerate(col_names)],
    )

    # JavaScript callback for updating the visibility of the lines based on selection
    callback = bokeh.models.CustomJS(args=dict(lines=lines),
                        code="""
        var selected_index = parseInt(cb_obj.value);
        for (var i = 0; i < lines.length; i++) {
            lines[i][0].visible = (i == selected_index);  // Show the selected series (df1)
            lines[i][1].visible = (i == selected_index);  // Show the corresponding model (df2)
            lines[i][2].visible = (i == selected_index);  // Show the corresponding model (df2)
        }
    """)

    # Attach the callback to the dropdown menu
    series_select.js_on_change('value', callback)

    # Initially, set the visibility for the first series
    for i, (line1, line2, points2) in enumerate(lines):
        if i == 0:
            line1.visible = True
            line2.visible = True
            points2.visible = True
        else:
            line1.visible = False
            line2.visible = False
            points2.visible = False

    # Create a button to toggle the legend visibility
    button = bokeh.models.Button(label="Toggle Legend", button_type="success")

    # Custom JavaScript to toggle legend visibility
    button.js_on_click(
        bokeh.models.CustomJS(args=dict(legend=p.legend[0]),
                              code="\nlegend.visible = !legend.visible;\n"))

    # Return the plot, series select dropdown, and button in a layout
    return bokeh.layouts.column(p, series_select, button)


