import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math
from tqdm.auto import tqdm

# Pymoo imports
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.core.callback import Callback


# Progress Bar Callback
class GenerationProgressBar(Callback):
    def __init__(self, n_gen):
        super().__init__()
        self.pbar = tqdm(total=n_gen, desc="Optimizing Design", leave=False)

    def notify(self, algorithm):
        self.pbar.update(1)

    def close(self):
        self.pbar.close()


# Main Inverse Model Class
class NanoparticleInverseModel:
    def __init__(self, data_packet):
        # extract metadata and features from the loaded data dictionary
        self.forward_model = data_packet['models']

        # metadata extraction from the 'features' key
        self.numeric_features = data_packet['features']['numeric']
        self.categorical_features = data_packet['features']['categorical']
        self.specific_features = data_packet['features']['specific']
        self.target_names = data_packet['targets']

        # derive tumor target name from metadata
        self.tumor_target_name = next(t for t in self.target_names if t.startswith("Tumor"))

        # var_order defines the 10 optimization variable layout used in predict and plot methods.
        self.var_order = [
            "Size(TEM) (nm)", "Size(HD) (nm)", "Zeta Potential (mV)",
            "PDI", "Administration Dose (mg/kg)", "Time Point (h)",
            "Core Material", "Targeting Strategy", "Shape", "Particle Type"
        ]

        # categorical options for optimization mapping
        self.core_options = [
            "polymeric", "gold", "liposome", "silica", "hydrogel",
            "dendrimer", "graphene", "iron oxide", "drug_based",
            "albumin", "carbon_nanotube", "magnetic",
            "protein_based", "polymeric_composite", "manganese",
            "copper", "anticancer drug", "platinum",
            "lipid_based", "silica_based", "2d_material"
        ]
        self.strategy_options = ["active", "passive"]
        self.shape_options = ["spherical", "rod", "plate"]
        self.particle_type_options = ["inm", "hybrid", "onm"]

        # ordered list matching the categorical section of var_order (indices 6-9)
        self.categorical_options_ordered = [
            self.core_options,
            self.strategy_options,
            self.shape_options,
            self.particle_type_options
        ]

        print(f"Targets: {self.target_names}")

    def predict_optimal_design(self, tumor_cell, tumor_size, body_weight,
                               sensitive_organ_weights=None,
                               custom_constraints=None,
                               generations=60):
        # default weight of 1.0
        full_weights = {organ: 1.0 for organ in self.target_names}
        if sensitive_organ_weights:
            # merge provided weights with defaults
            full_weights.update(sensitive_organ_weights)

        # handle constraints
        # Variables:
        # [0]Size(TEM), [1]Size(HD), [2]Zeta Potential, [3]PDI, [4]Dose, [5]Time,
        # [6]Core(idx), [7]Strategy(idx), [8]Shape(idx), [9]Particle Type

        xl = np.array([2.0, 2.7, -65.12, 0.01, 0.0001, 0.5, 0, 0, 0, 0])
        xu = np.array([358.0, 1200.0, 71.3, 1.91, 1290.0, 72.0,
                       len(self.core_options) - 0.1,
                       len(self.strategy_options) - 0.1,
                       len(self.shape_options) - 0.1,
                       len(self.particle_type_options) - 0.1])

        # apply custom numeric constraints
        if custom_constraints:
            for feature, bounds in custom_constraints.items():
                if feature in self.var_order:
                    i = self.var_order.index(feature)
                    xl[i] = bounds[0]
                    xu[i] = bounds[1]

        # setup NSGA-3 based on the actual number of objectives found
        n_obj = len(self.target_names)
        self.ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=4)
        self.algorithm = NSGA3(ref_dirs=self.ref_dirs, pop_size=150)

        # context-specific fixed values
        fixed_context = {
            "Tumor Size (cm3)": tumor_size,
            "Body weight (g)": body_weight,
            "Tumor Cell": tumor_cell.lower()
        }

        # capture references needed inside the nested class
        outer_categorical_options = self.categorical_options_ordered
        outer_specific_features = self.specific_features
        outer_categorical_features = self.categorical_features
        outer_target_names = self.target_names
        outer_tumor_target_name = self.tumor_target_name

        class DynamicInverseProblem(ElementwiseProblem):
            def __init__(self, outer, xl_val, xu_val):
                n_variables = len(xl_val)
                super().__init__(n_var=n_variables, n_obj=len(outer.target_names), n_constr=7, xl=xl_val, xu=xu_val)
                self.outer = outer

            def _evaluate(self, x, out, *args, **kwargs):

                # map categorical indices back to strings using the ordered options list
                categorical_start_index = 6
                resolved_categorical_values = [
                    outer_categorical_options[offset][int(x[categorical_start_index + offset])]
                    for offset in range(len(outer_categorical_options))
                ]

                # design parameters from optimizer
                design_params = {
                    "Size(TEM) (nm)": x[0],
                    "Size(HD) (nm)": x[1],
                    "Zeta Potential (mV)": x[2],
                    "PDI": x[3],
                    "Administration Dose (mg/kg)": np.log1p(x[4]),
                    "Time Point (h)": np.log1p(x[5]),
                    "Core Material": resolved_categorical_values[0],
                    "Targeting Strategy": resolved_categorical_values[1],
                    "Shape": resolved_categorical_values[2],
                    "Particle Type": resolved_categorical_values[3]
                }

                # merge design parameters with the fixed scenario inputs
                full_row_data = {**design_params, **fixed_context}

                df_input = pd.DataFrame([full_row_data])

                # apply category type casting
                for col in self.outer.categorical_features:
                    if col in df_input.columns:
                        df_input[col] = df_input[col].astype(str).astype("category")

                # forward prediction
                tumor_type = fixed_context["Tumor Cell"]
                tumor_specific_models = self.outer.forward_model[tumor_type]
                # order columns as the boosters expect (excluding Tumor Cell)
                prediction_ready_df = df_input[outer_specific_features]

                dmatrix = xgb.DMatrix(prediction_ready_df, enable_categorical=True)

                preds = []
                for organ in self.outer.target_names:
                    booster = tumor_specific_models[organ]
                    log_val = booster.predict(dmatrix)[0]
                    preds.append(np.expm1(np.nan_to_num(log_val, nan=0.0)))

                # find the index of the tumor to maximize it (minimize its negative)
                tumor_idx = outer_target_names.index(outer_tumor_target_name)

                # objectives: minimize [-tumor, heart, liver, lung, spleen, kidney]
                # negate index 0 (Tumor) because NSGA3 minimizes by default
                objectives = [preds[i] if i != tumor_idx else -preds[i] for i in range(len(preds))]
                out["F"] = np.array(objectives)

                # Capture variables
                size_tem = x[0]
                size_hd = x[1]
                zeta = x[2]
                pdi = x[3]
                # strategy: 0 = active, 1 = passive
                strategy = int(np.round(x[7]))
                strategy = np.clip(strategy, 0, 1)

                # Identify the tumor prediction
                tumor_idx = outer_target_names.index(outer_tumor_target_name)
                predicted_tumor_conc = preds[tumor_idx]

                constraints = []

                # Physical Consistency (2 constraints)
                # HD must be >= TEM
                constraints.append(size_tem - size_hd)  # <= 0
                # Enforce PDI <= 0.2
                constraints.append(pdi - 0.2)

                # Biological Efficacy (1 constraint)
                # Reject if tumor concentration is less than 10%
                # If tumor is 5%, 10 - 5 = 5 (Positive = Violation)
                # If tumor is 15%, 10 - 15 = -5 (Negative = Success)
                constraints.append(10 - predicted_tumor_conc)

                # Strategy Dependent (4 constraints)
                if strategy == 1:  # passive
                    # Size constraints
                    constraints.append(20 - size_hd)  # size >= 20
                    constraints.append(size_hd - 150)  # size <= 150
                    # Zeta constraints
                    constraints.append(-10 - zeta)  # zeta >= -10
                    constraints.append(zeta - 10)  # zeta <= 10
                else:  # active
                    # Size constraints
                    constraints.append(20 - size_hd)
                    constraints.append(size_hd - 100)
                    # Zeta constraints
                    constraints.append(-20 - zeta)
                    constraints.append(zeta - 10)

                out["G"] = np.array(constraints)

        problem = DynamicInverseProblem(self, xl, xu)

        progress_callback = GenerationProgressBar(generations)

        try:
            res = minimize(
                problem,
                self.algorithm,
                termination=('n_gen', generations),
                seed=42,
                verbose=False,
                callback=progress_callback
            )
        finally:
            progress_callback.close()

        # rank results by selectivity score: maximize tumor while minimizing weighted toxicity
        tumor_idx = self.target_names.index(self.tumor_target_name)
        tumor_uptake = -res.F[:, tumor_idx]

        # filter objectives and weights to exclude tumor for toxicity calculation
        healthy_organ_indices = [i for i in range(len(self.target_names)) if i != tumor_idx]
        healthy_toxicity = res.F[:, healthy_organ_indices]

        # extract weights for healthy organs only
        weights_array = np.array([full_weights[self.target_names[i]] for i in healthy_organ_indices])

        # calculate weighted average toxicity: sum(value * weight) / sum(weights)
        weighted_toxicity = np.sum(healthy_toxicity * weights_array, axis=1) / np.sum(weights_array)

        # selectivity score: maximize (tumor / Average healthy organ uptake)
        selectivity_scores = tumor_uptake / np.maximum(weighted_toxicity, 1e-6)

        # get indices of the top 10 designs, sorted by score descending
        top_indices = np.argsort(selectivity_scores)[::-1][:10]

        all_top_designs = []
        for idx in top_indices:
            current_x = res.X[idx]
            # res.F contains [-Tumor, Healthy1, Healthy2...]
            current_f_raw = res.F[idx].copy()
            current_f_raw[tumor_idx] = -current_f_raw[tumor_idx]

            design_entry = {
                "design_params": {
                    "Size(TEM) (nm)": current_x[0],
                    "Size(HD) (nm)": current_x[1],
                    "Zeta Potential (mV)": current_x[2],
                    "PDI": current_x[3],
                    "Core Material": self.core_options[int(current_x[6])],
                    "Targeting Strategy": self.strategy_options[int(current_x[7])],
                    "Shape": self.shape_options[int(current_x[8])],
                    "Particle Type": self.particle_type_options[int(current_x[9])],
                    "Dose (mg/kg)": current_x[4],
                    "Time Point (h)": current_x[5]
                },
                "predicted_distribution": {
                    self.target_names[i]: float(current_f_raw[i]) for i in range(len(self.target_names))
                },
                "selectivity_score": float(selectivity_scores[idx])
            }
            all_top_designs.append(design_entry)

        return all_top_designs, res

    def plot_pareto_front(self, res, save_folder='/content/drive/MyDrive/Nanoparticle_Project_Saves/'):
        # visualize the 6D Pareto front as trade-offs against Tumor Uptake

        # res.F contains objectives: [-Tumor, Heart, Liver, Lung, Spleen, Kidney]
        # convert them back to positive raw values for the plot
        tumor_idx = self.target_names.index(self.tumor_target_name)
        tumor_uptake = -res.F[:, tumor_idx]

        healthy_organ_indices = [i for i in range(len(self.target_names)) if i != tumor_idx]
        organs_data = res.F[:, healthy_organ_indices]

        # exclude 'Tumor'
        organ_names = [self.target_names[i] for i in healthy_organ_indices]

        n_organs = organs_data.shape[1]
        fig, axes = plt.subplots(1, n_organs, figsize=(20, 4), sharey=True)

        for i in range(n_organs):
            ax = axes[i]
            ax.scatter(organs_data[:, i], tumor_uptake, alpha=0.5, c='teal', edgecolors='k')
            ax.set_title(f"Tumor vs {organ_names[i]}")
            ax.set_xlabel(f"{organ_names[i]} (%ID/g)")
            if i == 0:
                ax.set_ylabel("Tumor Uptake (%ID/g)")
            ax.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        # save to Colab
        filename = "inverse_design_pareto_front.png"
        plt.savefig(filename, dpi=300)

        # Save to Drive

        #plt.show()

    # Visualizes the impact of numeric and categorical index features
    # on the trade-off between Liver toxicity and Tumor uptake.
    def plot_design_analysis(self, res, tumor_cell_type, toxic_organ_name="Liver Concentration (%ID/g)",
                             save_folder='/content/drive/MyDrive/Nanoparticle_Project_Saves/'):

        # identify indices for the axes
        try:
            tumor_idx = self.target_names.index(self.tumor_target_name)
            toxic_idx = self.target_names.index(toxic_organ_name)
        except ValueError:
            available = ", ".join(self.target_names)
            print(f"Error: '{toxic_organ_name}' not found. Available targets: {available}")
            return

        tumor_vals = -res.F[:, tumor_idx]
        toxic_vals = res.F[:, toxic_idx]

        # identify all features being optimized
        n_features = res.X.shape[1]

        # calculate dynamic grid dimensions (aiming for 3 columns)
        ncols = 3
        nrows = math.ceil(n_features / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(20, 5 * nrows))
        axes = axes.flatten()

        clean_toxic_name = toxic_organ_name.split(" ")[0]

        # number of numeric variables before the categorical section starts
        categorical_start_index = 6

        for i in range(n_features):
            ax = axes[i]

            label = self.var_order[i]

            # Determine if feature is categorical (index 6, 7, 8, or 9)
            if i >= categorical_start_index:
                options = self.categorical_options_ordered[i - categorical_start_index]
                n_options = len(options)

                # Create discrete boundaries for categorical data
                # This ensures the color range maps exactly to the number of categories
                cmap = plt.get_cmap('plasma', n_options)
                norm = mcolors.BoundaryNorm(np.arange(n_options + 1) - 0.5, n_options)

                sc = ax.scatter(
                    toxic_vals,
                    tumor_vals,
                    c=res.X[:, i],
                    cmap=cmap,
                    norm=norm,
                    s=45,
                    edgecolors='k',
                    alpha=0.7
                )

                # set discrete ticks on colorbar
                cbar = plt.colorbar(sc, ax=ax, ticks=range(n_options))
                cbar.ax.set_yticklabels(options)
            else:
                # continuous mapping for numeric features
                sc = ax.scatter(
                    toxic_vals,
                    tumor_vals,
                    c=res.X[:, i],
                    cmap='plasma',
                    s=45,
                    edgecolors='k',
                    alpha=0.7
                )
                plt.colorbar(sc, ax=ax, label=label)

            ax.set_title(f"Impact of {label}", fontsize=14, fontweight='bold')
            ax.set_xlabel(f"Predicted {clean_toxic_name} Conc. (%ID/g)")
            ax.set_ylabel(f"{tumor_cell_type.capitalize()} Tumor (%ID/g)")
            ax.grid(True, linestyle='--', alpha=0.3)

        # remove any empty subplots if n_features isn't a multiple of 3
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.suptitle(
            f"Inverse Design Sensitivity: TUMOR vs. {clean_toxic_name.upper()}",
            fontsize=22, y=1.01, fontweight='bold'
        )
        plt.tight_layout()
        # save to Colab
        filename = "inverse_design_analysis_plot.png"
        plt.savefig(filename, dpi=300)


        #plt.show()

