import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
from main import NanoparticleInverseModel
import io
import matplotlib
matplotlib.use('Agg')

st.set_page_config(page_title="Nanoparticle Design Portal", layout="wide")




# model loading
@st.cache_resource
def load_framework():
    model_path = 'xgboost_packaged_tumor_models.joblib'
    if not os.path.exists(model_path):
        st.error(f"Error: {model_path} not found.")
        st.stop()
    data_packet = joblib.load(model_path)
    return NanoparticleInverseModel(data_packet)


inverse_model = load_framework()

# session state initialization
if 'results_ready' not in st.session_state:
    st.session_state.results_ready = False
if 'top_designs' not in st.session_state:
    st.session_state.top_designs = None
if 'opt_res_object' not in st.session_state:
    st.session_state.opt_res_object = None
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'cached_organ_plots' not in st.session_state:
    st.session_state.cached_organ_plots = {}

def clear_results():
    st.session_state.results_ready = False
    st.session_state.is_running = False
    st.session_state.top_designs = None
    st.session_state.opt_res_object = None
    st.session_state.cached_organ_plots = {}

# sidebar configuration
st.sidebar.header("Target Scenario")

# filter out "other" from available tumor types
available_tumors = [t for t in inverse_model.forward_model.keys() if t.lower() != "other"]
tumor_type = st.sidebar.selectbox("Select Tumor Type", available_tumors)

tumor_size = st.sidebar.number_input("Tumor Size (cm³)", min_value=0.01, max_value=1000.0, value=0.45, step=0.1)
body_weight = st.sidebar.number_input("Body Weight (g)", min_value=1.0, max_value=150000.0, value=22.0, step=1.0)
gens = st.sidebar.slider("Optimization Generations", 10, 500, 60)

st.sidebar.header("Organ Sensitivity")
st.sidebar.info("Adjust weights to penalize accumulation")

# dynamic sliders for all healthy organs
healthy_organs = [org for org in inverse_model.target_names if "Tumor" not in org]
organ_weights = {}

for organ in healthy_organs:
    clean_label = organ.split(" ")[0]
    organ_weights[organ] = st.sidebar.slider(f"{clean_label} Penalty Weight", 1, 10, 1)

# optimization constraints UI
st.sidebar.header("Optimization Constraints")

st.sidebar.info("Filter core materials and set target penalties.")

with st.sidebar.expander("Allowed Core Materials", expanded=False):
    st.caption("Particle Type is mapped automatically based on the cores selected.")
    ui_allowed_cores = st.multiselect(
        "Select Cores to explore:",
        options=inverse_model.core_options,
        default=inverse_model.core_options
    )

with st.sidebar.expander("Adjust soft constraints", expanded=False):
    ui_min_tumor_conc = st.number_input("Min Tumor Accumulation (%ID/g)", value=10.0, step=1.0)
    ui_max_pdi = st.slider("Target Max PDI", min_value=0.01, max_value=0.95, value=0.30, step=0.01)
    ui_hd_range = st.slider("Target Size(HD) Bounds", min_value=3.0, max_value=400.0, value=(6.0, 310.0), step=1.0)
    ui_zeta_range = st.slider("Target Zeta Bounds", min_value=-60.0, max_value=40.0, value=(-45.0, 30.0), step=1.0)


# main dashboard
st.title("Nanoparticle Inverse-Design Portal")

run_button_col, clear_button_col = st.columns([1, 1])

with run_button_col:
    run_clicked = st.button(
        "Run Inverse Design Optimization",
        disabled=st.session_state.results_ready or st.session_state.is_running
    )

with clear_button_col:
    if st.session_state.results_ready:
        if st.button("Clear Results", type="secondary"):
            clear_results()
            st.rerun()

if run_clicked:
    st.session_state.is_running = True
    st.rerun()

# run optimization
if st.session_state.is_running:
    with st.spinner(f"Optimizing designs for tumor in {tumor_type}"):

        # create a Streamlit progress bar for the Optimization phase
        opt_progress_bar = st.progress(0, text="Starting NSGA-III algorithm")

        # run optimization and save to session state
        top_designs, opt_res = inverse_model.predict_optimal_design(
            tumor_cell=tumor_type,
            tumor_size=tumor_size,
            body_weight=body_weight,
            sensitive_organ_weights=organ_weights,
            generations=gens,
            allowed_cores=ui_allowed_cores,  # Passes only the cores
            max_pdi=ui_max_pdi,
            min_tumor_conc=ui_min_tumor_conc,
            min_hd_size=ui_hd_range[0],
            max_hd_size=ui_hd_range[1],
            min_zeta=ui_zeta_range[0],
            max_zeta=ui_zeta_range[1],
            ui_progress_bar=opt_progress_bar
        )

        # snap optimization progress to 100% when finished
        opt_progress_bar.progress(100, text="Optimization complete")

        # store everything in session state
        if top_designs is None or len(top_designs) == 0:
            st.session_state.is_running = False
            st.error("Optimization returned no valid designs. Try increasing generations or adjusting constraints.")
        else:
            # create a Streamlit progress bar for the plotting phase
            total_plots = len(healthy_organs)
            plot_progress_bar = st.progress(0, text="Initializing plot generation")
            # pre-generate all organ plots while still in spinner
            cached_plots = {}
            for i, organ in enumerate(healthy_organs):
                # Update the progress bar text and percentage for each organ
                current_percent = int((i / total_plots) * 100)
                clean_organ_name = organ.split(" ")[0]
                plot_progress_bar.progress(
                    current_percent,
                    text=f"Generating sensitivity plots for {clean_organ_name} ({i + 1}/{total_plots})..."
                )

                # generate and save the plot
                inverse_model.plot_design_analysis(opt_res, tumor_type, organ, allowed_cores=ui_allowed_cores)
                fig = plt.gcf()
                buffer = io.BytesIO()
                fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
                buffer.seek(0)
                cached_plots[organ] = buffer.getvalue()
                plt.close(fig)

            # snap progress to 100% when finished
            plot_progress_bar.progress(100, text="All plots generated")

            st.session_state.top_designs = top_designs
            st.session_state.opt_res_object = opt_res
            st.session_state.cached_organ_plots = cached_plots
            st.session_state.results_ready = True
            st.session_state.is_running = False
            st.rerun()

# check if results exist in memory to display them
if st.session_state.results_ready:
    # always pull the data from session_state
    top_designs = st.session_state.top_designs
    optimization_result = st.session_state.opt_res_object

    # result Table
    st.header("Top 10 Optimized Designs")
    results_list = [d['design_params'].copy() for d in top_designs]
    for i, row in enumerate(results_list):
        row['Selectivity Score'] = round(top_designs[i]['selectivity_score'], 4)

    params_df = pd.DataFrame(results_list)

    st.dataframe(params_df, width='stretch')

    # distribution visualization
    st.divider()
    st.header("Predicted Biodistributions (Top 10)")
    for row_idx in range(2):
        cols = st.columns(5)
        for col_idx in range(5):
            design_idx = row_idx * 5 + col_idx
            if design_idx < len(top_designs):
                design = top_designs[design_idx]
                with cols[col_idx]:
                    st.subheader(f"Rank #{design_idx + 1}")
                    st.metric("Score", f"{design['selectivity_score']:.2f}")
                    clean_dist = {k.split(" ")[0]: v for k, v in design['predicted_distribution'].items()}
                    #dist_df = pd.DataFrame.from_dict(clean_dist, orient='index', columns=['%ID/g'])
                    #st.bar_chart(dist_df)
                    dist_df = pd.DataFrame({
                        'Organ': list(clean_dist.keys()),
                        '%ID/g': list(clean_dist.values())
                    })
                    dist_df.columns = ['Organ', '%ID/g']

                    st.vega_lite_chart(dist_df, {
                        "mark": "bar",
                        "encoding": {
                            "x": {"field": "Organ", "type": "nominal", "title": ""},
                            "y": {"field": "%ID/g", "type": "quantitative", "title": "%ID/g"}
                        }
                    }, width='stretch')

    # Trade-off Analysis
    st.divider()
    st.header("Sensitivity & Trade-off Analysis")

    # Selecting this menu reruns the script, but finds the results in st.session_state
    analysis_organ = st.selectbox("Select Organ for Toxicity Trade-off", healthy_organs)
    # update the plot every time 'analysis_organ' changes
    with st.container():
        st.image(st.session_state.cached_organ_plots[analysis_organ], width='content')

        #inverse_model.plot_design_analysis(optimization_result, tumor_type, analysis_organ)

        # grab the plot from the global canvas and show it in the app
        #st.pyplot(plt.gcf())

        # clear the canvas for the next time the script runs
        #plt.clf()


    # CSV Download
    csv = params_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name=f"NP_Optimization_{tumor_type}.csv",
        mime='text/csv'
    )