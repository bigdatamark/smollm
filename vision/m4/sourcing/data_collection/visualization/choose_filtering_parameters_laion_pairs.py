# [MODULE DOC] Streamlit app (script only, no functions) to help choose filtering thresholds for image/text
# pair datasets.
# Loads a pickled DataFrame of per-pair stats from ./large_files/stats_vis_choose_filtering_params.pkl
# (produced by get_stats_vis_choose_filtering_parameters_laion_pairs.py; `img` column holds HTML <img> tags),
# lets the user pick the number of rows (head) and a sort column/order, and renders it as raw HTML.
import pandas as pd
import streamlit as st


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.title("Visualization to help choosing the filtering parameters for image / text pair datasets")

    path_stats = "./large_files/stats_vis_choose_filtering_params.pkl"
    df_stats = pd.read_pickle(path_stats)

    num_considered_examples = st.number_input(
        "Choose the number of image / text pairs to consider",
        min_value=0,
        max_value=len(df_stats),
        value=1_000,
        help=f"Enter a number between 0 and {len(df_stats)}",
    )
    df_stats = df_stats.head(num_considered_examples)

    order_sort = st.selectbox("Sorting in", options=["ascending order", "descending order"], index=0)
    stat_sort_on = st.selectbox(
        "Sorting on",
        options=[name for name in list(df_stats.columns.values) if name not in ["img", "caption"]],
        index=0,
    )
    ascending_sort = True if "ascending" in order_sort else False
    df_stats = df_stats.sort_values(stat_sort_on, ascending=ascending_sort)

    html_data_frame = df_stats.to_html(escape=False)
    st.markdown(html_data_frame, unsafe_allow_html=True)
