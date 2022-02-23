import streamlit as st


def show_home() -> None:
    st.write("""

        This demo lets you explore spelling error detection and correction models. You
        can run an interactive demo, run benchmarks, inspect model configurations and more.

        To start, open the **sidebar on the left** (if not already open) and choose the page you want to go to
        in the navigation. You can also configure other **options** there, e.g. which
        experiment/model to load and on which device you want to run it.

        """)
