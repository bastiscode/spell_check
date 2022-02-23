import time
from typing import Any, Dict

import streamlit as st

from gnn_lib import models, tasks
from gnn_lib.data import utils
from gnn_lib.modules import inference


def show_inference(task: tasks.Task, model: models.Model, inference_kwargs: Dict[str, Any]) -> None:
    st.write("#### Test a model interactively")

    sequence = utils.clean_sequence(st.text_input("Input a sequence"))
    if sequence == "":
        st.stop()

    start = time.monotonic()
    g = task.variant.prepare_sequences_for_inference([sequence])
    outputs = task.inference(model, [sequence], **inference_kwargs)[0]
    end = time.monotonic()

    st.write(f"###### *Inference took {(end - start) * 1000:.2f}ms*")
    outputs = inference.inference_output_to_str(outputs)
    if isinstance(outputs, list):
        output_str = "\n".join(outputs)
    else:
        output_str = str(outputs)
    st.code(output_str)

    st.write(f"###### *Input graph*")
    st.code(g)
