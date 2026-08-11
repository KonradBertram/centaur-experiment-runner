import streamlit as st
import requests
from collections import Counter

st.set_page_config(
    page_title="Centaur Experiment Runner",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Centaur Experiment Runner")

st.write(
    "Run behavioural experiments using Llama-3.1-Centaur-70B by Binz et al. (2025) and examine simulated response patterns."
)

st.divider()

experiment_name = st.text_input(
    "Experiment name",
    placeholder="e.g. Risk choice experiment"
)

instructions = st.text_area(
    "Experiment instructions",
    placeholder="Describe exactly what the participant sees.",
    height=150
)

condition = st.text_area(
    "Experimental condition",
    placeholder="Enter the choice or treatment shown to the participant.",
    height=150
)

responses = st.text_input(
    "Possible response codes",
    value="A, B",
    help="For now, use short response codes such as A and B."
)

simulations = st.selectbox(
    "Number of Centaur simulations",
    [1, 10],
    index=1
)

if st.button("Run experiment", type="primary"):

    if not instructions or not condition or not responses:
        st.warning("Please fill in all fields.")

    else:

        prompt = f"""{instructions}

{condition}

Possible responses: {responses}

You choose <<"""

        headers = {
            "Authorization": f"Bearer {st.secrets['FEATHERLESS_API_KEY']}",
            "Content-Type": "application/json"
        }

        answers = []

        progress = st.progress(0)
        status = st.empty()

        for i in range(simulations):

            status.write(
                f"Running Centaur simulation {i + 1} of {simulations}..."
            )

            data = {
                "model": "marcelbinz/Llama-3.1-Centaur-70B",
                "prompt": prompt,
                "max_tokens": 1,
                "temperature": 1.0
            }

            try:

                response = requests.post(
                    "https://api.featherless.ai/v1/completions",
                    headers=headers,
                    json=data,
                    timeout=120
                )

                response.raise_for_status()

                result = response.json()

                answer = result["choices"][0]["text"].strip()

                answers.append(answer)

            except Exception as e:

                answers.append("ERROR")

            progress.progress((i + 1) / simulations)

        status.empty()

        st.success("Experiment complete!")

        counts = Counter(answers)

        st.subheader("Results")

        for answer, count in counts.items():

            percentage = (count / simulations) * 100

            st.metric(
                label=f"Choice {answer}",
                value=f"{percentage:.0f}%",
                delta=f"{count} of {simulations} simulations"
            )

        st.subheader("Individual Centaur simulations")

        st.write(answers)

        st.caption(
            "These are Centaur simulations, not observations from real human participants."
        )
