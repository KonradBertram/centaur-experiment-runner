import streamlit as st
import requests

st.set_page_config(
    page_title="Centaur Experiment Runner",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Centaur Experiment Runner")

st.write(
    "Run behavioural experiments using Centaur and compare simulated responses across conditions."
)

st.divider()

experiment_name = st.text_input(
    "Experiment name",
    placeholder="e.g. Loss framing experiment"
)

instructions = st.text_area(
    "Experiment instructions",
    placeholder="Describe what the participant should imagine or what information they receive.",
    height=150
)

col1, col2 = st.columns(2)

with col1:
    condition_a = st.text_area(
        "Condition A",
        placeholder="Enter the first version of your treatment.",
        height=150
    )

with col2:
    condition_b = st.text_area(
        "Condition B",
        placeholder="Leave blank for this first test.",
        height=150
    )

responses = st.text_input(
    "Possible responses",
    placeholder="e.g. A, B"
)

simulations = st.selectbox(
    "Number of Centaur simulations per condition",
    [1, 10, 20, 50, 100, 500],
    index=0
)

if st.button("Run experiment", type="primary"):

    if not instructions or not condition_a or not responses:
        st.warning("Please fill in the instructions, Condition A, and possible responses.")

    else:
        prompt = f"""
You are participating in a behavioural experiment.

Experiment instructions:
{instructions}

Condition:
{condition_a}

Possible responses:
{responses}

Give your choice only.

Your choice: <<
"""

        headers = {
            "Authorization": f"Bearer {st.secrets['FEATHERLESS_API_KEY']}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "marcelbinz/Llama-3.1-Centaur-70B",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 10,
            "temperature": 1.0
        }

        with st.spinner("Centaur is thinking..."):
            try:
                response = requests.post(
                    "https://api.featherless.ai/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=120
                )

                response.raise_for_status()

                result = response.json()
                answer = result["choices"][0]["message"]["content"]

                st.success("Centaur responded successfully!")
                st.subheader("Centaur response")
                st.write(answer)

            except Exception as e:
                st.error("Something went wrong.")
                st.code(str(e))