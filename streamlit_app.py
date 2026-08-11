import hashlib
import json
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote

import pandas as pd
import requests
import streamlit as st


MODEL_ID = "marcelbinz/Llama-3.1-Centaur-70B"
API_BASE = "https://api.featherless.ai/v1"
TEMPERATURE = 1.0
TOP_P = 1.0
MAX_RETRIES = 4
MAX_APP_WORKERS = 16


st.set_page_config(
    page_title="Centaur Experiment Runner",
    layout="wide",
)


# ---------- Helpers ----------

def api_headers(api_key):
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": "Centaur Experiment Runner",
    }


def get_model_and_plan(api_key):
    headers = api_headers(api_key)
    encoded_model = quote(MODEL_ID, safe="")

    model_response = requests.get(
        f"{API_BASE}/models/{encoded_model}",
        headers=headers,
        timeout=30,
    )
    model_response.raise_for_status()

    plan_response = requests.get(
        f"{API_BASE}/plan",
        headers=headers,
        timeout=30,
    )
    plan_response.raise_for_status()

    return model_response.json(), plan_response.json()


def tokenize(api_key, text):
    response = requests.post(
        f"{API_BASE}/tokenize",
        headers=api_headers(api_key),
        json={"model": MODEL_ID, "text": text},
        timeout=30,
    )
    response.raise_for_status()
    return len(response.json().get("tokens", []))


def parse_response_codes(text):
    codes = [item.strip() for item in text.split(",") if item.strip()]
    return codes


def normalize_answer(raw_text, response_codes):
    text = (raw_text or "").strip()

    if text.startswith("<<"):
        text = text[2:].lstrip()
    if ">>" in text:
        text = text.split(">>", 1)[0].strip()

    text = text.strip(" \t\r\n\"'`.,;:()[]{}")

    for code in response_codes:
        if text.casefold() == code.casefold():
            return code

    for code in sorted(response_codes, key=len, reverse=True):
        if text.casefold().startswith(code.casefold()):
            remainder = text[len(code):].strip()
            if not remainder or remainder[0:1] in ".,;:!?)]}":
                return code

    return "INVALID"


def build_prompt(instructions, condition_text, response_codes):
    codes = ", ".join(response_codes)
    return f"""{instructions.strip()}

{condition_text.strip()}

Possible response codes: {codes}

You choose <<"""


def largest_remainder_counts(total, percentages):
    exact = [total * p / 100 for p in percentages]
    counts = [int(value) for value in exact]
    remainder = total - sum(counts)

    order = sorted(
        range(len(exact)),
        key=lambda i: exact[i] - counts[i],
        reverse=True,
    )

    for i in order[:remainder]:
        counts[i] += 1

    return counts


def config_fingerprint(config):
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_runtime_info(api_key):
    try:
        model_info, plan_info = get_model_and_plan(api_key)
        pricing = model_info.get("pricing", {})
        prompt_price = float(pricing.get("prompt", 0) or 0)
        completion_price = float(pricing.get("completion", 0) or 0)

        plan_concurrency = int(plan_info.get("concurrency", 1) or 1)
        model_concurrency_cost = int(model_info.get("concurrency_cost", 1) or 1)
        available_workers = max(1, plan_concurrency // max(1, model_concurrency_cost))
        workers = min(MAX_APP_WORKERS, available_workers)

        return {
            "model_info": model_info,
            "plan_info": plan_info,
            "prompt_price": prompt_price,
            "completion_price": completion_price,
            "workers": workers,
            "pricing_available": prompt_price > 0 or completion_price > 0,
            "error": None,
        }
    except Exception as exc:
        return {
            "model_info": {},
            "plan_info": {},
            "prompt_price": 0.0,
            "completion_price": 0.0,
            "workers": 1,
            "pricing_available": False,
            "error": str(exc),
        }


def estimate_run(api_key, conditions, response_codes, runtime_info):
    token_counts = {}
    used_fallback = False

    for condition in conditions:
        prompt = condition["prompt"]
        try:
            token_counts[condition["name"]] = tokenize(api_key, prompt)
        except Exception:
            # Only used if the tokenizer endpoint is temporarily unavailable.
            token_counts[condition["name"]] = max(1, round(len(prompt) / 4))
            used_fallback = True

    code_token_counts = []
    for code in response_codes:
        try:
            code_token_counts.append(max(1, tokenize(api_key, code)))
        except Exception:
            code_token_counts.append(max(1, round(len(code) / 4)))
            used_fallback = True

    estimated_output_tokens_per_run = max(code_token_counts) if code_token_counts else 1
    max_tokens = min(16, estimated_output_tokens_per_run + 2)

    estimated_prompt_tokens = sum(
        token_counts[c["name"]] * c["count"] for c in conditions
    )
    estimated_completion_tokens = (
        sum(c["count"] for c in conditions) * estimated_output_tokens_per_run
    )

    estimated_cost = None
    if runtime_info["pricing_available"]:
        estimated_cost = (
            estimated_prompt_tokens * runtime_info["prompt_price"]
            + estimated_completion_tokens * runtime_info["completion_price"]
        )

    return {
        "prompt_tokens": estimated_prompt_tokens,
        "completion_tokens": estimated_completion_tokens,
        "cost": estimated_cost,
        "max_tokens": max_tokens,
        "used_fallback": used_fallback,
    }


def run_one(job, api_key, response_codes, max_tokens, runtime_info):
    data = {
        "model": MODEL_ID,
        "prompt": job["prompt"],
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "stop": [">>", "\n"],
    }

    retryable_statuses = {408, 425, 429, 500, 502, 503, 504}
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                f"{API_BASE}/completions",
                headers=api_headers(api_key),
                json=data,
                timeout=120,
            )

            if response.status_code in retryable_statuses:
                raise RuntimeError(
                    f"Temporary Featherless error {response.status_code}: {response.text[:200]}"
                )

            response.raise_for_status()
            payload = response.json()
            raw = payload["choices"][0]["text"]
            answer = normalize_answer(raw, response_codes)
            usage = payload.get("usage", {})
            prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
            completion_tokens = int(usage.get("completion_tokens", 0) or 0)

            cost = None
            if runtime_info["pricing_available"]:
                cost = (
                    prompt_tokens * runtime_info["prompt_price"]
                    + completion_tokens * runtime_info["completion_price"]
                )

            return {
                "simulation_id": job["simulation_id"],
                "condition": job["condition"],
                "answer": answer,
                "raw": raw,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost": cost,
                "error": None,
                "attempts": attempt + 1,
            }

        except Exception as exc:
            last_error = str(exc)
            if attempt < MAX_RETRIES - 1:
                time.sleep((1.6 ** attempt) + random.uniform(0.1, 0.8))

    return {
        "simulation_id": job["simulation_id"],
        "condition": job["condition"],
        "answer": "ERROR",
        "raw": "",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost": None,
        "error": last_error,
        "attempts": MAX_RETRIES,
    }


def execute_experiment(config, api_key, runtime_info, estimate):
    jobs = []
    simulation_id = 1

    for condition in config["conditions"]:
        for _ in range(condition["count"]):
            jobs.append(
                {
                    "simulation_id": simulation_id,
                    "condition": condition["name"],
                    "prompt": condition["prompt"],
                }
            )
            simulation_id += 1

    random.shuffle(jobs)

    total = len(jobs)
    results = []
    completed = 0
    input_tokens = 0
    output_tokens = 0
    actual_cost = 0.0
    cost_known = runtime_info["pricing_available"]
    api_errors = 0
    invalid_outputs = 0

    st.subheader("Running experiment")
    progress_bar = st.progress(0)
    status = st.empty()

    m1, m2, m3, m4 = st.columns(4)
    completed_box = m1.empty()
    valid_box = m2.empty()
    token_box = m3.empty()
    cost_box = m4.empty()

    completed_box.metric("Completed", f"0 / {total:,}")
    valid_box.metric("Valid responses", "0")
    token_box.metric("Tokens used", "0")
    cost_box.metric("Cost so far", "$0.0000" if cost_known else "Unavailable")

    last_ui_update = 0.0
    workers = runtime_info["workers"]

    status.caption(
        f"Centaur is running with up to {workers} parallel request(s). "
        "Temporary capacity errors are retried automatically."
    )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                run_one,
                job,
                api_key,
                config["response_codes"],
                estimate["max_tokens"],
                runtime_info,
            ): job
            for job in jobs
        }

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            input_tokens += result["prompt_tokens"]
            output_tokens += result["completion_tokens"]

            if result["error"]:
                api_errors += 1
            elif result["answer"] == "INVALID":
                invalid_outputs += 1

            if result["cost"] is not None:
                actual_cost += result["cost"]

            now = time.time()
            if now - last_ui_update >= 0.25 or completed == total:
                valid_count = completed - api_errors - invalid_outputs
                progress_bar.progress(completed / total)
                completed_box.metric("Completed", f"{completed:,} / {total:,}")
                valid_box.metric("Valid responses", f"{valid_count:,}")
                token_box.metric("Tokens used", f"{input_tokens + output_tokens:,}")
                if cost_known:
                    cost_box.metric("Cost so far", f"${actual_cost:,.4f}")
                last_ui_update = now

    results.sort(key=lambda item: item["simulation_id"])
    progress_bar.progress(1.0)
    status.empty()

    return {
        "rows": results,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "actual_cost": actual_cost if cost_known else None,
        "api_errors": api_errors,
        "invalid_outputs": invalid_outputs,
        "total": total,
        "workers": workers,
    }


def render_results(run_data, config):
    rows = run_data["rows"]
    response_codes = config["response_codes"]
    primary = config["primary_outcome"]

    st.success(f"Experiment complete — {run_data['total']:,} simulation rounds finished.")

    if run_data["api_errors"]:
        st.warning(
            f"{run_data['api_errors']:,} API request(s) still failed after automatic retries. "
            "They are excluded from response percentages."
        )

    if run_data["invalid_outputs"]:
        st.warning(
            f"Centaur produced {run_data['invalid_outputs']:,} response(s) that did not match "
            "the allowed response codes. They are excluded from response percentages."
        )

    st.subheader(f"Primary outcome — {primary}")
    metric_columns = st.columns(len(config["conditions"]))

    baseline_share = None
    condition_summaries = {}

    for index, condition in enumerate(config["conditions"]):
        condition_rows = [
            r for r in rows if r["condition"] == condition["name"] and r["answer"] in response_codes
        ]
        valid_n = len(condition_rows)
        primary_n = sum(r["answer"] == primary for r in condition_rows)
        share = (primary_n / valid_n * 100) if valid_n else 0.0
        condition_summaries[condition["name"]] = {
            "valid_n": valid_n,
            "primary_n": primary_n,
            "primary_share": share,
        }

        if index == 0:
            baseline_share = share
            delta = None
        else:
            delta = share - baseline_share

        with metric_columns[index]:
            st.metric(
                condition["name"],
                f"{share:.1f}%",
                None if delta is None else f"{delta:+.1f} pp vs {config['conditions'][0]['name']}",
            )
            st.caption(f"{primary_n:,} of {valid_n:,} valid simulations")

    st.subheader("Response distribution")
    chart_rows = []
    summary_rows = []

    for condition in config["conditions"]:
        valid_answers = [
            r["answer"]
            for r in rows
            if r["condition"] == condition["name"] and r["answer"] in response_codes
        ]
        counts = Counter(valid_answers)
        valid_n = len(valid_answers)

        for code in response_codes:
            count = counts.get(code, 0)
            share = (count / valid_n * 100) if valid_n else 0.0
            chart_rows.append(
                {
                    "Response": code,
                    "Share (%)": share,
                    "Condition": condition["name"],
                }
            )
            summary_rows.append(
                {
                    "Condition": condition["name"],
                    "Response": code,
                    "Count": count,
                    "Share (%)": round(share, 2),
                    "Valid N": valid_n,
                }
            )

    chart_df = pd.DataFrame(chart_rows)
    st.bar_chart(
        chart_df,
        x="Response",
        y="Share (%)",
        color="Condition",
        stack=False,
        height=420,
    )

    with st.expander("View result table"):
        st.dataframe(pd.DataFrame(summary_rows), hide_index=True, width="stretch")

    st.subheader("Usage & cost")
    u1, u2, u3, u4 = st.columns(4)
    total_tokens = run_data["input_tokens"] + run_data["output_tokens"]

    u1.metric("Input tokens", f"{run_data['input_tokens']:,}")
    u2.metric("Output tokens", f"{run_data['output_tokens']:,}")
    u3.metric("Total tokens", f"{total_tokens:,}")

    if run_data["actual_cost"] is not None:
        u4.metric("Calculated API cost", f"${run_data['actual_cost']:,.4f}")
        per_round = run_data["actual_cost"] / run_data["total"] if run_data["total"] else 0
        st.caption(
            f"Average API cost per simulation round: ${per_round:,.6f}. "
            "Cost is calculated from the token counts returned by Featherless and its live model pricing."
        )
    else:
        u4.metric("Calculated API cost", "Unavailable")
        st.caption(
            "Token counts were recorded, but Featherless pricing could not be retrieved for this run."
        )

    with st.expander("Run quality details"):
        st.write(f"API failures after retries: **{run_data['api_errors']:,}**")
        st.write(f"Invalid model outputs: **{run_data['invalid_outputs']:,}**")
        st.write(f"Parallel requests used: **up to {run_data['workers']}**")
        st.write(f"Model: `{MODEL_ID}`")
        st.write(f"Temperature: `{TEMPERATURE}`")

    st.caption(
        "These are Centaur simulations, not observations from real human participants. "
        "Sampling uncertainty describes Centaur's generated responses, not uncertainty in a human population."
    )


# ---------- App ----------

st.caption("Updated by KB, 11.08.2026, 14.00 CET")
st.title("Centaur Experiment Runner")
st.write(
    "Run behavioural experiments using Llama-3.1-Centaur-70B by Binz et al. (2025) "
    "and examine simulated response patterns."
)

if "FEATHERLESS_API_KEY" not in st.secrets:
    st.error("Featherless API key not found. Add FEATHERLESS_API_KEY in Streamlit Secrets.")
    st.stop()

api_key = st.secrets["FEATHERLESS_API_KEY"]

st.divider()

experiment_name = st.text_input(
    "Experiment name",
    placeholder="e.g. Loss framing experiment",
)

instructions = st.text_area(
    "Shared experiment instructions",
    placeholder="Enter the information all simulated participants receive.",
    height=130,
)

setup_col1, setup_col2 = st.columns(2)

with setup_col1:
    number_conditions = st.selectbox(
        "Number of conditions",
        [1, 2, 3],
        index=1,
    )

with setup_col2:
    total_simulations = int(
        st.number_input(
            "Total simulation rounds",
            min_value=1,
            max_value=1000,
            value=100,
            step=1,
            help="Use the +/- buttons or type any number from 1 to 1,000.",
        )
    )

responses_text = st.text_input(
    "Possible response codes",
    value="A, B",
    help="Separate responses with commas. Short codes such as A, B, C are the most robust.",
)
response_codes = parse_response_codes(responses_text)

primary_outcome = None
if response_codes:
    primary_outcome = st.selectbox(
        "Primary outcome",
        response_codes,
        help="The result you care about most. Treatment differences will be shown for this response.",
    )

st.subheader("Conditions")

if number_conditions == 1:
    default_allocations = [100]
elif number_conditions == 2:
    default_allocations = [50, 50]
else:
    default_allocations = [34, 33, 33]

conditions_input = []
default_names = ["Control", "Treatment A", "Treatment B"]

for i in range(number_conditions):
    st.markdown(f"#### Condition {chr(65 + i)}")
    c1, c2 = st.columns([4, 1])

    with c1:
        name = st.text_input(
            f"Condition {chr(65 + i)} name",
            value=default_names[i] if number_conditions > 1 else "Condition A",
            key=f"condition_name_{i}",
        )

        text = st.text_area(
            f"Condition {chr(65 + i)} content",
            placeholder="Enter exactly what this condition shows or tells the participant.",
            height=120,
            key=f"condition_text_{i}",
        )

    with c2:
        allocation = int(
            st.number_input(
                "Allocation %",
                min_value=0,
                max_value=100,
                value=default_allocations[i],
                step=1,
                key=f"allocation_{number_conditions}_{i}",
            )
        )

    conditions_input.append(
        {
            "name": name.strip(),
            "text": text.strip(),
            "allocation": allocation,
        }
    )

allocation_total = sum(c["allocation"] for c in conditions_input)
allocation_counts = largest_remainder_counts(
    total_simulations,
    [c["allocation"] for c in conditions_input],
) if allocation_total == 100 else [0] * number_conditions

allocation_cols = st.columns(number_conditions)
for i, condition in enumerate(conditions_input):
    with allocation_cols[i]:
        st.metric(
            condition["name"] or f"Condition {chr(65 + i)}",
            f"{condition['allocation']}%",
            f"{allocation_counts[i]:,} simulation rounds" if allocation_total == 100 else None,
        )

if allocation_total != 100:
    st.error(f"Condition allocations currently total {allocation_total}%. They must total exactly 100%.")

with st.expander("View the Centaur prompt format"):
    if response_codes and conditions_input and conditions_input[0]["text"] and instructions:
        st.code(build_prompt(instructions, conditions_input[0]["text"], response_codes))
    else:
        st.caption("Fill in the experiment fields to preview the exact prompt Centaur will receive.")


def validate_inputs():
    errors = []

    if not experiment_name.strip():
        errors.append("Add an experiment name.")
    if not instructions.strip():
        errors.append("Add the shared experiment instructions.")
    if not response_codes:
        errors.append("Add at least one possible response code.")
    if len({code.casefold() for code in response_codes}) != len(response_codes):
        errors.append("Response codes must be unique.")
    if allocation_total != 100:
        errors.append("Condition allocations must total exactly 100%.")

    names = [condition["name"] for condition in conditions_input]
    if any(not name for name in names):
        errors.append("Give every condition a name.")
    if len({name.casefold() for name in names if name}) != len(names):
        errors.append("Condition names must be unique.")

    for i, condition in enumerate(conditions_input):
        if not condition["text"]:
            errors.append(f"Add content for Condition {chr(65 + i)}.")
        if condition["allocation"] <= 0:
            errors.append(f"Condition {chr(65 + i)} must receive more than 0% of simulations.")
        if allocation_total == 100 and allocation_counts[i] == 0:
            errors.append(
                f"Condition {chr(65 + i)} receives 0 rounds after rounding. "
                "Increase total simulation rounds or its allocation."
            )

    if total_simulations < number_conditions:
        errors.append("Total simulation rounds must be at least the number of active conditions.")

    return errors


def build_config():
    built_conditions = []
    for i, condition in enumerate(conditions_input):
        built_conditions.append(
            {
                "name": condition["name"],
                "text": condition["text"],
                "allocation": condition["allocation"],
                "count": allocation_counts[i],
                "prompt": build_prompt(instructions, condition["text"], response_codes),
            }
        )

    return {
        "experiment_name": experiment_name.strip(),
        "instructions": instructions.strip(),
        "response_codes": response_codes,
        "primary_outcome": primary_outcome,
        "total_simulations": total_simulations,
        "conditions": built_conditions,
    }


run_clicked = st.button("Run experiment", type="primary", width="stretch")

if run_clicked:
    errors = validate_inputs()
    if errors:
        for error in errors:
            st.error(error)
    else:
        config = build_config()
        fingerprint = config_fingerprint(config)

        with st.spinner("Checking Featherless pricing and preparing the run..."):
            runtime_info = get_runtime_info(api_key)
            estimate = estimate_run(api_key, config["conditions"], response_codes, runtime_info)

        st.session_state["prepared_run"] = {
            "fingerprint": fingerprint,
            "config": config,
            "runtime_info": runtime_info,
            "estimate": estimate,
        }

        if total_simulations <= 500:
            st.session_state["run_now"] = True
        else:
            st.session_state["run_now"] = False


prepared = st.session_state.get("prepared_run")
current_config = None
current_fingerprint = None

if not validate_inputs():
    current_config = build_config()
    current_fingerprint = config_fingerprint(current_config)

if prepared and current_fingerprint == prepared["fingerprint"]:
    estimate = prepared["estimate"]
    runtime_info = prepared["runtime_info"]

    st.subheader("Run plan")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Simulation rounds", f"{prepared['config']['total_simulations']:,}")
    p2.metric("Estimated input tokens", f"{estimate['prompt_tokens']:,}")
    p3.metric("Estimated output tokens", f"{estimate['completion_tokens']:,}")
    p4.metric(
        "Estimated API cost",
        f"~${estimate['cost']:,.4f}" if estimate["cost"] is not None else "Unavailable",
    )

    if estimate["used_fallback"]:
        st.caption("Some token estimates used a rough fallback because the Featherless tokenizer was unavailable.")

    if runtime_info["error"]:
        st.warning(
            "Live Featherless model/plan information could not be retrieved. "
            "The run can still proceed, but cost calculation may be unavailable and requests will run sequentially."
        )

    if prepared["config"]["total_simulations"] > 500 and not st.session_state.get("run_now", False):
        confirmation = st.checkbox(
            f"I confirm I want to run {prepared['config']['total_simulations']:,} paid Centaur simulations.",
            key="large_run_confirmation",
        )
        if st.button(
            "Confirm and start large run",
            type="primary",
            disabled=not confirmation,
            width="stretch",
        ):
            st.session_state["run_now"] = True

    if st.session_state.get("run_now", False):
        # Reset first so an unrelated Streamlit rerun does not accidentally launch the paid job again.
        st.session_state["run_now"] = False
        run_data = execute_experiment(
            prepared["config"],
            api_key,
            prepared["runtime_info"],
            prepared["estimate"],
        )
        st.session_state["last_run"] = {
            "fingerprint": prepared["fingerprint"],
            "config": prepared["config"],
            "run_data": run_data,
        }

last_run = st.session_state.get("last_run")
if last_run:
    st.divider()
    st.header("Results")
    render_results(last_run["run_data"], last_run["config"])
    