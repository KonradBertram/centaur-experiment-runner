import hashlib
import json
import math
import random
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from urllib.parse import quote

import pandas as pd
import requests
import streamlit as st


MODEL_ID = "marcelbinz/Llama-3.1-Centaur-70B"
API_BASE = "https://api.featherless.ai/v1"

DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 1.0
DEFAULT_USE_TOP_K = False
DEFAULT_TOP_K = 50
MAX_NEW_TOKENS = 1

MAX_HTTP_RETRIES = 4
MAX_APP_WORKERS = 10
MAX_SIMULATIONS = 1000
MAX_SLOT_ATTEMPT_MULTIPLIER = 3


st.set_page_config(
    page_title="Centaur Experiment Runner",
    layout="wide",
)
st.html("""
<style>
button[data-testid="stBaseButton-primary"],
button[data-testid="stBaseButton-primary"] * {
    color: #000000 !important;
}
</style>
""")

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


def heuristic_token_count(text):
    return max(1, math.ceil(len(text or "") / 4))


def tokenize_count(api_key, text):
    response = requests.post(
        f"{API_BASE}/tokenize",
        headers=api_headers(api_key),
        json={"model": MODEL_ID, "text": text},
        timeout=30,
    )
    response.raise_for_status()

    tokens = response.json().get("tokens")

    if not isinstance(tokens, list):
        raise ValueError("Tokenizer response did not contain a token list.")

    count = len(tokens)

    if text.strip() and count <= 0:
        raise ValueError("Tokenizer returned zero tokens for non-empty text.")

    return max(1, count)


def parse_response_codes(text):
    return [item.strip() for item in text.split(",") if item.strip()]


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
        available_workers = max(
            1,
            plan_concurrency // max(1, model_concurrency_cost),
        )
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


def estimate_run(api_key, conditions, runtime_info):
    condition_estimates = []
    used_fallback = False
    total_prompt_tokens = 0

    for condition in conditions:
        prompt = condition["prompt"]

        try:
            prompt_tokens_per_attempt = tokenize_count(api_key, prompt)
            source = "Featherless tokenizer"
        except Exception:
            prompt_tokens_per_attempt = heuristic_token_count(prompt)
            source = "character-based fallback"
            used_fallback = True

        condition_prompt_tokens = (
            prompt_tokens_per_attempt * condition["count"]
        )
        total_prompt_tokens += condition_prompt_tokens

        condition_estimates.append(
            {
                "condition": condition["name"],
                "tokens_per_prompt": prompt_tokens_per_attempt,
                "target_valid_n": condition["count"],
                "estimated_input_tokens": condition_prompt_tokens,
                "token_source": source,
            }
        )

    estimated_completion_tokens = (
        sum(condition["count"] for condition in conditions)
        * MAX_NEW_TOKENS
    )

    base_cost = None
    safety_ceiling_cost = None

    if runtime_info["pricing_available"]:
        base_cost = (
            total_prompt_tokens * runtime_info["prompt_price"]
            + estimated_completion_tokens * runtime_info["completion_price"]
        )
        safety_ceiling_cost = (
            base_cost * MAX_SLOT_ATTEMPT_MULTIPLIER
        )

    return {
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": estimated_completion_tokens,
        "base_cost": base_cost,
        "safety_ceiling_cost": safety_ceiling_cost,
        "used_fallback": used_fallback,
        "condition_estimates": condition_estimates,
    }


def run_one(job, api_key, response_codes, settings, runtime_info):
    data = {
        "model": MODEL_ID,
        "prompt": job["prompt"],
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": settings["temperature"],
        "top_p": settings["top_p"],
        "stop": [">>", "\n"],
    }

    if settings["use_top_k"]:
        data["top_k"] = settings["top_k"]

    retryable_statuses = {408, 425, 429, 500, 502, 503, 504}
    last_error = None
    retry_events = 0
    http_attempts = 0

    for attempt in range(MAX_HTTP_RETRIES):
        http_attempts += 1

        try:
            response = requests.post(
                f"{API_BASE}/completions",
                headers=api_headers(api_key),
                json=data,
                timeout=120,
            )

            if response.status_code in retryable_statuses:
                last_error = (
                    f"Temporary Featherless error "
                    f"{response.status_code}: {response.text[:200]}"
                )

                if attempt < MAX_HTTP_RETRIES - 1:
                    retry_events += 1
                    time.sleep(
                        (1.6 ** attempt)
                        + random.uniform(0.1, 0.8)
                    )
                    continue

                break

            response.raise_for_status()

            payload = response.json()
            raw = payload["choices"][0]["text"]
            answer = normalize_answer(raw, response_codes)

            usage = payload.get("usage", {})
            prompt_tokens = int(
                usage.get("prompt_tokens", 0) or 0
            )
            completion_tokens = int(
                usage.get("completion_tokens", 0) or 0
            )

            cost = None
            if runtime_info["pricing_available"]:
                cost = (
                    prompt_tokens * runtime_info["prompt_price"]
                    + completion_tokens * runtime_info["completion_price"]
                )

            return {
                "attempt_id": job["attempt_id"],
                "condition": job["condition"],
                "answer": answer,
                "raw": raw,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost": cost,
                "error": None,
                "http_attempts": http_attempts,
                "retry_events": retry_events,
            }

        except requests.RequestException as exc:
            last_error = str(exc)

            status = getattr(
                getattr(exc, "response", None),
                "status_code",
                None,
            )

            should_retry = (
                status is None
                or status in retryable_statuses
            )

            if (
                should_retry
                and attempt < MAX_HTTP_RETRIES - 1
            ):
                retry_events += 1
                time.sleep(
                    (1.6 ** attempt)
                    + random.uniform(0.1, 0.8)
                )
                continue

            break

        except Exception as exc:
            last_error = str(exc)

            if attempt < MAX_HTTP_RETRIES - 1:
                retry_events += 1
                time.sleep(
                    (1.6 ** attempt)
                    + random.uniform(0.1, 0.8)
                )
                continue

            break

    return {
        "attempt_id": job["attempt_id"],
        "condition": job["condition"],
        "answer": "ERROR",
        "raw": "",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost": None,
        "error": last_error or "Unknown API error",
        "http_attempts": http_attempts,
        "retry_events": retry_events,
    }


def execute_experiment(config, api_key, runtime_info):
    response_codes = config["response_codes"]
    settings = config["settings"]

    target_by_condition = {
        condition["name"]: condition["count"]
        for condition in config["conditions"]
    }

    max_slots_by_condition = {
        name: target * MAX_SLOT_ATTEMPT_MULTIPLIER
        for name, target in target_by_condition.items()
    }

    prompt_by_condition = {
        condition["name"]: condition["prompt"]
        for condition in config["conditions"]
    }

    valid_by_condition = {
        name: 0 for name in target_by_condition
    }
    submitted_by_condition = {
        name: 0 for name in target_by_condition
    }

    all_results = []
    accepted_rows = []

    input_tokens = 0
    output_tokens = 0
    actual_cost = 0.0
    cost_known = runtime_info["pricing_available"]

    invalid_outputs = 0
    api_failed_slots = 0
    retry_events = 0
    http_requests = 0

    total_target = sum(target_by_condition.values())
    next_attempt_id = 1

    st.subheader("Running experiment")
    progress_bar = st.progress(0)

    m1, m2, m3, m4, m5 = st.columns(5)
    valid_box = m1.empty()
    invalid_box = m2.empty()
    api_error_box = m3.empty()
    token_box = m4.empty()
    cost_box = m5.empty()

    valid_box.metric(
        "Valid responses",
        f"0 / {total_target:,}",
    )
    invalid_box.metric(
        "Invalid outputs retried",
        "0",
    )
    api_error_box.metric(
        "API failures retried",
        "0",
    )
    token_box.metric(
        "Tokens used",
        "0",
    )
    cost_box.metric(
        "Cost so far",
        "$0.0000" if cost_known else "Unavailable",
    )

    status = st.empty()
    workers = runtime_info["workers"]
    last_ui_update = 0.0

    def make_job(condition_name):
        nonlocal next_attempt_id

        job = {
            "attempt_id": next_attempt_id,
            "condition": condition_name,
            "prompt": prompt_by_condition[condition_name],
        }

        next_attempt_id += 1
        submitted_by_condition[condition_name] += 1
        return job

    initial_jobs = []

    for condition_name, target in target_by_condition.items():
        for _ in range(target):
            initial_jobs.append(
                make_job(condition_name)
            )

    random.shuffle(initial_jobs)

    with ThreadPoolExecutor(
        max_workers=workers
    ) as executor:
        pending = {}

        for job in initial_jobs:
            future = executor.submit(
                run_one,
                job,
                api_key,
                response_codes,
                settings,
                runtime_info,
            )
            pending[future] = job

        while pending:
            done, _ = wait(
                pending,
                return_when=FIRST_COMPLETED,
            )

            for future in done:
                job = pending.pop(future)
                condition_name = job["condition"]

                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "attempt_id": job["attempt_id"],
                        "condition": condition_name,
                        "answer": "ERROR",
                        "raw": "",
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "cost": None,
                        "error": str(exc),
                        "http_attempts": 0,
                        "retry_events": 0,
                    }

                all_results.append(result)
                input_tokens += result["prompt_tokens"]
                output_tokens += result["completion_tokens"]
                retry_events += result["retry_events"]
                http_requests += result["http_attempts"]

                if result["cost"] is not None:
                    actual_cost += result["cost"]

                is_valid = (
                    result["error"] is None
                    and result["answer"] in response_codes
                )

                needs_replacement = False

                if is_valid:
                    valid_by_condition[condition_name] += 1
                    accepted_rows.append(result)

                elif result["error"] is not None:
                    api_failed_slots += 1
                    needs_replacement = True

                else:
                    invalid_outputs += 1
                    needs_replacement = True

                if (
                    needs_replacement
                    and submitted_by_condition[condition_name]
                    < max_slots_by_condition[condition_name]
                    and valid_by_condition[condition_name]
                    < target_by_condition[condition_name]
                ):
                    replacement = make_job(condition_name)

                    replacement_future = executor.submit(
                        run_one,
                        replacement,
                        api_key,
                        response_codes,
                        settings,
                        runtime_info,
                    )
                    pending[replacement_future] = replacement

                valid_total = sum(
                    valid_by_condition.values()
                )
                now = time.time()

                if (
                    now - last_ui_update >= 0.25
                    or valid_total == total_target
                ):
                    progress_bar.progress(
                        min(
                            1.0,
                            valid_total / total_target,
                        )
                    )

                    valid_box.metric(
                        "Valid responses",
                        f"{valid_total:,} / {total_target:,}",
                    )
                    invalid_box.metric(
                        "Invalid outputs retried",
                        f"{invalid_outputs:,}",
                    )
                    api_error_box.metric(
                        "API failures retried",
                        f"{api_failed_slots:,}",
                    )
                    token_box.metric(
                        "Tokens used",
                        f"{input_tokens + output_tokens:,}",
                    )

                    if cost_known:
                        cost_box.metric(
                            "Cost so far",
                            f"${actual_cost:,.4f}",
                        )

                    status.caption(
                        f"Up to {workers} requests in parallel · "
                        f"{len(all_results):,} completed simulation attempts · "
                        f"{retry_events:,} temporary API retry event(s)"
                    )

                    last_ui_update = now

    valid_total = sum(valid_by_condition.values())

    progress_bar.progress(
        min(1.0, valid_total / total_target)
    )
    status.empty()

    all_results.sort(
        key=lambda item: item["attempt_id"]
    )
    accepted_rows.sort(
        key=lambda item: item["attempt_id"]
    )

    return {
        "rows": all_results,
        "accepted_rows": accepted_rows,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "actual_cost": (
            actual_cost if cost_known else None
        ),
        "invalid_outputs": invalid_outputs,
        "api_failed_slots": api_failed_slots,
        "retry_events": retry_events,
        "http_requests": http_requests,
        "simulation_attempts": len(all_results),
        "valid_by_condition": valid_by_condition,
        "target_by_condition": target_by_condition,
        "valid_total": valid_total,
        "total_target": total_target,
        "workers": workers,
    }


def render_results(run_data, config):
    response_codes = config["response_codes"]

    if (
        run_data["valid_total"]
        == run_data["total_target"]
    ):
        st.success(
            "Experiment complete — collected all "
            f"{run_data['total_target']:,} requested "
            "valid Centaur responses."
        )
    else:
        st.warning(
            "The safety ceiling was reached before the "
            "full target was collected: "
            f"{run_data['valid_total']:,} of "
            f"{run_data['total_target']:,} valid responses. "
            "Review the response codes/prompt before running again."
        )

    st.subheader("Response distribution")

    chart_rows = []
    table_rows = []

    for condition in config["conditions"]:
        condition_name = condition["name"]

        valid_answers = [
            row["answer"]
            for row in run_data["accepted_rows"]
            if row["condition"] == condition_name
        ]

        counts = Counter(valid_answers)
        valid_n = len(valid_answers)

        table_row = {
            "Condition": condition_name,
            "Valid N": valid_n,
            "Target N": condition["count"],
        }

        for code in response_codes:
            count = counts.get(code, 0)
            share = (
                count / valid_n * 100
                if valid_n
                else 0.0
            )

            chart_rows.append(
                {
                    "Condition": condition_name,
                    "Response": code,
                    "Share (%)": share,
                }
            )

            table_row[code] = (
                f"{share:.1f}% ({count:,})"
            )

        table_rows.append(table_row)

    chart_df = pd.DataFrame(chart_rows)

    st.bar_chart(
        chart_df,
        x="Condition",
        y="Share (%)",
        color="Response",
        stack=False,
        height=430,
    )

    st.caption(
        "Percentages are calculated within each condition. "
        "The response bars for each condition therefore sum to 100%."
    )

    st.subheader("Results table")

    ordered_columns = (
        ["Condition"]
        + response_codes
        + ["Valid N", "Target N"]
    )

    table_df = pd.DataFrame(
        table_rows
    )[ordered_columns]

    st.dataframe(
        table_df,
        hide_index=True,
        width="stretch",
    )

    st.subheader("Usage & run quality")

    u1, u2, u3, u4 = st.columns(4)

    total_tokens = (
        run_data["input_tokens"]
        + run_data["output_tokens"]
    )

    u1.metric(
        "Input tokens",
        f"{run_data['input_tokens']:,}",
    )
    u2.metric(
        "Output tokens",
        f"{run_data['output_tokens']:,}",
    )
    u3.metric(
        "Total tokens",
        f"{total_tokens:,}",
    )

    if run_data["actual_cost"] is not None:
        u4.metric(
            "Calculated API cost",
            f"${run_data['actual_cost']:,.4f}",
        )
    else:
        u4.metric(
            "Calculated API cost",
            "Unavailable",
        )

    q1, q2, q3, q4 = st.columns(4)

    q1.metric(
        "Invalid Centaur outputs",
        f"{run_data['invalid_outputs']:,}",
        help=(
            "Successful model responses that did not "
            "match an allowed response code. They were replaced."
        ),
    )

    q2.metric(
        "API failures after retries",
        f"{run_data['api_failed_slots']:,}",
        help=(
            "Simulation slots where the API still failed "
            "after its internal retries. They were replaced "
            "where the safety ceiling allowed."
        ),
    )

    q3.metric(
        "Temporary API retry events",
        f"{run_data['retry_events']:,}",
        help=(
            "Temporary capacity/network problems that "
            "triggered an automatic retry."
        ),
    )

    q4.metric(
        "Simulation attempts",
        f"{run_data['simulation_attempts']:,}",
        help=(
            "Valid responses plus invalid/API-failed "
            "simulation slots. This can exceed requested N."
        ),
    )

    with st.expander("Technical run details"):
        st.write(
            "Underlying HTTP requests: "
            f"**{run_data['http_requests']:,}**"
        )
        st.write(
            "Parallel requests used: "
            f"**up to {run_data['workers']}**"
        )
        st.write(f"Model: `{MODEL_ID}`")
        st.write(
            "Temperature: "
            f"`{config['settings']['temperature']}`"
        )
        st.write(
            "Top-p: "
            f"`{config['settings']['top_p']}`"
        )

        if config["settings"]["use_top_k"]:
            st.write(
                "Top-k: "
                f"`{config['settings']['top_k']}`"
            )
        else:
            st.write(
                "Top-k: **disabled / provider default**"
            )

        st.write(
            "Generated tokens per Centaur choice: "
            f"`{MAX_NEW_TOKENS}`"
        )

    st.caption(
        "These are Centaur simulations, not observations "
        "from real human participants."
    )


st.image("logo.png", width=130)

st.caption("Updated by KB, 12.08.2026, 18.30 CET")
st.title("Centaur Experiment Runner")
st.write(
    "Run behavioural experiments using "
    "Llama-3.1-Centaur-70B by Binz et al. (2025) "
    "and examine simulated response patterns."
)

try:
    api_key = st.secrets[
        "FEATHERLESS_API_KEY"
    ]
except Exception:
    st.error(
        "Featherless API key is not available in this "
        "environment. If you are viewing the app inside "
        "GitHub Codespaces, open the deployed Streamlit app instead."
    )
    st.stop()

st.divider()

experiment_name = st.text_input(
    "Experiment name",
    placeholder="e.g. Hypertemporal discounting experiment",
)

instructions = st.text_area(
    "Survey question or experiment instructions including answer options",
    placeholder=(
        "Which of the options would you prefer=" \
        "A: 10 CHF now" \
        "B: 20 CHF with 50% probability"
    ),
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
            "Total valid simulation responses",
            min_value=1,
            max_value=MAX_SIMULATIONS,
            value=100,
            step=1,
            help=(
                "This is the number of valid Centaur "
                "responses the app will try to collect, "
                "from 1 to 1,000."
            ),
        )
    )

responses_text = st.text_input(
    "Possible response codes",
    value="A, B",
    help=(
        "Separate responses with commas. Short response "
        "codes such as A, B, C are the most robust."
    ),
)

response_codes = parse_response_codes(
    responses_text
)


if "centaur_temperature" not in st.session_state:
    st.session_state[
        "centaur_temperature"
    ] = DEFAULT_TEMPERATURE

if "centaur_top_p" not in st.session_state:
    st.session_state[
        "centaur_top_p"
    ] = DEFAULT_TOP_P

if "centaur_use_top_k" not in st.session_state:
    st.session_state[
        "centaur_use_top_k"
    ] = DEFAULT_USE_TOP_K

if "centaur_top_k" not in st.session_state:
    st.session_state[
        "centaur_top_k"
    ] = DEFAULT_TOP_K


with st.expander(
    "Advanced model settings"
):
    st.caption(
        "These settings change how Centaur samples its "
        "answer distribution. For comparable experiments, "
        "keep them fixed across conditions and record the values used."
    )

    if st.button(
        "Reset to author-aligned defaults"
    ):
        st.session_state[
            "centaur_temperature"
        ] = DEFAULT_TEMPERATURE
        st.session_state[
            "centaur_top_p"
        ] = DEFAULT_TOP_P
        st.session_state[
            "centaur_use_top_k"
        ] = DEFAULT_USE_TOP_K
        st.session_state[
            "centaur_top_k"
        ] = DEFAULT_TOP_K
        st.rerun()

    s1, s2 = st.columns(2)

    with s1:
        temperature = float(
            st.number_input(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                step=0.1,
                key="centaur_temperature",
                help=(
                    "Higher values produce more random/diverse "
                    "sampling. Centaur's authors use 1.0 in "
                    "their minimal example."
                ),
            )
        )

    with s2:
        top_p = float(
            st.number_input(
                "Top-p",
                min_value=0.01,
                max_value=1.0,
                step=0.05,
                key="centaur_top_p",
                help=(
                    "Restricts sampling to the most probable "
                    "tokens whose cumulative probability reaches "
                    "this value. 1.0 applies no top-p truncation."
                ),
            )
        )

    use_top_k = st.checkbox(
        "Enable Top-k",
        key="centaur_use_top_k",
        help=(
            "Optional. Restricts sampling to the k most "
            "likely next tokens."
        ),
    )

    top_k = int(
        st.number_input(
            "Top-k",
            min_value=1,
            max_value=500,
            step=1,
            key="centaur_top_k",
            disabled=not use_top_k,
        )
    )

settings = {
    "temperature": temperature,
    "top_p": top_p,
    "use_top_k": use_top_k,
    "top_k": top_k if use_top_k else None,
}


st.subheader("Conditions")

if number_conditions == 1:
    default_allocations = [100]
elif number_conditions == 2:
    default_allocations = [50, 50]
else:
    default_allocations = [34, 33, 33]

conditions_input = []
default_names = [
    "Control",
    "Treatment A",
    "Treatment B",
]

for i in range(number_conditions):
    st.markdown(
        f"#### Condition {chr(65 + i)}"
    )

    c1, c2 = st.columns([4, 1])

    with c1:
        name = st.text_input(
            f"Condition {chr(65 + i)} name",
            value=(
                default_names[i]
                if number_conditions > 1
                else "Condition A"
            ),
            key=f"condition_name_{i}",
        )

        text = st.text_area(
            f"Condition {chr(65 + i)} content",
            placeholder=(
                "Enter exactly what this condition "
                "shows or tells the participant, e.g. 'Most people tend to pick option A'."
            ),
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
                key=(
                    f"allocation_"
                    f"{number_conditions}_{i}"
                ),
            )
        )

    conditions_input.append(
        {
            "name": name.strip(),
            "text": text.strip(),
            "allocation": allocation,
        }
    )

allocation_total = sum(
    condition["allocation"]
    for condition in conditions_input
)

allocation_counts = (
    largest_remainder_counts(
        total_simulations,
        [
            condition["allocation"]
            for condition in conditions_input
        ],
    )
    if allocation_total == 100
    else [0] * number_conditions
)

allocation_cols = st.columns(
    number_conditions
)

for i, condition in enumerate(
    conditions_input
):
    with allocation_cols[i]:
        st.metric(
            condition["name"]
            or f"Condition {chr(65 + i)}",
            f"{condition['allocation']}%",
            (
                f"{allocation_counts[i]:,} valid responses"
                if allocation_total == 100
                else None
            ),
        )

if allocation_total != 100:
    st.error(
        "Condition allocations currently total "
        f"{allocation_total}%. They must total exactly 100%."
    )


with st.expander(
    "Preview exact Centaur prompts"
):
    if (
        response_codes
        and instructions.strip()
        and all(
            condition["text"]
            for condition in conditions_input
        )
    ):
        prompt_tabs = st.tabs(
            [
                condition["name"]
                or f"Condition {chr(65 + i)}"
                for i, condition
                in enumerate(conditions_input)
            ]
        )

        for tab, condition in zip(
            prompt_tabs,
            conditions_input,
        ):
            with tab:
                st.code(
                    build_prompt(
                        instructions,
                        condition["text"],
                        response_codes,
                    ),
                    language=None,
                )
    else:
        st.caption(
            "Fill in the shared instructions, response codes, "
            "and all condition content to preview the exact "
            "prompt for each condition."
        )


def validate_inputs():
    errors = []

    if not experiment_name.strip():
        errors.append(
            "Add an experiment name."
        )

    if not instructions.strip():
        errors.append(
            "Add the shared experiment instructions."
        )

    if not response_codes:
        errors.append(
            "Add at least one possible response code."
        )

    if (
        len(
            {
                code.casefold()
                for code in response_codes
            }
        )
        != len(response_codes)
    ):
        errors.append(
            "Response codes must be unique."
        )

    if allocation_total != 100:
        errors.append(
            "Condition allocations must total exactly 100%."
        )

    names = [
        condition["name"]
        for condition in conditions_input
    ]

    if any(not name for name in names):
        errors.append(
            "Give every condition a name."
        )

    if (
        len(
            {
                name.casefold()
                for name in names
                if name
            }
        )
        != len(names)
    ):
        errors.append(
            "Condition names must be unique."
        )

    for i, condition in enumerate(
        conditions_input
    ):
        if not condition["text"]:
            errors.append(
                f"Add content for Condition "
                f"{chr(65 + i)}."
            )

        if condition["allocation"] <= 0:
            errors.append(
                f"Condition {chr(65 + i)} must "
                "receive more than 0% of simulations."
            )

        if (
            allocation_total == 100
            and allocation_counts[i] == 0
        ):
            errors.append(
                f"Condition {chr(65 + i)} receives "
                "0 valid responses after rounding. "
                "Increase total N or its allocation."
            )

    if total_simulations < number_conditions:
        errors.append(
            "Total valid simulation responses must be "
            "at least the number of active conditions."
        )

    return errors


def build_config():
    built_conditions = []

    for i, condition in enumerate(
        conditions_input
    ):
        built_conditions.append(
            {
                "name": condition["name"],
                "text": condition["text"],
                "allocation": condition[
                    "allocation"
                ],
                "count": allocation_counts[i],
                "prompt": build_prompt(
                    instructions,
                    condition["text"],
                    response_codes,
                ),
            }
        )

    return {
        "experiment_name": (
            experiment_name.strip()
        ),
        "instructions": instructions.strip(),
        "response_codes": response_codes,
        "total_simulations": total_simulations,
        "conditions": built_conditions,
        "settings": settings,
    }


review_clicked = st.button(
    "Review experiment & estimate cost",
    type="primary",
    width="stretch",
)

if review_clicked:
    errors = validate_inputs()

    if errors:
        for error in errors:
            st.error(error)

    else:
        config = build_config()
        fingerprint = config_fingerprint(
            config
        )

        with st.spinner(
            "Checking Featherless pricing "
            "and estimating the run..."
        ):
            runtime_info = get_runtime_info(
                api_key
            )
            estimate = estimate_run(
                api_key,
                config["conditions"],
                runtime_info,
            )

        st.session_state[
            "prepared_run"
        ] = {
            "fingerprint": fingerprint,
            "config": config,
            "runtime_info": runtime_info,
            "estimate": estimate,
        }


prepared = st.session_state.get(
    "prepared_run"
)

current_config = None
current_fingerprint = None

if not validate_inputs():
    current_config = build_config()
    current_fingerprint = (
        config_fingerprint(current_config)
    )

if (
    prepared
    and current_fingerprint
    != prepared["fingerprint"]
):
    st.info(
        "The experiment has changed since the last "
        "cost review. Click 'Review experiment & estimate cost' "
        "again before running."
    )


if (
    prepared
    and current_fingerprint
    == prepared["fingerprint"]
):
    estimate = prepared["estimate"]
    runtime_info = prepared[
        "runtime_info"
    ]

    st.subheader("Run plan")

    p1, p2, p3, p4 = st.columns(4)

    p1.metric(
        "Target valid responses",
        f"{prepared['config']['total_simulations']:,}",
    )

    p2.metric(
        "Estimated input tokens",
        f"{estimate['prompt_tokens']:,}",
    )

    p3.metric(
        "Estimated output tokens",
        f"{estimate['completion_tokens']:,}",
    )

    p4.metric(
        "Base estimated API cost",
        (
            f"~${estimate['base_cost']:,.4f}"
            if estimate["base_cost"]
            is not None
            else "Unavailable"
        ),
    )

    if estimate["base_cost"] is not None:
        st.caption(
            "The base estimate assumes every paid Centaur "
            "completion produces a valid response. Invalid "
            "responses are replaced, so actual cost can be higher. "
            "The approximate 3× safety-ceiling cost is "
            f"${estimate['safety_ceiling_cost']:,.4f}."
        )

    if estimate["used_fallback"]:
        st.warning(
            "Featherless returned an unusable tokenization "
            "result for at least one prompt, so the estimate "
            "used a character-based fallback instead of "
            "treating the prompt as zero tokens."
        )

    with st.expander(
        "See token estimate by condition"
    ):
        estimate_df = pd.DataFrame(
            estimate["condition_estimates"]
        ).rename(
            columns={
                "condition": "Condition",
                "tokens_per_prompt": (
                    "Estimated tokens / prompt"
                ),
                "target_valid_n": (
                    "Target valid N"
                ),
                "estimated_input_tokens": (
                    "Estimated input tokens"
                ),
                "token_source": (
                    "Token estimate source"
                ),
            }
        )

        st.dataframe(
            estimate_df,
            hide_index=True,
            width="stretch",
        )

    if runtime_info["error"]:
        st.warning(
            "Live Featherless model/plan information "
            "could not be retrieved. The experiment can "
            "still run, but live price calculation may be "
            "unavailable and requests will run sequentially."
        )

    require_confirmation = (
        prepared["config"][
            "total_simulations"
        ]
        > 500
    )

    confirmed = True

    if require_confirmation:
        confirmed = st.checkbox(
            "I confirm I want to run "
            f"{prepared['config']['total_simulations']:,} "
            "valid Centaur simulations and accept that "
            "retries can increase the final token cost.",
            key="large_run_confirmation_v2",
        )

    run_clicked = st.button(
        "Run experiment",
        type="primary",
        width="stretch",
        disabled=not confirmed,
    )

    if run_clicked:
        run_data = execute_experiment(
            prepared["config"],
            api_key,
            prepared["runtime_info"],
        )

        st.session_state[
            "last_run"
        ] = {
            "fingerprint": (
                prepared["fingerprint"]
            ),
            "config": prepared["config"],
            "run_data": run_data,
        }


last_run = st.session_state.get(
    "last_run"
)

if last_run:
    st.divider()
    st.header("Results")
    render_results(
        last_run["run_data"],
        last_run["config"],
    )