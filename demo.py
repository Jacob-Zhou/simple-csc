import time
import re
import torch
import streamlit as st

from lmcsc import LMCorrector

import yaml

escape_dict = {
    "!": "\!",
    '"': '"',
    "#": "\#",
    "$": "\$",
    "%": "\%",
    "&": "\&",
    "'": "'",
    "(": "\(",
    ")": "\)",
    "*": "\*",
    "+": "\+",
    ",": "\,",
    "-": "\-",
    ".": "\.",
    "/": "\/",
    ":": "\:",
    ";": "\;",
    "<": "\<",
    "=": "\=",
    ">": "\>",
    "?": "\?",
    "@": "\@",
    "[": "\[",
    "\\": "\\\\",
    "]": "\]",
    "^": "\^",
    "_": "\_",
    "`": "\`",
    "{": "\{",
    "|": "\|",
    "}": "\}",
    "~": "\~",
    "\n": "\n\n",
}

oom_error = False

reSPLIT = re.compile(
    "(?:(?![。！？!?，,])(?<=[。！？!?，,])(?![!！?？”】]))|(?<=(?:[。！？!?，,])[”】])|(?<=[\n\r])"
)


def preallocate(lmcsc_model):
    batch = ["中" *  128]
    (
        model_kwargs,
        context_input_ids,
        context_attention_mask,
        beam_scorer,
        observed_sequence_generator,
    ) = lmcsc_model.preprocess(batch)
    try:
        lmcsc_model.model.distortion_guided_beam_search(
            observed_sequence_generator,
            input_ids=context_input_ids,
            attention_mask=context_attention_mask,
            beam_scorer=beam_scorer,
            **model_kwargs,
        )
    except RuntimeError:
        pass


def excepthook(args, streamer):
    global oom_error
    exc_type, exc_value, *args = args
    streamer.end()
    if issubclass(exc_type, RuntimeError):
        oom_error = True
    else:
        raise exc_value


def split_text(text, max_length=256):
    sentences = reSPLIT.split(text)

    # Split sentences that still exceed max_length
    sentence_num = len(sentences)
    new_sentences = []
    for i in range(sentence_num):
        if len(sentences[i]) > max_length:
            new_sentences.extend([
                sentences[i][j : j + max_length]
                for j in range(0, len(sentences[i]), max_length)
            ])
        else:
            new_sentences.append(sentences[i])
    sentences = new_sentences

    # Merge sentences that are too short
    new_sentences = []
    new_sentence = ""
    for sentence in sentences:
        if len(new_sentence) + len(sentence) > max_length:
            new_sentences.append(new_sentence)
            new_sentence = ""
        new_sentence += sentence
    if new_sentence != "":
        new_sentences.append(new_sentence)
    return new_sentences

def correct_sentences(obversed_text, prompt, lmcsc_model):
    if lmcsc_model is None:
        raise ValueError("LMCSCModel is not initialized")

    start_time = time.time()

    status_placeholder = st.empty()
    result_placeholder = st.empty()

    with status_placeholder:
        st.status("纠错中...")

    global oom_error
    oom_error = False

    sentences = split_text(
        obversed_text,
        max_length=st.session_state.config["context_window"]["chunk_size"],
    )
    n_predictioned = 0
    predictions = []
    for sentence in sentences:
        if oom_error:
            break
        # the maximum length of the context is window_size * chunk_size
        context = "".join(
            predictions[-st.session_state.config["context_window"]["window_size"] :]
        )
        prompt_context = prompt + context
        # torch.cuda.empty_cache()

        with result_placeholder:
            stream_preds = lmcsc_model(sentence, prompt_context, stream=True)
            for new_text in stream_preds:
                if isinstance(new_text, Exception):
                    if isinstance(new_text, torch.cuda.OutOfMemoryError):
                        oom_error = True
                        break
                    else:
                        raise new_text
                output = new_text[0][0]
                predicted_text = "".join(predictions) + output
                n_predictioned = len(predicted_text)
                # decorate output, different as red color
                printable_text = ""
                continue_mistake = ["", ""]
                for o, s in zip(predicted_text, obversed_text):
                    printable_o = escape_dict.get(o, o)
                    printable_s = escape_dict.get(s, s)
                    if o == s:
                        if (continue_mistake[0] + continue_mistake[1]) != "":
                            printable_text += (
                                f":red[~~{continue_mistake[1]}~~]:green[**{continue_mistake[0]}**]"
                            )
                            continue_mistake = ["", ""]
                        printable_text += printable_o
                    else:
                        continue_mistake[0] += printable_o
                        continue_mistake[1] += printable_s
                if (continue_mistake[0] + continue_mistake[1]) != "":
                    printable_text += (
                        f":red[~~{continue_mistake[1]}~~]:green[**{continue_mistake[0]}**]"
                    )
                if len(predicted_text) != len(obversed_text):
                    for s in obversed_text[len(predicted_text) :]:
                        printable_s = escape_dict.get(s, s)
                        if s == "\n":
                            printable_text += printable_s
                        else:
                            printable_text += f":orange[{printable_s}]"
                st.write(printable_text)
            predictions.append(output)
    time_cost = time.time() - start_time
    with status_placeholder:
        if oom_error:
            st.error(
                f"纠错失败，显存不足 (已完成 {n_predictioned}/{len(obversed_text)})",
                icon="❌",
            )
        else:
            st.success(f"纠错完成 ({time_cost:.2f}s)", icon="✅")


@st.cache_resource(show_spinner="模型加载中...")
def load_model(selected_model):
    lmcsc_model = LMCorrector(
        selected_model,
        config_path=st.session_state["config_path"],
        n_beam=st.session_state["default_n_beam"],
        n_beam_hyps_to_keep=st.session_state["default_n_beam_hyps_to_keep"],
        alpha=st.session_state["default_alpha"],
        n_observed_chars=st.session_state["default_n_observed_chars"],
        distortion_model_smoothing=st.session_state["default_distortion_model_smoothing"],
        use_faithfulness_reward=st.session_state["default_use_faithfulness_reward"],
        max_length=st.session_state["default_max_length"],
    )

    if st.session_state.config["preallocate_memory"]:
        preallocate(lmcsc_model)

    return lmcsc_model


def update_params(selected_model, n_beam, alpha, use_faithfulness_reward):
    lmcsc_model = load_model(selected_model)
    lmcsc_model.update_params(
        n_beam=n_beam,
        alpha=alpha,
        use_faithfulness_reward=use_faithfulness_reward,
    )
    lmcsc_model.print_params()
    return lmcsc_model


def example_format_func(example):
    if len(example) < 512:
        return f"{example[:10]}... ({len(example)} chars)"
    else:
        return f"Long Example ({len(example)} chars)"


config = yaml.safe_load(open("configs/demo_app_config.yaml", "r", encoding="utf-8"))

if "config" not in st.session_state:
    st.session_state.config = config

for key, value in config["default_parameters"].items():
    if key not in st.session_state:
        st.session_state[key] = value

# App title
st.set_page_config(page_title="Simple CSC")

st.title("Simple CSC")

if "obversed_text" not in st.session_state:
    st.session_state.obversed_text = st.session_state["default_obversed_text"]

if "prompt" not in st.session_state:
    st.session_state.prompt = "\n"

with st.sidebar:
    st.sidebar.subheader("Models")
    model_families = list(st.session_state.config["model_families"].keys())
    model_family = st.sidebar.selectbox(
        "LLM familys",
        model_families,
        index=model_families.index(st.session_state["default_model_family"]),
        key="model_family",
    )

    models = st.session_state.config["model_families"][model_family]
    if st.session_state["default_model"] not in models:
        st.session_state["default_model"] = models[0]
    selected_model = st.sidebar.selectbox(
        "LLMs",
        models,
        index=models.index(st.session_state["default_model"]),
        key="selected_model",
    )
    st.sidebar.divider()
    st.sidebar.subheader("Parameters")
    n_beam = st.sidebar.select_slider(
        st.session_state.config["parameter_weights"]["n_beam"]["text"],
        options=st.session_state.config["parameter_weights"]["n_beam"]["options"],
        value=st.session_state["default_n_beam"],
    )
    alpha = st.sidebar.slider(
        st.session_state.config["parameter_weights"]["alpha"]["text"],
        min_value=st.session_state.config["parameter_weights"]["alpha"]["min"],
        max_value=st.session_state.config["parameter_weights"]["alpha"]["max"],
        value=st.session_state["default_alpha"],
        step=st.session_state.config["parameter_weights"]["alpha"]["step"],
    )
    use_faithfulness_reward = st.sidebar.checkbox(
        st.session_state.config["parameter_weights"]["use_faithfulness_reward"][
            "text"
        ],
        value=st.session_state["default_use_faithfulness_reward"],
    )

    st.sidebar.divider()
    st.sidebar.subheader("Examples")
    example_text = st.sidebar.selectbox(
        "Choose an example",
        [st.session_state.config["long_example"]] + st.session_state.config["examples"],
        index=None,
        format_func=example_format_func,
    )
    if example_text is not None:
        st.session_state.obversed_text = example_text

lmcsc_model = update_params(
    selected_model, n_beam, alpha, use_faithfulness_reward
)

customised_prompt = st.toggle(
    "Use prompt (*By default, a prompt is unnecessary*)", value=False
)
if customised_prompt:
    prompt = st.text_area(
        "Enter prompt",
        value=st.session_state.prompt,
        placeholder="By default, a prompt is unnecessary",
    )

    if prompt != "":
        st.write("Prompt:")
        st.code(repr(prompt), language="python")
        st.session_state.prompt = prompt
else:
    prompt = ""
    st.session_state.prompt = ""

obversed_text = st.text_area(
    "Enter text to correct",
    value=st.session_state.obversed_text,
    height=200,
)

if obversed_text != st.session_state.obversed_text:
    st.session_state.obversed_text = obversed_text

button = st.button("开始纠错")

st.divider()

st.subheader("纠错结果")
if button:
    with st.container():
        correct_sentences(st.session_state.obversed_text, prompt, lmcsc_model)
