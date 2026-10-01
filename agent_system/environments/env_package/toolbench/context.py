"""Budget real conversation messages without replaying hidden or malformed reasoning."""

from .projection import parse_tool_action
from .stabletoolbench import parse_arguments, StableToolBenchError

INVALID_HISTORY_ACTION = "[Invalid action omitted; no tool was executed.]"


def history_action(action, valid):
    """Only executed action/arguments belong in subsequent model context."""
    if not valid:
        return INVALID_HISTORY_ACTION
    name, arguments = parse_tool_action(action)
    try:
        parse_arguments(arguments)
    except StableToolBenchError:
        return INVALID_HISTORY_ACTION
    return f"Action: {name}\nAction Input: {arguments}" if name else INVALID_HISTORY_ACTION


def format_context(initial, history):
    """Readable diagnostic rendering; model inputs use build_messages instead."""
    if not history:
        return initial
    steps = "\n\n".join(
        f"Previous action:\n{item['action']}\nObservation:\n{item['observation']}"
        for item in history
    )
    return f"{initial}\n\nInteraction history:\n{steps}\n\nContinue with exactly one Action/Action Input."


def build_messages(initial, history):
    messages = [{"role": "user", "content": initial}]
    for item in history:
        messages.append({"role": "assistant", "content": item["action"]})
        messages.append({"role": "user", "content": item["observation"]})
    return messages


def _fit(initial, history, measure, render, max_tokens):
    if measure(render(initial, [])) > max_tokens:
        raise ValueError(
            "ToolBench task and tool definitions exceed max_prompt_length; "
            "filter this task or increase the prompt budget instead of truncating the task."
        )
    history = [dict(item) for item in history]
    while len(history) > 1 and measure(render(initial, history)) > max_tokens:
        history.pop(0)
    result = render(initial, history)
    if measure(result) <= max_tokens:
        return result
    latest = history[0]
    marker = "\n[truncated to fit the prompt budget]\n"

    def shortened(count):
        item = {}
        for key in ("action", "observation"):
            value = latest[key]
            item[key] = value if len(value) <= count else value[:count // 2] + marker + value[-(count-count // 2):] if count else marker
        return render(initial, [item])

    low, high = 0, max(len(latest["action"]), len(latest["observation"]))
    best = None
    while low <= high:
        middle = (low + high) // 2
        candidate = shortened(middle)
        if measure(candidate) <= max_tokens:
            best, low = candidate, middle + 1
        else:
            high = middle - 1
    if best is None:
        raise ValueError("ToolBench prompt budget leaves no room for even a shortened latest observation.")
    return best


def fit_messages(initial, history, tokenizer, max_tokens, template_kwargs=None):
    def size(messages):
        rendered = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, **dict(template_kwargs or {}),
        )
        return len(tokenizer.encode(rendered, add_special_tokens=False))
    return _fit(initial, history, size, build_messages, max_tokens)


def fit_context(initial, history, tokenizer, max_tokens, template_kwargs=None):
    """Compatibility for diagnostics that still render a single text context."""
    def size(text):
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True,
            tokenize=False, **dict(template_kwargs or {}),
        )
        return len(tokenizer.encode(rendered, add_special_tokens=False))
    return _fit(initial, history, size, format_context, max_tokens)
