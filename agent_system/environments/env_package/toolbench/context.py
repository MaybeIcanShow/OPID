"""Keep task and tool definitions intact while trimming old interaction history."""


def format_context(initial, history):
    if not history:
        return initial
    steps = "\n\n".join(
        f"Previous action:\n{item['action']}\nObservation:\n{item['observation']}"
        for item in history
    )
    return f"{initial}\n\nInteraction history:\n{steps}\n\nContinue with one Thought/Action/Action Input step."


def fit_context(initial, history, tokenizer, max_tokens, template_kwargs=None):
    """Drop oldest complete turns first; shorten only the newest turn if necessary."""
    template_kwargs = dict(template_kwargs or {})

    def size(text):
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True,
            tokenize=False, **template_kwargs,
        )
        return len(tokenizer.encode(rendered, add_special_tokens=False))

    if size(initial) > max_tokens:
        raise ValueError(
            "ToolBench task and tool definitions exceed max_prompt_length; "
            "filter this task or increase the prompt budget instead of truncating the task."
        )
    history = [dict(item) for item in history]
    while len(history) > 1 and size(format_context(initial, history)) > max_tokens:
        history.pop(0)
    result = format_context(initial, history)
    if size(result) <= max_tokens:
        return result
    # Preserve the task. Retain both ends of the latest action/response with an explicit marker.
    latest = history[0]
    marker = "\n[truncated to fit the prompt budget]\n"
    def shortened(count):
        item = {}
        for key in ("action", "observation"):
            value = latest[key]
            item[key] = value if len(value) <= count else value[:count // 2] + marker + value[-(count-count // 2):] if count else marker
        return format_context(initial, [item])
    low, high = 0, max(len(latest["action"]), len(latest["observation"]))
    best = None
    while low <= high:
        middle = (low + high) // 2
        candidate = shortened(middle)
        if size(candidate) <= max_tokens:
            best, low = candidate, middle + 1
        else:
            high = middle - 1
    if best is None:
        raise ValueError("ToolBench prompt budget leaves no room for even a shortened latest observation.")
    return best
