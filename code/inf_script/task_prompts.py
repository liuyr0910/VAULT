"""Inference-only task prompts; annotation dictionaries are never read.

Observation metadata (RGB timestamps/ROI and event windows) is supplied by the
visual pipeline separately. Training prompts are defined separately in scripts/build_sft.py.
"""

CATEGORY_LIST = [
    "traffic accident",
    "building collapse",
    "fighting",
    "hijack",
    "environmental pollution",
    "fire",
    "burglary",
    "marine accident",
    "vandalism",
    "facility malfunction",
    "illegal hunting",
    "robbery",
    "riot",
    "shooting",
    "natural hazard",
    "normal",
    "theft",
    "traffic violation",
    "object falling",
    "people falling",
    "explosion",
    "abuse",
    "assault",
    "arrest",
    "drowning",
    "animal aggression",
]
DESCRIPTION_PROMPT = "Please give a concise description of the anomaly in the video. If there is no anomaly, just describe the normal event."
CLASSIFICATION_PROMPT = (
    "Classify the video content into one or multiple of the following categories:\n["
    + ", ".join((f'"{category}"' for category in CATEGORY_LIST))
    + "]\nRespond with the category name(s) only, separated by commas if multiple."
)
TEMPORAL_PROMPT = "Identify the start and end timestamps (in seconds) for all distinct anomalous events in the video. If multiple anomalies occur, separate the time intervals with commas. Format strictly as numbers: start_sec - end_sec, start_sec - end_sec. If the video contains no anomalies (is normal), output -1 - -1."
VQA_TEMPLATE = "Question: {question}\nOptions:\n{options}\nRespond with the correct option letter (e.g., A)."
TASK_PROMPT_POLICY = "eventvault_independent_tasks"


def build_task_prompt(task_type, *, question=None, options=None):
    """Accept only public task inputs, never GT, prior predictions or history."""
    if task_type == "d":
        return DESCRIPTION_PROMPT
    if task_type == "c":
        return CLASSIFICATION_PROMPT
    if task_type == "t":
        return TEMPORAL_PROMPT
    return VQA_TEMPLATE.format(
        question=question,
        options="\n".join(
            (f"{key}: {value}" for key, value in sorted(options.items()))
        ),
    )


class PromptFactory:
    """Compatibility boundary: ignore data/data_item, whitelist VQA fields."""

    @staticmethod
    def task_prompt(task_type, data=None, vqa_item=None):
        if task_type == "vqa":
            return build_task_prompt(
                task_type,
                question=vqa_item.get("question"),
                options=vqa_item.get("options"),
            )
        return build_task_prompt(task_type)

    @staticmethod
    def get_prompt(task_type, data_item=None, vqa_item=None):
        return PromptFactory.task_prompt(task_type, vqa_item=vqa_item)
