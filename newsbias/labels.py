"""Label sets for the two fine-tuned axes. The order fixes the output index of each head."""

AXES = {
    "political": ("Left", "Center", "Right"),
    "gender": ("Misogynistic", "Misandrist", "Neutral"),
}

# The class that means "no bias found" on each axis.
NEUTRAL = {"political": "Center", "gender": "Neutral"}


def label_index(axis, label):
    return AXES[axis].index(label)
