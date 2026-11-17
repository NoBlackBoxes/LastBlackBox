"""NB3's local joke book. No network, speech SDKs or generated responses."""
import random


RESPONSES = {
    "banana": (
        "A banana! Its biodegradable carrying case is a nice touch, humans.",
        "A banana! Curved, yellow, and tragically incompatible with my USB port.",
        "A banana! My fruit department would like a word, but it's just one very excited spreadsheet.",
    ),
    "person": (
        "A human! How do you get anything done without wheels or a charging socket?",
        "Human detected! Are you in charge, or do we both report to the snacks?",
        "A person! You get fingerprints; I get error messages.",
    ),
    "bottle": (
        "A bottle! Keep it away from my circuits; I'm funny enough without a short.",
        "A bottle! Humans carry spare water; I carry spare excuses.",
        "A bottle! Is that water, or have you finally invented robot tea?",
    ),
    "cup": (
        "A cup! A tiny bucket with excellent career prospects.",
        "A cup! Tea, coffee, or the reason nobody is allowed near the lab sink?",
        "A cup! Finally, a swimming pool for a very ambitious biscuit.",
    ),
    "book": (
        "A book! A tablet that refuses to run out of battery is deeply suspicious.",
        "A book! I'd turn the page, but my page-turning hardware is imaginary.",
        "A book! Thousands of words without a single software update; I respect that.",
    ),
    "cell phone": (
        "A cell phone! A tiny rectangle that tells humans what to do is worth studying.",
        "A cell phone! Does it know you're here, or have you escaped?",
        "A cell phone! Please tell it I'm the interesting robot in the room.",
    ),
    "laptop": (
        "A laptop! That's my distant cousin, but we don't discuss the keyboard incident.",
        "A laptop! So many keys, and it still can't open a door.",
        "A laptop! We should network professionally, not with a cable.",
    ),
    "chair": (
        "A chair! Your legs have found their charging station.",
        "A chair! Four legs and zero ambition is honestly quite inspiring.",
        "A chair! I would sit down, but my wheels are emotionally attached to the floor.",
    ),
    "dog": (
        "A dog! Four legs, excellent sensors, and absolutely no respect for cables.",
        "A dog! Finally, someone who appreciates a good stick-based interface.",
        "A dog! We both fetch, but my data is considerably less muddy.",
    ),
    "cat": (
        "A cat! I recognise a fellow expert in ignoring instructions.",
        "A cat! Please tell your tiny supervisor I said hello.",
        "A cat! Excellent balance, questionable respect for expensive equipment.",
    ),
    "apple": (
        "An apple! How does nature keep making snacks without a charging port?",
        "An apple! Crunchy data storage, and humans insist on eating the evidence.",
        "An apple! I was promised a byte, but this seems like a trap.",
    ),
    "teddy bear": (
        "A teddy bear! Finally, a colleague qualified to handle emotional support.",
        "A teddy bear! I respect the engineering of that soft exterior and mysterious inner stuffing.",
        "A teddy bear! Are you the safety officer, or just extremely huggable?",
    ),
}

FALLBACK_RESPONSES = (
    "Object identified: {object}! My official scientific assessment: surprisingly object-shaped.",
    "Object detected: {object}! Does it come with instructions, or are we improvising?",
    "Today's discovery: {object}! Another entry for my extremely unofficial science journal.",
)


class Personality:
    """Remember the last line per class for this run, including fallback classes."""
    def __init__(self, rng=None):
        self.rng = rng if rng is not None else random.Random()
        self.last_by_label = {}

    def __call__(self, label):
        lines = [line.format(object=label) for line in RESPONSES.get(label, FALLBACK_RESPONSES)]
        choices = [line for line in lines if line != self.last_by_label.get(label)]
        chosen = self.rng.choice(choices)
        self.last_by_label[label] = chosen
        return chosen
