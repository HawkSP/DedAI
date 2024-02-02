import re
from Music_Theory_Ruleset.music_theory.note.notes import Note


class Scale:
    def __init__(self, root, mode):
        self.root = root  # Root note as a Note object
        self.mode = mode  # Mode name as a string
        self.tones = self.calculate_tones()

    def calculate_tones(self):
        # Logic to calculate tones based on the root and mode
        intervals = MODES[self.mode]
        tones = [self.root]
        for interval in intervals:
            tones.append(tones[-1].step(interval))
        return tones

    def notes(self):
        return self.tones


def parse_scale_name(name):
    # Split the name into root note part and mode part
    match = re.match(r"([A-G][#b]?)(.*)", name)
    root_note_str, mode_str = match.groups()
    root_note = Note.from_string(root_note_str)
    mode = mode_str.strip().lower() if mode_str.strip() else 'major'  # Default to major
    return root_note, mode


# Scale modes defined by their intervals
MODES = {
    'major': [2, 2, 1, 2, 2, 2, 1],  # Also known as Ionian
    'minor': [2, 1, 2, 2, 1, 2, 2],  # Also known as Aeolian
    'dorian': [2, 1, 2, 2, 2, 1, 2],
    'phrygian': [1, 2, 2, 2, 1, 2, 2],
    'lydian': [2, 2, 2, 1, 2, 2, 1],
    'mixolydian': [2, 2, 1, 2, 2, 1, 2],
    'locrian': [1, 2, 2, 1, 2, 2, 2],
    'melodic_minor': [2, 1, 2, 2, 2, 2, 1],
    'harmonic_minor': [2, 1, 2, 2, 1, 3, 1],
    'harmonic_major': [2, 2, 1, 2, 1, 3, 1],
    'double_harmonic': [1, 3, 1, 2, 1, 3, 1],
    'enigmatic': [1, 3, 2, 2, 2, 1, 1],
    'neapolitan_minor': [1, 2, 2, 2, 1, 3, 1],
    'neapolitan_major': [1, 2, 2, 2, 2, 2, 1],
    'hungarian_minor': [2, 1, 3, 1, 1, 3, 1],
    'hungarian_major': [3, 1, 2, 1, 2, 1, 2],
    'romanian_minor': [2, 1, 3, 1, 2, 1, 2],
    'ukrainian_dorian': [2, 1, 3, 1, 2, 1, 2],
    'altered': [1, 2, 1, 2, 2, 2, 2],
    'whole_tone': [2, 2, 2, 2, 2, 2],
    'pentatonic_major': [2, 2, 3, 2, 3],
    'pentatonic_minor': [3, 2, 2, 3, 2],
    'blues': [3, 2, 1, 1, 3, 2]
}


def create_scale(name):
    root, mode = parse_scale_name(name)
    return Scale(root, mode)


# Example usage
scale = create_scale("C major")
print("Scale:", scale.notes())
