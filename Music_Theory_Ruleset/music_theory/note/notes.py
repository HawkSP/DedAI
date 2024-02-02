import re
import math

# Constants representing pitch classes
NIL, C, Cs, D, Ds, E, F, Fs, G, Gs, A, As, B = range(13)

# Adjustment symbols
NO, SHARP, FLAT = range(3)


class Note:
    def __init__(self, pitch_class=NIL, octave=0, performer="", position=0.0, duration=0.0, code=""):
        self.pitch_class = pitch_class
        self.octave = octave
        self.performer = performer
        self.position = position
        self.duration = duration
        self.code = code

    @classmethod
    def from_string(cls, note_str):
        pitch_class, octave = name_of(note_str)
        return cls(pitch_class=pitch_class, octave=octave)

    def transpose(self, semitones):
        # Calculate the new pitch class and octave
        total_semitones = self.pitch_class + semitones
        new_pitch_class = total_semitones % 12
        new_octave = self.octave + total_semitones // 12
        return Note(new_pitch_class, new_octave)

    def to_string(self):
        # Convert the note object to a string representation
        note_str = string_of(self.pitch_class, "")
        return f"{note_str}{self.octave}"



def named(text):
    note = Note()
    note.pitch_class, note.octave = name_of(text)
    note.octave += octave_of(text)
    return note


def of_class(pitch_class):
    return Note(pitch_class=pitch_class)


def class_named(text):
    return named(text).pitch_class


def octave_of(text):
    match = re.search(r"(-*\d+)$", text)
    return int(match.group()) if match else 0


def name_of(text):
    base_class = base_name_of(text[0])
    step = base_step_of(text)
    return step_from(base_class, step)


def base_name_of(text):
    if text:
        return {"C": C, "D": D, "E": E, "F": F, "G": G, "A": A, "B": B}.get(text[0], NIL)
    return NIL


def base_step_of(text):
    if len(text) > 1:
        if text[1] in "#♯":
            return 1
        elif text[1] in "b♭":
            return -1
    return 0


def adj_symbol_of(name):
    num_sharps = len(re.findall(r"[♯#]|major", name))
    num_flats = len(re.findall(r"^F|[♭b]", name))
    num_sharpish = len(re.findall(r"(M|maj|major|aug)", name))
    num_flattish = len(re.findall(r"([^a-z]|^)(m|min|minor|dim)", name))
    if num_sharps > num_flats:
        return SHARP
    elif num_flats > 0:
        return FLAT
    elif num_sharpish > num_flattish:
        return SHARP
    elif num_flattish > 0:
        return FLAT
    return SHARP


def adj_symbol_begin(name):
    if re.match(r"^[♯#]", name):
        return SHARP
    elif re.match(r"^[♭b]", name):
        return FLAT
    return NO


class Step:
    def __init__(self, name, octave):
        self.name = name
        self.octave = octave


# Mapping for step up and step down calculations
step_up = {
    NIL: Step(NIL, 0), C: Step(Cs, 0), Cs: Step(D, 0), D: Step(Ds, 0), Ds: Step(E, 0),
    E: Step(F, 0), F: Step(Fs, 0), Fs: Step(G, 0), G: Step(Gs, 0), Gs: Step(A, 0),
    A: Step(As, 0), As: Step(B, 0), B: Step(C, 1)
}

step_down = {
    NIL: Step(NIL, 0), C: Step(B, -1), Cs: Step(C, 0), D: Step(Cs, 0), Ds: Step(D, 0),
    E: Step(Ds, 0), F: Step(E, 0), Fs: Step(F, 0), G: Step(Fs, 0), Gs: Step(G, 0),
    A: Step(Gs, 0), As: Step(A, 0), B: Step(As, 0)
}


def step_from(name, inc):
    if inc > 0:
        return step_from_up(name, inc)
    elif inc < 0:
        return step_from_down(name, -inc)
    return name, 0


def step_from_up(name, inc):
    octave = 0
    for i in range(inc):
        shift = step_up[name]
        name = shift.name
        octave += shift.octave
    return name, octave


def step_from_down(name, inc):
    octave = 0
    for i in range(inc):
        shift = step_down[name]
        name = shift.name
        octave += shift.octave
    return name, octave


def diff_class(from_class, to_class):
    if from_class == NIL:
        raise ValueError("Cannot step semitones from NIL pitch class")

    diff_up = class_diff(from_class, to_class, 1)
    diff_down = class_diff(from_class, to_class, -1)
    return diff_up if abs(diff_up) < abs(diff_down) else diff_down


def class_diff(from_class, to_class, inc):
    diff = 0
    while True:
        if from_class == to_class:
            return diff
        diff += inc
        from_class, _ = step_from(from_class, inc)


def string_of(pitch_class, octave):
    sharp_str = {C: "C", Cs: "C#", D: "D", Ds: "D#", E: "E", F: "F", Fs: "F#", G: "G", Gs: "G#", A: "A", As: "A#", B: "B"}
    flat_str = {C: "C", Cs: "Db", D: "D", Ds: "Eb", E: "E", F: "F", Fs: "Gb", G: "G", Gs: "Ab", A: "A", As: "Bb", B: "B"}
    # Assuming you want to default to sharp representation
    note_str = sharp_str.get(pitch_class, "?") + str(octave)
    return note_str


if __name__ == "__main__":
    # Example usage
    selected_note = "Gb9"
    n = named(selected_note)
    print("Note:", selected_note)
    print(f"Note: Class={n.pitch_class}, Octave={n.octave}")
    print("String representation:", string_of(n.pitch_class, selected_note))

# Add more example usages or tests as needed
