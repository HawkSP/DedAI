from music_theory.note.notes import Note
from music_theory.scale.scale import Scale

# Example usage
if __name__ == "__main__":
    # Create a C major scale instance
    c_major_scale = Scale(Note.from_string("C"), "major")
    print("C Major Scale Notes:", [note.to_string() for note in c_major_scale.notes()])
