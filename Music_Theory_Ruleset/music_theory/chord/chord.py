# chords.py

import re
from functools import lru_cache
from Music_Theory_Ruleset.music_theory.note.notes import Note

# Constants representing intervals in semitones
INTERVALS = {
	'unison': 0,
	'minor_second': 1,
	'major_second': 2,
	'minor_third': 3,
	'major_third': 4,
	'perfect_fourth': 5,
	'diminished_fifth': 6,
	'perfect_fifth': 7,
	'augmented_fifth': 8,
	'minor_sixth': 8,
	'major_sixth': 9,
	'minor_seventh': 10,
	'major_seventh': 11,
	'octave': 12,
	'minor_ninth': 13,
	'major_ninth': 14,
	'augmented_ninth': 15,
	'perfect_eleventh': 17,
	'augmented_eleventh': 18,
	'perfect_thirteenth': 21,
	# Add more intervals as needed
}

# Chord extension patterns
EXTENSIONS = {
	'6': 'major_sixth',
	'7': 'minor_seventh',
	'maj7': 'major_seventh',
	'9': 'major_ninth',
	'#9': 'augmented_ninth',
	'11': 'perfect_eleventh',
	'#11': 'augmented_eleventh',
	'13': 'perfect_thirteenth',
	# Add more extensions as needed
}


# Function to parse additional chord extensions
def parse_extensions(additional_str):
	intervals = []
	for ext, interval in EXTENSIONS.items():
		if ext in additional_str:
			intervals.append(interval)
	return intervals


class Chord:
	def __init__(self, root, intervals, bass_note=None):
		self.root = root
		print("Starting Chord Initialization")
		print(self.root)
		self.intervals = intervals
		print("Intervals Complete")
		print(self.intervals)
		print("Bass Note")
		self.bass_note = bass_note
		print("Bass Note Complete")
		print(self.bass_note)
		print("Notes")
		self.notes = self.calculate_notes()
		print("Notes Complete")
		print(self.notes)
		print("Octave Normalization")
		self.octave_normalization()
		print("Octave Normalization Complete")

	@lru_cache(maxsize=None)
	def transpose_note(self, note, semitones):
		# This method is cached to avoid redundant transpositions
		return note.transpose(semitones)

	def calculate_notes(self):
		# Start with the root note
		notes = [self.root]
		# Add the intervals above the root note
		for interval in self.intervals[1:]:  # Skip 'unison'
			interval_semitones = INTERVALS[interval]
			next_note = self.root.transpose(interval_semitones)
			# Wrap to the next octave if necessary
			if next_note.pitch_class <= self.root.pitch_class:
				next_note = next_note.transpose(12)  # Move to the next octave
			notes.append(next_note)
		print("Calculate Notes Complete")
		return [self.transpose_note(self.root, INTERVALS[interval]) for interval in self.intervals]

	def octave_normalization(self):
		# Normalize octaves so that chord notes are within the 3rd to 5th octaves
		for i, note in enumerate(self.notes):
			while note.octave < 3:
				self.notes[i] = self.transpose_note(note, 12)
				note = self.notes[i]
			while note.octave > 5:
				self.notes[i] = self.transpose_note(note, -12)
				note = self.notes[i]

		# Ensure the bass note is below the lowest chord note but not lower than the 2nd octave
		if self.bass_note:
			lowest_chord_note_octave = min(note.octave for note in self.notes)
			while self.bass_note.octave >= lowest_chord_note_octave:
				self.bass_note = self.transpose_note(self.bass_note, -12)
			while self.bass_note.octave < 2:
				self.bass_note = self.transpose_note(self.bass_note, 12)

		print("Octave Normalization Complete")

	def transpose(self, semitones):
		# Transpose the chord by a number of semitones
		transposed_root = self.transpose_note(self.root, semitones)
		print("Transposed Root")
		return Chord(transposed_root, self.intervals, self.bass_note)

	def voice_chord(self):
		# Sort the chord notes by pitch class and octave
		sorted_notes = sorted(self.notes, key=lambda note: (note.pitch_class, note.octave))

		# Start voicing with the bass note if it exists
		voiced_notes = [self.bass_note] if self.bass_note else []

		# Attempt to distribute the notes in a practical range
		for note in sorted_notes:
			if voiced_notes:
				# Ensure the note is voiced above the previous note
				while note.octave <= voiced_notes[-1].octave and note.pitch_class <= voiced_notes[-1].pitch_class:
					note = note.transpose(12)  # Move up an octave
			voiced_notes.append(note)

		# Check that no adjacent notes are identical after voicing
		for i in range(1, len(voiced_notes)):
			if voiced_notes[i].pitch_class == voiced_notes[i - 1].pitch_class and \
					voiced_notes[i].octave == voiced_notes[i - 1].octave:
				voiced_notes[i] = voiced_notes[i].transpose(12)  # Move up an octave

		return voiced_notes

def parse_chord_name(chord_name):
	# Split the name into root note part and mode part
	match = re.match(r"([A-G][#b]?)(m?)([^/]*)/?(.*)", chord_name)
	if not match:
		raise ValueError(f"Invalid chord name: {chord_name}")

	root_note_str, minor_str, ext_str, bass_note_str = match.groups()
	root_note = Note.from_string(root_note_str)
	intervals = ['unison']

	# Determine the chord type based on the name components
	intervals.append('minor_third' if minor_str else 'major_third')
	intervals += parse_extensions(ext_str)

	# Handle bass note if slash chord
	bass_note = Note.from_string(bass_note_str) if bass_note_str else None

	# Adjust the root note to a reasonable default octave if necessary
	root_note_octave = 4  # Middle C octave
	if not bass_note or bass_note.octave >= root_note_octave:
		root_note = root_note.transpose(12 * (root_note_octave - root_note.octave))

	# Only return the Chord object
	print("Parse Chord Name Complete")
	return Chord(root_note, intervals, bass_note)


# Example usage
if __name__ == "__main__":
	print("Starting")
	chord = parse_chord_name("Cm9/G")
	print("Chord Notes:", [note.to_string() for note in chord.notes])
	if chord.bass_note:
		print("Bass Note:", chord.bass_note.to_string())

	voiced_chord_notes = chord.voice_chord()
	print("Voiced Chord Notes:", [note.to_string() for note in voiced_chord_notes])

	print("Code Complete")