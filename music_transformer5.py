import os
from music21 import converter, instrument, note, chord
from transformers import GPT2LMHeadModel, GPT2Tokenizer


midi_folder = "./music"
notes = []
for file in os.listdir(midi_folder):
    midi = converter.parse(os.path.join(midi_folder, file))
    notes_to_parse = None
    try: 
        s2 = instrument.partitionByInstrument(midi)
        notes_to_parse = s2.parts[0].recurse() 
    except:
        notes_to_parse = midi.flat.notes
    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token 

batch_size = 100 
for i in range(0, len(notes), batch_size):
    batch_notes = notes[i:i+batch_size]
    inputs = tokenizer(batch_notes, padding='max_length', truncation=True, max_length=300, return_tensors='pt')
    
    model = GPT2LMHeadModel.from_pretrained('gpt2')
   
    output = model.generate(inputs.input_ids, max_new_tokens=100) # Adjust max_new_tokens as needed
  
    decoded_output = tokenizer.decode(output[0])

    print(decoded_output)
