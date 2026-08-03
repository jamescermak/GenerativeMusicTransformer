# Play Music with AI from your MIDI Keyboard
As a jazz musician, I thought the concept of 'trading 4s' with AI could be quite fun. Essentially, I play a bit on the piano, and the AI will try to improvise a line that continues off of what the user plays.
This model was trained on the PDMX dataset which is a collection of 250,000 MuseScore files - it was trained as a next token transformer - given a sequence of notes, predict the next notes that should come after it.

## Some Requirements
- To use this you will need a MIDI keyboard and also a SoundFont downloaded. I've downloaded this one by Chris Collins `https://github.com/mrbumpy409/GeneralUser-GS`.
- Once you have a SoundFont downloaded on your machine this needs to be updated as the `SF2_PATH` variable in `play_from_midi.py`
- To clone this repo you will need to recurse submodules to get access to the PDMX helper functions (these are mainly used in model training) run `git clone --recurse-submodules https://github.com/jamescermak/GenerativeMusicTransformer`


## Files Included
- `model_pipeline.ipynb`: This is the whole model training pipeline - you can see how data was processed, how features were extracted and embedded and are welcome to retrain the model on different features
- `music_helpers.py`: This file is all the helper functions that I created to be used across files. This includes turning note sequences to MIDI sequences as well as quantizing rhythm
- `music_transformer.pth` The saved weights of the model
- `music_transformer.py` The Model itself as well as its inference method
- `play_from_midi.py` **This is the app of this project. Run this with your MIDI keyboard to play music with AI**
- `requirements.txt` Requirements and dependencies for this project

## Additional Changes and Improvements to Make
- I would like to improve model output. Trying to estimate something musical can be very difficult. My next iteration would be more discrimination in gathering the training set; i.e. filtering by specific instruments/genres to have more specific output
- Bundling a .sf2 file with this app so users do not need to source their own SoundFont
- Probably the hardest problem this project faces is timing. Right now the user can set how many notes they wish to play (8, 16, 32, 64, etc.) And the model will give back the exact same amount of notes. I would like to add a feature that can auto detect when the user is done playing, and add more variety to the model output so it does not just copy the same number of notes
- Again with timing, adding a metronome feature or rhythm background music would likely be necessary. Without tempo being audible to the user, the user might play a quarter note, but the model may interpret this as a whole note. More interfacing between the AI and the user should be implemented so output is more referential of what the user plays.
