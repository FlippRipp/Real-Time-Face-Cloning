# Characters

Put your character images in this folder. `run.bat` (with no arguments)
lists every `.png` here and lets you pick one by number.

Image requirements (THA3 spec):

- 512×512 PNG with an alpha channel (RGBA), transparent background
- forward-facing anime-style character
- head roughly in the upper half of the frame

Arbitrary character art (wrong size, opaque background, off-center) can be
converted with the preparation wizard — `run.bat` menu option 5, or:

    python -m tha_wrapper.prepare path\to\art.png

It does AI background removal, places the head in THA3's target box, and
lets you supervise/adjust every step before anything is saved here.

See the main README for details on preparing character art.
