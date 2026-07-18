# SpellForge

I play a lot of D&D, so I wanted to see if a small language model could
write spells that actually sound like they came from the book: give it a
level, school, and a few tags, and get back a spell description in the
right register, or give it a description and get back a fitting name.

## How it works

`datasetup.ipynb` scrapes and cleans the 5e SRD spell list from 5etools
into `spells_master.csv`: name, description, level, school, casting time,
range, duration, damage type, and the rest of a spell's real attributes.

`t5finetune.ipynb` fine-tunes T5 as a multi-task model on that data. One
model learns two directions at once, tagged by task prefix:

- **name-to-description**: `describe spell: Fireball | school: Evocation |
  level: 3` produces the spell text.
- **attributes-to-name**: given a level, school, and a couple of tags,
  invent a name that fits.

Training uses a custom `Dataset` and a hand-written loop rather than the
Hugging Face `Trainer`, with a `find_max_batch_size` routine that
binary-searches the largest batch that fits in VRAM and then trains at a
fraction of that headroom. Generation quality is tracked with ROUGE, and
there's a small Gradio demo at the end for trying prompts interactively.

## Setup

```bash
pip install torch transformers pandas scikit-learn tqdm evaluate gradio
```

Run `datasetup.ipynb` first to produce `spells_master.csv` (or use the one
already in this repo), then `t5finetune.ipynb` to train and try the demo.
