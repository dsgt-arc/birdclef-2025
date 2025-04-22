# acmiyaguchi

Here's some scratch work to prevent clogging up the rest of the repo.
This might include some experimental work that gets promoted into the main repo.

## apptainer

There's some really annoying work to get torch and tensorflow working in the same environment.
Here I just create an apptainer environment with the right dependencies and then hope that things work out.
This is the command im using for testing:

```bash
python -m birdclef.infer.workflow process-audio /storage/coda1/p-dsgt_clef2025/0/shared/birdclef/raw/birdclef-2025/train_soundscapes ~/scratch/birdclef/2025/infer-soundscape --assert-gpu --model-name Perch --num-workers 1 --limit 4
```

I can't do torch 2.5.1 and tensorflow 2.18 a the same time because I need cudnn 9.3.
This sucks, and it looks like tf 1.17 doesn't like it when its using cudnn 12.8.
Im going to downgrade to 12.1 and see what that compatibility looks like.
