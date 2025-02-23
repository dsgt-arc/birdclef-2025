# embedding soundscapes

We're going to do a bit of analysis on unlabeled soundscapes using a motif mining technique.

We'll embed the soundscapes using the `birdclef etl embed birdnet soundscapes` command.
We run this on PACE, so we run it via an sbatch wrapper.
There are 8444 soundscapes each 10 minutes long.

We run this with a limit first to verify that things run as expected, and to get a rough idea of how long each track takes to embed.

```bash
# if not already, make sure the shared directory is symlinked in home
ln -s /storage/coda1/p-dsgt_clef2025/0/shared ~/shared

TOTAL_BATCHES=1000 \
LIMIT=8 \
NUM_PARTITIONS=1 \
sbatch scripts/sbatch/embed-soundscapes.sbatch \
    ~/shared/birdclef/raw/birdclef-2024/unlabeled_soundscapes \
    ~/scratch/tmp/birdclef/intermediate/birdclef-2024/unlabeled_soundscapes/embed/birdnet \
    ~/shared/birdclef/processed/birdclef-2024/unlabeled_soundscapes/embed/birdnet
```

From the logs, it looks like each soundscape takes ~7 seconds where each task gets 2 cpus.
In our example where we have 1000 batches, each batch takes about 10 minutes to finish.

```bash
...
 20%|██        | 17/85 [02:00<08:01,  7.08s/it]
 21%|██        | 18/85 [02:07<07:55,  7.09s/it]
 22%|██▏       | 19/85 [02:15<07:48,  7.10s/it]
...
 ```

This means the entire dataset would take about 16*2 hours of cpu time.
We should get this done in an hour if we split this across 16 tasks with 2 cpus each.
We try to increase the parallelism further to get this done faster.

Now we run the full thing:

```bash
sbatch scripts/sbatch/embed-soundscapes.sbatch \
    ~/shared/birdclef/raw/birdclef-2024/unlabeled_soundscapes \
    ~/scratch/birdclef/intermediate/birdclef-2024/unlabeled_soundscapes/embed/birdnet \
    ~/shared/birdclef/processed/birdclef-2024/unlabeled_soundscapes/embed/birdnet
```
