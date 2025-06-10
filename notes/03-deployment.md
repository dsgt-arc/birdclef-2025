# deployment

We upload the models to our cloud bucket for syncing:

```bash
gcloud storage rsync -r ~/shared/birdclef/models/2025/ \
    gs://dsgt-arc-birdclef-2025/models/
```
