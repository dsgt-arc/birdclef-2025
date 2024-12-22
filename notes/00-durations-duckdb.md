# reading durations using duckdb

We're doing a little bit of experimentation with PACE.
The first thing we do is actually get the durations by processing it using an interactive slurm session:

```bash
salloc \
	-A paceship-dsgt_clef2025 \
	-qinferno -N1 --ntasks-per-node=2 \
	-t1:00:00
```

Then we install the dependencies.
There is a bit of optimizing that needs to be done here because this takes quite a bit of time.
There are a bunch of development dependencies that probably don't need to be included when we actually do a batch job.

```bash
# get into fast scratch disk
cd $TMPDIR
python -m venv venv
# needed for creating packages from a pyproject.toml
pip install --upgrade pip
source venv/bin/activate
pip install -e ~/clef/birdclef-2025
```

Now we can just run the main script:

```bash
birdclef-eda-durations
```

This generates a file in our main project directory.
We CD to it and run `duckdb` which we've installed into our user `bin/` directory.

```
D select * from parquet_schema('durations.parquet');
┌───────────────────┬──────────┬────────────┬─────────────┬───┬───────┬───────────┬──────────┬──────────────┐
│     file_name     │   name   │    type    │ type_length │ … │ scale │ precision │ field_id │ logical_type │
│      varchar      │ varchar  │  varchar   │   varchar   │   │ int64 │   int64   │  int64   │   varchar    │
├───────────────────┼──────────┼────────────┼─────────────┼───┼───────┼───────────┼──────────┼──────────────┤
│ durations.parquet │ schema   │            │             │ … │       │           │          │              │
│ durations.parquet │ path     │ BYTE_ARRAY │             │ … │       │           │          │ StringType() │
│ durations.parquet │ duration │ DOUBLE     │             │ … │       │           │          │              │
├───────────────────┴──────────┴────────────┴─────────────┴───┴───────┴───────────┴──────────┴──────────────┤
│ 3 rows                                                                               11 columns (8 shown) │
└───────────────────────────────────────────────────────────────────────────────────────────────────────────┘

D select sum(duration)/3600 from 'durations.parquet';
┌────────────────────────┐
│ (sum(duration) / 3600) │
│         double         │
├────────────────────────┤
│      843.1613001302157 │
└────────────────────────┘

D select sum(duration)/3600 from 'durations.parquet' where path like 'train%';
┌────────────────────────┐
│ (sum(duration) / 3600) │
│         double         │
├────────────────────────┤
│      284.8415706163194 │
└────────────────────────┘

D select sum(duration)/3600 from 'durations.parquet' where path like '%soundscape%';
┌────────────────────────┐
│ (sum(duration) / 3600) │
│         double         │
├────────────────────────┤
│       558.319729513889 │
└────────────────────────┘
```
