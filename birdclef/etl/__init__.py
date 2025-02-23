import typer
from .embed import app as embed_app
from .parquet.workflow import app as parquet_app

app = typer.Typer(no_args_is_help=True)
app.add_typer(embed_app, name="embed")
app.add_typer(parquet_app, name="parquet")
