import typer
from .etl import app as etl_app

app = typer.Typer(no_args_is_help=True)
app.add_typer(etl_app, name="etl")
