import typer
from .etl import app as etl_app
from .model import app as model_app

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
app.add_typer(etl_app, name="etl")
app.add_typer(model_app, name="model")
