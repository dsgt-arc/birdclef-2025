import typer
from .birdnet.workflow import app as birdnet_app

app = typer.Typer(no_args_is_help=True)
app.add_typer(birdnet_app, name="birdnet")
