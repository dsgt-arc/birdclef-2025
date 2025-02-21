import typer
from .birdnet.workflow import main as birdnet_main

app = typer.Typer(no_args_is_help=True)
app.command("birdnet")(birdnet_main)
