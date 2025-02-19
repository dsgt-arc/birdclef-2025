import typer
from .birdnet.workflow import main as birdnet_main

app = typer.Typer()
app.command("birdnet")(birdnet_main)
