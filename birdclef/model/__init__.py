import typer
from . import birdnet

# create app for birdnet
birdnet_app = typer.Typer(no_args_is_help=True)
birdnet_app.command("train")(birdnet.train_model)

# create model_app
app = typer.Typer(no_args_is_help=True)
app.add_typer(birdnet_app, name="birdnet")
