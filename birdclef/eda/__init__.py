import typer
from .duration import duration

app = typer.Typer(no_args_is_help=True)
app.command("duration")(duration)
