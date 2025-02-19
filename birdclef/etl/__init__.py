import typer
from .embed import app as embed_app

app = typer.Typer()
app.add_typer(embed_app, name="embed")
