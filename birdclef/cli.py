import typer
from .eda import app as eda_app
from .etl import app as etl_app
from .model import app as model_app

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
app.add_typer(eda_app, name="eda")
app.add_typer(etl_app, name="etl")
app.add_typer(model_app, name="model")

if __name__ == "__main__":
    app()
