from .cli import app
if __name__ == "__main__":
    # Typer exposes .app() via main(); prefer app() for help to avoid version quirks.
    app()