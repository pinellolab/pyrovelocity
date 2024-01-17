"""Main module."""
import click

try:
    from . import __version__
except:
    __version__ = "unknown"

@click.command()
@click.version_option(version=__version__)
def main():
    """Command line interface for pyrovelocity."""
    click.echo("pyrovelocity")
