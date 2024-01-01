"""Main module."""
import click

from . import __version__


@click.command()
@click.version_option(version=__version__)
def main():
    """Command line interface for pyrovelocity."""
    click.echo("pyrovelocity")
