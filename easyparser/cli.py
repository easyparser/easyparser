import importlib.util
import subprocess
import sys

import click


@click.group()
def easyparser():
    """CLI tool for document parsing and related utilities."""
    pass


@easyparser.command()
@click.argument("package")
def install(package):
    """Install packages like pandoc using the appropriate method."""
    if package.lower() == "pandoc":
        install_pandoc()
    else:
        click.echo(f"Installation of {package} is not supported yet.")


def is_module_installed(module_name):
    """Check if a Python module is installed."""
    return importlib.util.find_spec(module_name) is not None


def install_pandoc():
    """Install pandoc if not already available using pypandoc's built-in functions."""
    # First, ensure pypandoc is installed
    if not is_module_installed("pypandoc"):
        click.echo("⚙️ Installing pypandoc first...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pypandoc"])
            click.echo("✅ pypandoc installed successfully.")
        except subprocess.CalledProcessError:
            click.echo(
                "❌ Failed to install pypandoc. "
                "Please install it manually with 'pip install pypandoc'."
            )
            return

    # Now use pypandoc to check for pandoc
    import pypandoc

    try:
        pandoc_path = pypandoc.get_pandoc_path()
        click.echo(f"✅ pandoc is already installed at: {pandoc_path}")
        return
    except OSError:
        # This means pypandoc couldn't find pandoc
        click.echo("⚠️ pandoc is not found. Installing now...")

    # Install pandoc using pypandoc
    try:
        from pypandoc.pandoc_download import download_pandoc

        download_pandoc(delete_installer=True)

        # Verify installation
        try:
            pandoc_path = pypandoc.get_pandoc_path()
            click.echo(f"✅ pandoc installed successfully at: {pandoc_path}")
        except OSError:
            click.echo("❌ Failed to verify pandoc installation after installing.")
            click.echo(
                "Please install pandoc manually by following the instructions at:"
            )
            click.echo("https://pandoc.org/installing.html")
    except Exception as e:
        click.echo(f"❌ Failed to install pandoc: {str(e)}")
        click.echo("Please install pandoc manually by following the instructions at:")
        click.echo("https://pandoc.org/installing.html")


if __name__ == "__main__":
    easyparser()
