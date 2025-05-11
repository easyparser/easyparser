import importlib.util
import os
import platform
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
    """Install packages like pandoc or libreoffice using the appropriate method"""
    if package.lower() == "pandoc":
        install_pandoc()
    elif package.lower() == "libreoffice":
        install_libreoffice()
    else:
        click.echo(f"Installation of {package} is not supported yet.")


def is_module_installed(module_name):
    """Check if a Python module is installed."""
    return importlib.util.find_spec(module_name) is not None


def is_command_available(command):
    """Check if a command is available in the system path."""
    try:
        result = subprocess.run(
            ["which", command] if os.name != "nt" else ["where", command],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except Exception:
        return False


def install_pandoc():
    """Install pandoc if not already available using pypandoc's built-in functions"""
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
        if not os.path.exists(pandoc_path):
            pf = sys.platform
            pandoc_path = os.path.join(pypandoc.DEFAULT_TARGET_FOLDER[pf], pandoc_path)
        click.echo(f"✅ pandoc is already installed at: {pandoc_path}")
        return
    except OSError:
        # This means pypandoc couldn't find pandoc
        click.echo("⚠️ pandoc is not found. Installing now...")

    # Install pandoc using pypandoc
    try:
        pypandoc.download_pandoc(delete_installer=True)

        # Verify installation
        try:
            pandoc_path = pypandoc.get_pandoc_path()
            if not os.path.exists(pandoc_path):
                pf = sys.platform
                pandoc_path = os.path.join(
                    pypandoc.DEFAULT_TARGET_FOLDER[pf], pandoc_path
                )
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


def install_libreoffice():
    """Check if LibreOffice is installed and provide installation instructions if not"""
    # Commands to check for LibreOffice, depending on platform
    check_commands = {
        "Windows": ["soffice.exe"],
        "Darwin": ["soffice", "libreoffice"],  # macOS
        "Linux": ["soffice", "libreoffice"],
    }

    system = platform.system()
    commands = check_commands.get(system, ["libreoffice"])

    # Check if any of the commands are available
    for cmd in commands:
        if is_command_available(cmd):
            click.echo(
                f"✅ LibreOffice is already installed and available via '{cmd}'."
            )
            return

    click.echo("⚠️ LibreOffice is not found on your system.")

    # Platform-specific installation instructions
    if system == "Windows":
        click.echo("To install LibreOffice on Windows:")
        click.echo(
            "1. Visit https://www.libreoffice.org/download/download-libreoffice/"
        )
        click.echo("2. Download the Windows version")
        click.echo("3. Run the installer and follow the instructions")

    elif system == "Darwin":  # macOS
        click.echo("To install LibreOffice on macOS:")
        click.echo(
            "Option 1: Using Homebrew (recommended if you have Homebrew installed):"
        )
        click.echo("    brew install --cask libreoffice")
        click.echo("")
        click.echo("Option 2: Manual installation:")
        click.echo(
            "1. Visit https://www.libreoffice.org/download/download-libreoffice/"
        )
        click.echo("2. Download the macOS version")
        click.echo(
            "3. Open the .dmg file and drag LibreOffice to your Applications folder"
        )

    elif system == "Linux":
        # Try to detect the Linux distribution
        try:
            with open("/etc/os-release") as f:
                os_info = f.read()

            if "ID=ubuntu" in os_info or "ID=debian" in os_info:
                click.echo("To install LibreOffice on Ubuntu/Debian:")
                click.echo("    sudo apt update")
                click.echo("    sudo apt install libreoffice")

            elif (
                "ID=fedora" in os_info or "ID=rhel" in os_info or "ID=centos" in os_info
            ):
                click.echo("To install LibreOffice on Fedora/RHEL/CentOS:")
                click.echo("    sudo dnf install libreoffice")

            elif "ID=arch" in os_info:
                click.echo("To install LibreOffice on Arch Linux:")
                click.echo("    sudo pacman -S libreoffice")

            else:
                click.echo("To install LibreOffice on Linux:")
                click.echo(
                    "1. Use your distribution's package manager to install the "
                    "libreoffice package"
                )
                click.echo(
                    "2. Or visit "
                    "https://www.libreoffice.org/download/download-libreoffice/ "
                    "for manual installation"
                )

        except FileNotFoundError:
            click.echo("To install LibreOffice on Linux:")
            click.echo(
                "1. Use your distribution's package manager to install the "
                "libreoffice package"
            )
            click.echo(
                "2. Or visit "
                "https://www.libreoffice.org/download/download-libreoffice/ "
                "for manual installation"
            )

    else:
        click.echo(f"Unsupported operating system: {system}")
        click.echo(
            "Please visit https://www.libreoffice.org/download/download-libreoffice/ "
            "for installation instructions."
        )


if __name__ == "__main__":
    easyparser()
