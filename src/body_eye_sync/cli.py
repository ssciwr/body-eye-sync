import click


@click.command()
@click.version_option(package_name="body-eye-sync", prog_name="body-eye-sync")
def main():
    click.echo("This is body_eye_sync's command line interface.")


if __name__ == "__main__":
    main()
