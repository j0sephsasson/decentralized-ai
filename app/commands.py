import click
from flask.cli import with_appcontext

from .extensions import db
from .db_models import Sale, User, Model, Subscriber

@click.command(name='create_tables')
@with_appcontext
def create_tables():
    db.create_all()