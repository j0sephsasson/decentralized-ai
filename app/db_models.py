from enum import unique
from flask_login import UserMixin
from . import db

## user table ##
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True) # primary keys are required by SQLAlchemy
    address = db.Column(db.String(100), unique=True)
    collection_size = db.Column(db.Integer, default=0)

## newsletter subscribers table ##
class Subscriber(db.Model):
    id = db.Column(db.Integer, primary_key=True) # primary keys are required by SQLAlchemy
    address = db.Column(db.String(100), unique=True)
    email = db.Column(db.String(100), unique=True)

## ml models table ##
class Model(db.Model):
    id = db.Column(db.Integer, primary_key=True) # primary keys are required by SQLAlchemy
    owner = db.Column(db.String(100), nullable=False) # owner address
    model = db.Column(db.PickleType, nullable=False) # model clf bytes
    name = db.Column(db.Text, nullable=False, unique=True) # model name
    desc = db.Column(db.Text, nullable=False) # model description
    target = db.Column(db.Text, nullable=False) # model target
    accuracy = db.Column(db.Text, nullable=False) # model accuracy
    on_sale = db.Column(db.Boolean, default=False) ## model is on sale --> default False
    pca = db.Column(db.PickleType, nullable=True) # model pca bytes
    gm = db.Column(db.PickleType, nullable=True) # model gm bytes
    type_ = db.Column(db.String(100), nullable=False) # model type (reg,multi,binary)
    

## models for sale table ##
class Sale(db.Model):
    id = db.Column(db.Integer, primary_key=True) # primary keys are required by SQLAlchemy
    owner = db.Column(db.String(100), nullable=False) # owner address
    name = db.Column(db.Text, nullable=False, unique=True) # model name
    desc = db.Column(db.Text, nullable=False) # model description
    accuracy = db.Column(db.Text, nullable=False) # model accuracy
    price = db.Column(db.Text, nullable=False) # sale price