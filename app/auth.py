from operator import add
from flask import Blueprint, redirect, url_for, request, flash
from werkzeug.security import generate_password_hash, check_password_hash
from flask.templating import render_template
from .db_models import User, Sale, Model
from . import db
from flask_login import login_user, logout_user, login_required
from werkzeug.utils import secure_filename

auth = Blueprint('auth', __name__)

@auth.route('/login')
def login():
    mint_count = len(Model.query.all())
    user_count = len(User.query.all())
    sale_count = len(Sale.query.all())
    return render_template('login.html', mint_count=mint_count, user_count=user_count, sale_count=sale_count)

## endpoint to login users
@auth.route('/login_post', methods=['POST'])
def login_post():
    address = request.form['data']
    address = str(address.lower())

    user = User.query.filter_by(address=address).first()

    # check if the user actually exists, if not, sign them up!
    if not user:
        new_user = User(address=address)
        db.session.add(new_user)
        db.session.commit()
        login_user(new_user)
        return render_template('profile.html')

    # if the above check passes, then we know the user has the right credentials
    login_user(user)
    return render_template('profile.html')