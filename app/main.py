from datetime import time
import json, pandas as pd, compress_pickle, numpy as np
import re, time
from os import error
from flask import Blueprint, render_template, request, flash, session, jsonify
from flask.helpers import url_for
from . import db, mail
from flask_mail import Message
from .db_models import Sale, User, Model, Subscriber
from flask_login import login_required, current_user
from werkzeug.utils import redirect, secure_filename
from .ml_models import Utilize
from rq import Queue
from worker import conn
from rq.job import Job

## initialize flask app
main = Blueprint('main', __name__)

## initialize redis
q = Queue(connection=conn)

# default page of our web-app
@main.route('/', methods=['GET', 'POST'])
def index():
    mint_count = len(Model.query.all())
    user_count = len(User.query.all())
    sale_count = len(Sale.query.all())

    return render_template('index.html', mint_count=mint_count, user_count=user_count, sale_count=sale_count)

# terms of service page of our web-app
@main.route('/terms_of_service', methods=['GET', 'POST'])
def terms_of_service():
    return render_template('terms_of_service.html')

# privacy_policy page of our web-app
@main.route('/privacy_policy', methods=['GET', 'POST'])
def privacy_policy():
    return render_template('privacy_policy.html')

# contact form API
@main.route('/send_message', methods=['POST'])
def send_message():
    try:
        name = str(request.form['name'])
        email = str(request.form['email'])
        subject = str(request.form['subject'])
        message = str(request.form['message'])

        msg = Message(subject=subject, 
        body='Hey Joe! {} has just sent a contact form submission using {} as their email. Here is the message: {}'.format(name, email, message),
                    sender="support@dojoapp.ai",
                    recipients=["sassonjoe66@gmail.com"])

        msg1 = Message(subject='Contact Form Submission',
        body='Hello, {}. Your message has been sent to our team. We will get back to you ASAP!'.format(name),
                    sender="support@dojoapp.ai",
                    recipients=[email])

        mail.send(msg)
        mail.send(msg1)

        return jsonify(resp='MESSAGE SENT SUCCESSFULLY!')
    except:
        return jsonify(resp='Sorry, your message could not be delivered at this time.')

# subscribe API of web-app
@main.route('/subscribe', methods=['POST'])
def subscribe():
    try:
        addy = current_user.address
        email = str(request.form['email'])

        new_sub = Subscriber(address=addy, email=email)
        db.session.add(new_sub)
        db.session.commit()

        return jsonify(resp='Subscribed Successfully!')
    except:
        return jsonify(resp="Please log in to subscribe. If you are logged in, this means you're already subscribed.")

# profile page of our web-app
@main.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    user_models = Model.query.filter_by(owner=current_user.address).all() # query all models on this account
    c_user = User.query.filter_by(address=current_user.address).first() # query the profile address

    if user_models:
        names = []
        metrics = []
        descriptions = []
        on_sale = []

        for model in user_models:
            if model.type_ == 'temp':
                Model.query.filter_by(name=model.name).delete()
                db.session.commit()

            names.append(model.name)
            metrics.append(model.accuracy)
            descriptions.append(model.desc)
            on_sale.append(model.on_sale)

        model_data = list(zip(names,metrics,descriptions,on_sale))

        mint_count = len(Model.query.all())
        user_count = len(User.query.all())
        sale_count = len(Sale.query.all())

        return render_template('profile.html', c_user=c_user.address, model_data=model_data, mint_count=mint_count, user_count=user_count, sale_count=sale_count)
    else:
        mint_count = len(Model.query.all())
        user_count = len(User.query.all())
        sale_count = len(Sale.query.all())
        return render_template('profile.html', c_user=c_user.address, model_data=False, mint_count=mint_count, user_count=user_count, sale_count=sale_count)

# list_item api of our web-app
@main.route('/list_item', methods=['POST'])
@login_required
def list_item():
    model_to_sell = str(request.form['name'])
    price = int(request.form['price'])

    model_db_instance = Model.query.filter_by(name=model_to_sell).first()

    model_sale_instance = Sale(owner=current_user.address, name=model_to_sell, 
    desc=model_db_instance.desc, accuracy=model_db_instance.accuracy, price=price)

    db.session.add(model_sale_instance)

    model_db_instance.on_sale = True

    time.sleep(2)

    db.session.commit()

    return redirect(url_for('main.profile'))

# buy_item api of our web-app
@main.route('/buy_item', methods=['POST'])
@login_required
def buy_item():
    model_to_buy = str(request.form['item'])
    buyer = str(request.form['buyer']).lower()

    model_db_instance = Model.query.filter_by(name=model_to_buy).first()

    model_old_owner = User.query.filter_by(address=model_db_instance.owner).first()
    model_old_owner.collection_size -= 1

    model_new_owner = User.query.filter_by(address=buyer).first()
    model_new_owner.collection_size += 1

    model_db_instance.owner = buyer
    model_db_instance.on_sale = False

    Sale.query.filter_by(name=model_to_buy).delete()

    time.sleep(2)

    db.session.commit()

    return redirect(url_for('main.profile'))

# documentation of our web-app
@main.route('/docs', methods=['GET', 'POST'])
def docs():
    return render_template('docs.html')

# destroy model api of our web-app
@main.route('/delete_item', methods=['POST'])
@login_required
def delete_item():
    model_name = str(request.form['name'])

    Model.query.filter_by(name=model_name).delete()

    time.sleep(2)

    db.session.commit()

    return redirect(url_for('main.index'))

# sports services page of our web-app
@main.route('/sports_services', methods=['GET', 'POST'])
def sports_services():
    mint_count = len(Model.query.all())
    user_count = len(User.query.all())
    sale_count = len(Sale.query.all())
    return render_template('sports.html', mint_count=mint_count, user_count=user_count, sale_count=sale_count)

# healthcare services page of our web-app
@main.route('/healthcare_services', methods=['GET', 'POST'])
def healthcare_services():
    mint_count = len(Model.query.all())
    user_count = len(User.query.all())
    sale_count = len(Sale.query.all())
    return render_template('healthcare.html', mint_count=mint_count, user_count=user_count, sale_count=sale_count)

# marketing services page of our web-app
@main.route('/marketing_services', methods=['GET', 'POST'])
def marketing_services():
    mint_count = len(Model.query.all())
    user_count = len(User.query.all())
    sale_count = len(Sale.query.all())
    return render_template('marketing.html', mint_count=mint_count, user_count=user_count, sale_count=sale_count)

# finance_accounting services page of our web-app
@main.route('/finance_accounting_services', methods=['GET', 'POST'])
def finance_accounting_services():
    mint_count = len(Model.query.all())
    user_count = len(User.query.all())
    sale_count = len(Sale.query.all())
    return render_template('finance_accounting.html', mint_count=mint_count, user_count=user_count, sale_count=sale_count)

# other_services page of our web-app
@main.route('/other_services', methods=['GET', 'POST'])
def other_services():
    mint_count = len(Model.query.all())
    user_count = len(User.query.all())
    sale_count = len(Sale.query.all())
    return render_template('other_services.html', mint_count=mint_count, user_count=user_count, sale_count=sale_count)

# revenue services page of our web-app
@main.route('/revenue_services', methods=['GET', 'POST'])
def revenue_services():
    mint_count = len(Model.query.all())
    user_count = len(User.query.all())
    sale_count = len(Sale.query.all())
    return render_template('revenue.html', mint_count=mint_count, user_count=user_count, sale_count=sale_count)

# use_model api of our web-app
@main.route('/use_model', methods=['POST'])
@login_required
def use_model():
    csv = request.files['upload']
    filename = secure_filename(csv.filename) ## hash filename
    mimetype = csv.mimetype ## get file type
    if not filename or not mimetype:
        return 'Bad upload!', 400

    if mimetype != 'application/vnd.ms-excel':
        return 'Bad upload!', 400

    csv = pd.read_csv(csv)
    csv = csv.loc[:,~csv.columns.duplicated()].copy()
    csv = csv[~csv.index.duplicated(keep='first')].copy()

    model_to_use = str(request.form['name'])
    instance_to_use = int(request.form['row'])

    time.sleep(3)

    model_db_instance = Model.query.filter_by(name=model_to_use).first()

    if not model_db_instance:
        return jsonify(resp='No model found with that name.')

    if model_db_instance.pca:
        pca = compress_pickle.loads(model_db_instance.pca, compression='gzip')
    elif not model_db_instance.pca:
        pca = False
        
    if model_db_instance.gm:
        gm = compress_pickle.loads(model_db_instance.gm, compression='gzip')
    elif not model_db_instance.gm:
        gm = False

    try:
        model = Utilize(data=csv,target=str(model_db_instance.target))
        inp = model.process_data_for_usage(pca=pca)
        f_inputs = np.array(inp.iloc[instance_to_use]).reshape(1,-1)

        model = compress_pickle.loads(model_db_instance.model, compression='gzip')

        preds = model.predict(f_inputs)

        return jsonify(resp=str(preds))
    except:
        return jsonify(resp='It appears your model was trained on different data.')

# endpoint to train regression & classification algorithms
@main.route('/reg_clf_models', methods=['POST'])
@login_required
def reg_clf_models():
    upload = request.files['upload']
    filename = secure_filename(upload.filename)
    mimetype = upload.mimetype

    if not filename or not mimetype:
        m = 'bad upload!'
        return render_template('invalid_upload.html', message=m)

    # if '.csv' in str(request.form['url']): --> for AWS direct upload add on
    if mimetype == 'application/vnd.ms-excel': 
        try:
            # data = pd.read_csv(str(request.form['url']))
            data = pd.read_csv(upload)
        except:
            m = 'No CSV upload found'
            return render_template('invalid_upload.html', message=m)
    # elif '.html' in str(request.form['url']): --> for AWS direct upload add on
    elif mimetype == 'text/html':
        try:
            # data = pd.read_html(str(request.form['url']))[0]
            data = pd.read_html(upload)[0]
        except:
            m = 'No HTML upload found'
            return render_template('invalid_upload.html', message=m)
    # elif '.xlsx' in str(request.form['url']): --> for AWS direct upload add on
    elif mimetype == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
        try:
            # data = pd.DataFrame(pd.read_excel(str(request.form['url'])))
            data = pd.DataFrame(pd.read_excel(upload))
        except:
            m = 'No XLSX upload found'
            return render_template('invalid_upload.html', message=m)
    else:
        m = 'No tabular data uploaded'
        return render_template('invalid_upload.html', message=m)
        
    data = data.loc[:,~data.columns.duplicated()].copy()
    data = data[~data.index.duplicated(keep='first')].copy()

    target = request.form.get('target') ## get model target
    desc = request.form.get('desc') ## get model description

    if str(target) not in list(data.columns):
        time.sleep(3)
        m = 'It appears the target is not in the dataset.'
        return render_template('invalid_upload.html', message=m)

    copy_ = data.copy()
    if len(copy_) < 100:
        time.sleep(3)
        m = 'The dataset is too small.'
        return render_template('invalid_upload.html', message=m)

    n_classes = len(data[str(target)].unique())

    if n_classes == 2: ## load binary-class algorithm
        if request.form['shift'] == True:
            from .utils import train_model
            job = q.enqueue(train_model, data=data, target=target, desc=desc, user=str(current_user.address), task='binary', shift=True)
        else:
            from .utils import train_model
            job = q.enqueue(train_model, data=data, target=target, desc=desc, user=str(current_user.address), task='binary')

        message = 'Your model has been sent to train! Job ID: {}'.format(job.id)

        flash(message)

        return redirect(url_for('main.profile'))

        # return render_template('mint.html', model_data=resp, mint_count=mint_count, user_count=user_count, sale_count=sale_count)


    elif n_classes > 2 and n_classes < np.percentile(range(0,len(data)),70) : ## load multi-class algorithm
        if request.form['shift'] == True:
            from .utils import train_model
            job = q.enqueue(train_model, data=data, target=target, desc=desc, user=str(current_user.address), task='multi', shift=True)
        else:
            from .utils import train_model
            job = q.enqueue(train_model, data=data, target=target, desc=desc, user=str(current_user.address), task='multi')

        message = 'Your model has been sent to train! Job ID: {}'.format(job.id)

        flash(message)

        return redirect(url_for('main.profile'))

        # return render_template('mint.html', model_data=resp, mint_count=mint_count, user_count=user_count, sale_count=sale_count)

    elif n_classes > 2 and n_classes > np.percentile(range(0,len(data)),70): ## load regression algorithm
        if request.form['shift'] == True:
            from .utils import train_model
            job = q.enqueue(train_model, data=data, target=target, desc=desc, user=str(current_user.address), task='regression', shift=True)
        else:
            from .utils import train_model
            job = q.enqueue(train_model, data=data, target=target, desc=desc, user=str(current_user.address), task='regression')

        message = 'Your model has been sent to train! Job ID: {}'.format(job.id)

        flash(message)

        return redirect(url_for('main.profile'))

        # return render_template('mint.html', model_data=resp, mint_count=mint_count, user_count=user_count, sale_count=sale_count)

# send minted NFT data to db
@main.route('/mint', methods=['POST'])
@login_required
def mint():
    address = request.form['data']
    address = str(address.lower())
    model_type = str(request.form['type'])
    model_name = str(request.form['name'])    

    db_instance = Model.query.filter_by(name=model_name).first()
    db_instance.type_ = model_type

    owner = User.query.filter_by(address=address).first()
    owner.collection_size += 1

    time.sleep(2)

    db.session.commit()

    # flash('NFT Minted Successfully!')

    return redirect(url_for('main.profile'))

## marketplace of web-app
@main.route('/marketplace', methods=['GET', 'POST'])
def marketplace():
    models_on_sale = Sale.query.all()

    names = []
    descriptions = []
    accuracys = []
    owners = []
    prices = []

    if models_on_sale:

        for model in models_on_sale:
            names.append(model.name)
            descriptions.append(model.desc)
            accuracys.append(model.accuracy)
            owners.append(model.owner)
            prices.append(model.price)

        final_data = list(zip(names,descriptions,accuracys,owners,prices))

        return render_template('marketplace.html', data=final_data)
    else:
        return render_template('marketplace.html', data=False)

@main.route("/model_data", methods=['GET', 'POST'])
@login_required
def model_data():
    d = {'name':request.args.get('name'), 'accuracy':request.args.get('accuracy'), 
        'description':request.args.get('description'), 'type':request.args.get('type_')}

    mint_count = len(Model.query.all())
    user_count = len(User.query.all())
    sale_count = len(Sale.query.all())

    return render_template('mint.html', model_data=d, mint_count=mint_count, user_count=user_count, sale_count=sale_count)

@main.route("/job_status", methods=['GET', 'POST'])
@login_required
def job_status():
    try:
        job_id = str(request.form['job'])
        job = Job.fetch(job_id, connection=conn)
        status = job.get_status()

        if status == 'finished':
            resp = job.result
            return redirect(url_for('main.model_data', name=resp['name'], accuracy=resp['accuracy'],
                                    description=resp['description'], type_=resp['type']))
        elif status == 'failed':
            message = 'Model has failed to train properly'
            flash(message)
            return redirect(url_for('main.profile'))
        else:
            message = 'Model is still training'
            flash(message)
            return redirect(url_for('main.profile'))
    except:
        flash('No status found for that job')
        return redirect(url_for('main.profile'))
