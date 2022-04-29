from .ml_models import Regression, BinaryClassification, MultiClassification
from . import db, create_app
from .db_models import Model
import requests, random, compress_pickle

app = create_app()

def train_model(data, target, desc, user, task, shift=False):
    if task == 'regression':
        model_class = Regression(data=data, target=str(target))
        model_class.train()
        model_class.score()

        accuracy = int(round(model_class.accuracy, 2) * 100)

        word_site = "https://www.mit.edu/~ecprice/wordlist.10000"
        response = requests.get(word_site)
        words = response.content.splitlines()
        two_words = random.sample(words, 2)
        word1 = two_words[0].decode()
        word2 = two_words[1].decode()
        final_name = word1 + ' ' + word2

        with app.app_context():
            if Model.query.filter_by(name=final_name).first():
                two_words = random.sample(words, 2)
                word1 = two_words[0].decode()
                word2 = two_words[1].decode()
                final_name = word1 + ' ' + word2

        model_name = final_name

        model_target = target

        model_description = desc

        model_bytes = compress_pickle.dumps(model_class.model, compression="gzip")

        if model_class.is_fitted(model_class.pca):
            pca_bytes = compress_pickle.dumps(model_class.pca, compression="gzip")
        else:
            pca_bytes = None

        gm_bytes = None

        model_accuracy = accuracy

        model_type = model_class.type

        with app.app_context():
            db_instance = Model(owner=str(user), model=model_bytes, 
                                name=model_name, desc=model_description, 
                                accuracy=model_accuracy, target=model_target,
                                pca=pca_bytes, gm=gm_bytes, type_='temp')
            
            db.session.add(db_instance)

            db.session.commit()

        resp = {'name':final_name, 'accuracy':accuracy, 'description':desc, 'type':model_type}

        return resp
    elif task == 'binary':
        if shift == True:
            model_class = BinaryClassification(data=data,target=str(target), shift=True)
            model_class.train()
            preds = model_class.predict()
            model_class.score(predictions=preds)
        else:
            model_class = BinaryClassification(data=data,target=str(target))
            model_class.train()
            preds = model_class.predict()
            model_class.score(predictions=preds)

        accuracy = int(round(model_class.accuracy, 2) * 100)

        word_site = "https://www.mit.edu/~ecprice/wordlist.10000"
        response = requests.get(word_site)
        words = response.content.splitlines()
        two_words = random.sample(words, 2)
        word1 = two_words[0].decode()
        word2 = two_words[1].decode()
        final_name = word1 + ' ' + word2

        with app.app_context():
            if Model.query.filter_by(name=final_name).first():
                two_words = random.sample(words, 2)
                word1 = two_words[0].decode()
                word2 = two_words[1].decode()
                final_name = word1 + ' ' + word2

        model_name = final_name

        model_target = target
       
        model_description = desc

        model_bytes = compress_pickle.dumps(model_class.model, compression="gzip")

        if model_class.is_fitted(model_class.pca):
            pca_bytes = compress_pickle.dumps(model_class.pca, compression="gzip")
        else:
            pca_bytes = None

        gm_bytes = None
    
        model_accuracy = accuracy
     
        model_type = model_class.type

        with app.app_context():
            db_instance = Model(owner=str(user), model=model_bytes, 
                                name=model_name, desc=model_description, 
                                accuracy=model_accuracy, target=model_target,
                                pca=pca_bytes, gm=gm_bytes, type_='temp')
            
            db.session.add(db_instance)

            db.session.commit()

        resp = {'name':final_name, 'accuracy':accuracy, 'description':desc, 'type':model_type}

        return resp
    elif task == 'multi':
        if shift == True:
            model_class = MultiClassification(data=data,target=str(target), shift=True)
            model_class.train()
            preds = model_class.predict()
            model_class.score(predictions=preds)
        else:
            model_class = MultiClassification(data=data,target=str(target))
            model_class.train()
            preds = model_class.predict()
            model_class.score(predictions=preds)

        accuracy = int(round(model_class.accuracy, 2) * 100)

        word_site = "https://www.mit.edu/~ecprice/wordlist.10000"
        response = requests.get(word_site)
        words = response.content.splitlines()
        two_words = random.sample(words, 2)
        word1 = two_words[0].decode()
        word2 = two_words[1].decode()
        final_name = word1 + ' ' + word2

        with app.app_context():
            if Model.query.filter_by(name=final_name).first():
                two_words = random.sample(words, 2)
                word1 = two_words[0].decode()
                word2 = two_words[1].decode()
                final_name = word1 + ' ' + word2

        model_name = final_name

        model_target = target

        model_description = desc

        model_bytes = compress_pickle.dumps(model_class.model, compression="gzip")

        if model_class.pca_is_fitted(model_class.pca):
            pca_bytes = compress_pickle.dumps(model_class.pca, compression="gzip")
        else:
            pca_bytes = None
        
        if model_class.gm_is_fitted(model_class.gm):
            gm_bytes = compress_pickle.dumps(model_class.gm, compression="gzip")
        else:
            gm_bytes = None

        model_accuracy = accuracy

        model_type = model_class.type

        with app.app_context():
            db_instance = Model(owner=str(user), model=model_bytes, 
                                name=model_name, desc=model_description, 
                                accuracy=model_accuracy, target=model_target,
                                pca=pca_bytes, gm=gm_bytes, type_='temp')
            
            db.session.add(db_instance)

            db.session.commit()

        resp = {'name':final_name, 'accuracy':accuracy, 'description':desc, 'type':model_type}

        return resp