import pandas as pd, numpy as np, re, shutil, os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE


##################################################### MULTI-CLASSIFICATION CLASS #####################################################
class MultiClassification:

  """ 
  CUSTOM CLASSIFICATION CLASS BUILT ON TOP OF SKLEARN 
  THIS CLASS CAN TAKE IN ARBITRARY DATA & RETURN A TRAINED CLF MODEL
  MultiClassification --> RandomForestClassifier
  Un-Supervised Methods --> GaussianMixture, PCA

  """

  def __init__(self, data, target, accuracy=0.0, model=RandomForestClassifier(), 
              scaler=StandardScaler(), encoder=LabelEncoder(), X_test=[], y_test=[], 
               gm=GaussianMixture(n_components=10), pca=PCA(n_components=30), 
               smote = SMOTE(), shift=False):

    """ 
    Initialze the class with this constructor 
        Defines:
          -- > model data
          -- > model type
          -- > numerical data scaler
          -- > categorical data encoder
          -- > initialize X & y test as empty
          -- > initialzie accuracy as 0.0
          -- > GaussianMixture
          -- > PCA
    """

    self.data = data
    self.target = target
    self.accuracy = accuracy
    self.model = model
    self.scaler = scaler
    self.encoder = encoder
    self.X_test = X_test
    self.y_test = y_test
    self.gm = gm
    self.pca = pca
    self.smote = smote
    self.shift = shift
    self.type = 'multi'

  def _get_features(self,data,target):

    """ 
    This function retrieves the model features 

    Params:
      -- model data
      -- model target

    Returns:
      -- list of features

    """

    cols = list(data.columns)
    features = [i for i in cols if i!=target]
    return features

  def _validate_date(self, string):

    """ 
    This function validates datetime format strings

    Params:
      -- string
    
    Returns:
      -- Bool
    """

    try:
        datetime.strptime(string, '%m/%d/%Y')
        return True
    except:
        return False

  def _feature_selection(self,data,features):

    """
    This function drops features with high cardinality

    Params:
      -- data
      -- features

    Returns:
      -- list of selected features 
    """

    good_features = []

    for f in features:
      cardinality = len(data[f].unique())
      null_count = data[f].isnull().sum()

      if cardinality > np.percentile(range(0,len(data)),85):
        continue
      elif cardinality < np.percentile(range(0,len(data)),0.1):
        continue
      elif null_count > np.percentile(range(0,len(data)),60):
        continue
      else:
        good_features.append(f)

    return good_features

  def _process_data_for_training(self,data,features,target):

    """ 
    MAIN FUNCTIONALITY 
    This function processes the data for the model
      -- scales numerical data
      -- encodes categorical data
      -- searches & validates datetime columns
      -- validates class amount using the following method:
        -- > if the amount of classes is > 3.5% of all the data
        -- > use the GuassianMixture algorithm to cluster the data
        -- > set the GM cluster outputs as the new target & discard old target
        -- > the GM attribute automatically updates with the fit model
      -- validates amount of features using the following method:
        -- > if the amount of features > 100
        -- > perform PCA on the data

    Params:
      -- model data
      -- model features
      -- model target

    Returns:
      -- X_train, X_test, y_train, y_test

    """

    # define scaler + encoder
    scaler = self.scaler
    enc = self.encoder
    pca = self.pca
    gm = self.gm

    for col in features:
      if data[col].dtypes == 'O':
          imputer = SimpleImputer(strategy='most_frequent')
          data[col] = imputer.fit_transform(data[col].values.reshape(-1, 1))
      elif data[col].dtypes == 'int64' or data[col].dtypes == 'float64':
          imputer = SimpleImputer(strategy='mean')
          data[col] = imputer.fit_transform(data[col].values.reshape(-1, 1))
      
    ## validate the amount of classes ##
    pct = (len(data[target].unique()) / len(data)) * 100
    if pct > 3.5:
      new_target = gm.fit_predict(np.array(data[target].values).reshape(-1, 1))
      new_target = pd.Series(new_target)
      data.drop(columns=target, inplace=True)
      data['new_target'] = new_target
      X = data[features].copy()
      y = data['new_target'].copy()
      self.gm = gm
      
      ## encode categorical features ##
      for col in list(X.columns):
        if X[col].dtypes == 'O':
          X[col] = enc.fit_transform(X[col])

      ### validate amount of features ###
      if len(list(X.columns)) > 100:
        X = pd.DataFrame(data=pca.fit_transform(X), columns=np.arange(30))
        self.pca = pca

      ## scale numercial features ##
      for col in list(X.columns):
        if X[col].dtypes == 'int64' or X[col].dtypes == 'float64':
          X[col] = scaler.fit_transform(X[col].values.reshape(-1, 1))

      X_train, X_test_multi, y_train, y_test_multi = train_test_split(X, y, test_size=0.2, random_state=2)
    else:
      if self.shift:
        data['new_target'] = data[target].shift(-1)
        tmp = data.dropna().copy()
        X = tmp[features].copy()
        y = tmp['new_target'].copy()
      else:
        X = data[features].copy()
        y = data[target].copy()
    
      if y.value_counts(normalize=True).max() > 0.85:
        ## search & validate datetime columns ##
        dt_check = X.iloc[0]
        for i in range(0,len(dt_check)):
          if self._validate_date(str(dt_check[i])) == True:
            X.drop(columns=features[i], inplace=True)
        
        ## encode categorical features ##
        for col in list(X.columns):
          if X[col].dtypes == 'O':
            X[col] = enc.fit_transform(X[col])

        ### validate amount of features ###
        if len(list(X.columns)) > 100:
          X = pd.DataFrame(data=pca.fit_transform(X), columns=np.arange(30))
          self.pca = pca

        ## scale numercial features ##
        for col in list(X.columns):
          if X[col].dtypes == 'int64' or X[col].dtypes == 'float64':
            X[col] = scaler.fit_transform(X[col].values.reshape(-1, 1))
        
        x_smote, y_smote = self.smote.fit_resample(X, y)

        X_train, X_test_multi, y_train, y_test_multi = train_test_split(x_smote, y_smote, test_size=0.2, random_state=2)
      else:
        ## search & validate datetime columns ##
        dt_check = X.iloc[0]
        for i in range(0,len(dt_check)):
          if self._validate_date(str(dt_check[i])) == True:
            X.drop(columns=features[i], inplace=True)
        
        ## encode categorical features ##
        for col in list(X.columns):
          if X[col].dtypes == 'O':
            X[col] = enc.fit_transform(X[col])

        ### validate amount of features ###
        if len(list(X.columns)) > 100:
          X = pd.DataFrame(data=pca.fit_transform(X), columns=np.arange(30))
          self.pca = pca

        ## scale numercial features ##
        for col in list(X.columns):
          if X[col].dtypes == 'int64' or X[col].dtypes == 'float64':
            X[col] = scaler.fit_transform(X[col].values.reshape(-1, 1))

        X_train, X_test_multi, y_train, y_test_multi = train_test_split(X, y, test_size=0.2, random_state=2)

    return X_train, X_test_multi, y_train, y_test_multi

  def process_data_for_usage(self):
    """
    Public function created for model utilization

    -- This function takes in uploaded data and returns processed inputs
    -- This is extremely important to ensure all model input data is uniform
    """

    enc = self.encoder
    scaler = self.scaler
    data = self.data
    target = self.target

    data.dropna(inplace=True)
    data.drop(columns=target, inplace=True)

    features = list(data.columns)
    X = data.copy()

    ## search & validate datetime columns ##
    dt_check = X.iloc[0]
    for i in range(0,len(dt_check)):
      if self._validate_date(str(dt_check[i])) == True:
        X.drop(columns=features[i], inplace=True)
    
    ## encode categorical features ##
    for col in list(X.columns):
      if X[col].dtypes == 'O':
        X[col] = enc.fit_transform(X[col])

    ### validate amount of features ###
    if len(list(X.columns)) > 100:
      X = pd.DataFrame(data=self.pca.fit_transform(X), columns=np.arange(30))

    ## scale numercial features ##
    for col in list(X.columns):
      if X[col].dtypes == 'int64' or X[col].dtypes == 'float64':
        X[col] = scaler.fit_transform(X[col].values.reshape(-1, 1))

    return X

  def train(self):

    """ 
    This function trains the model 

    Returns:
      -- trained model
    
    """

    model = self.model
    data = self.data 
    target = self.target
    features = self._get_features(data=data,target=target)
    good_features = self._feature_selection(data=data,features=features)
    X_train, X_test_multi, y_train, y_test_multi = self._process_data_for_training(data=data,features=good_features,target=target)
    
    model.fit(X_train,y_train)

    self.model = model
    self.X_test = X_test_multi
    self.y_test = y_test_multi

    return self

  def predict(self):

    """ 
    This function makes predictions

    Params:
        -- data (array-like or dataframe row)

    Returns:
        -- predictions
    
    """

    preds = self.model.predict(self.X_test)

    return preds

  def score(self, predictions):

    """ 
    This function scores the model

    Params:
        -- predictions
        -- actuals

    Returns:
        -- model accuray score
    
    """

    score = accuracy_score(predictions,self.y_test)
    self.accuracy = score

    return self

  def pca_is_fitted(self, pca):
    try:
      n = pca.n_features_in_
      return True
    except:
      return False

  def gm_is_fitted(self, gm):
    try:
      n = gm.weights_
      return True
    except:
      return False


##################################################### BINARY CLASSIFICATION CLASS #####################################################
class BinaryClassification:

  """ 
  CUSTOM CLASSIFICATION CLASS BUILT ON TOP OF SKLEARN 
  THIS CLASS CAN TAKE IN ARBITRARY DATA & RETURN A TRAINED CLF MODEL
  BinaryClassification --> LogisticRegression

  """

  def __init__(self, data, target, accuracy=0.0, model=LogisticRegression(), 
              scaler=StandardScaler(), encoder=LabelEncoder(), X_test=[], y_test=[],
              pca=PCA(n_components=30), shift=False, smote = SMOTE()):

    """ 
    Initialze the class with this constructor 
        Defines:
          -- > model data
          -- > model type
          -- > numerical data scaler
          -- > categorical data encoder
          -- > initialize X & y test as empty
          -- > initialzie accuracy as 0.0
          -- > PCA
    """

    self.data = data
    self.target = target
    self.accuracy = accuracy
    self.model = model
    self.scaler = scaler
    self.encoder = encoder
    self.X_test = X_test
    self.y_test = y_test
    self.pca = pca
    self.shift = shift
    self.smote = smote
    self.type = 'binary'

  def _get_features(self,data,target):

    """ 
    This function retrieves the model features 

    Params:
      -- model data
      -- model target

    Returns:
      -- list of features

    """

    cols = list(data.columns)
    features = [i for i in cols if i!=target]
    return features

  def _validate_date(self, string):

    """ 
    This function validates datetime format strings

    Params:
      -- string
    
    Returns:
      -- Bool
    """

    try:
        datetime.strptime(string, '%m/%d/%Y')
        return True
    except:
        return False

  def _feature_selection(self,data,features):

    """
    This function drops features with high cardinality

    Params:
      -- data
      -- features

    Returns:
      -- list of selected features 
    """

    good_features = []

    for f in features:
      cardinality = len(data[f].unique())
      null_count = data[f].isnull().sum()

      if cardinality > np.percentile(range(0,len(data)),85):
        continue
      elif cardinality < np.percentile(range(0,len(data)),0.1):
        continue
      elif null_count > np.percentile(range(0,len(data)),60):
        continue
      else:
        good_features.append(f)

    return good_features

  def _process_data_for_training(self,data,features,target):

    """ 
    MAIN FUNCTIONALITY 
    This function processes the data for the model
      -- scales numerical data
      -- encodes categorical data
      -- searches & validates datetime columns
      -- validates amount of features using the following method:
        -- > if the amount of features > 100
        -- > perform PCA on the data

    Params:
      -- model data
      -- model features
      -- model target

    Returns:
      -- X_train, X_test, y_train, y_test

    """

    scaler = self.scaler
    enc = self.encoder
    pca = self.pca

    for col in features:
      if data[col].dtypes == 'O':
          imputer = SimpleImputer(strategy='most_frequent')
          data[col] = imputer.fit_transform(data[col].values.reshape(-1, 1))
      elif data[col].dtypes == 'int64' or data[col].dtypes == 'float64':
          imputer = SimpleImputer(strategy='mean')
          data[col] = imputer.fit_transform(data[col].values.reshape(-1, 1))

    if data[target].dtypes == 'O':
      data[target] = enc.fit_transform(data[target])

    if self.shift:
      data['new_target'] = data[target].shift(-1)
      tmp = data.dropna().copy()
      X = tmp[features].copy()
      y = tmp['new_target'].copy()
    else:
      X = data[features].copy()
      y = data[target].copy()

    if y.value_counts(normalize=True).max() > 0.85:
      ## search & validate datetime columns ##
      dt_check = X.iloc[0]
      for i in range(0,len(dt_check)):
        if self._validate_date(str(dt_check[i])) == True:
          X.drop(columns=features[i], inplace=True)
      
      ## encode categorical features ##
      for col in list(X.columns):
        if X[col].dtypes == 'O':
          X[col] = enc.fit_transform(X[col])

      ### validate amount of features ###
      if len(list(X.columns)) > 100:
        X = pd.DataFrame(data=pca.fit_transform(X), columns=np.arange(30))
        self.pca = pca

      ## scale numercial features ##
      for col in list(X.columns):
        if X[col].dtypes == 'int64' or X[col].dtypes == 'float64':
          X[col] = scaler.fit_transform(X[col].values.reshape(-1, 1))
      
      x_smote, y_smote = self.smote.fit_resample(X, y)

      X_train, X_test_binary, y_train, y_test_binary = train_test_split(x_smote, y_smote, test_size=0.2, random_state=2)
    else:
      ## search & validate datetime columns ##
      dt_check = X.iloc[0]
      for i in range(0,len(dt_check)):
        if self._validate_date(str(dt_check[i])) == True:
          X.drop(columns=features[i], inplace=True)
      
      ## encode categorical features ##
      for col in list(X.columns):
        if X[col].dtypes == 'O':
          X[col] = enc.fit_transform(X[col])

      ### validate amount of features ###
      if len(list(X.columns)) > 100:
        X = pd.DataFrame(data=pca.fit_transform(X), columns=np.arange(30))
        self.pca = pca

      ## scale numercial features ##
      for col in list(X.columns):
        if X[col].dtypes == 'int64' or X[col].dtypes == 'float64':
          X[col] = scaler.fit_transform(X[col].values.reshape(-1, 1))

      X_train, X_test_binary, y_train, y_test_binary = train_test_split(X, y, test_size=0.2, random_state=2)

    return X_train, X_test_binary, y_train, y_test_binary

  def process_data_for_usage(self):
    """
    Public function created for model utilization

    -- This function takes in uploaded data and returns processed inputs
    -- This is extremely important to ensure all model input data is uniform
    """

    enc = self.encoder
    scaler = self.scaler
    data = self.data
    target = self.target

    data.dropna(inplace=True)
    data.drop(columns=target, inplace=True)

    features = list(data.columns)
    X = data.copy()

    ## search & validate datetime columns ##
    dt_check = X.iloc[0]
    for i in range(0,len(dt_check)):
      if self._validate_date(str(dt_check[i])) == True:
        X.drop(columns=features[i], inplace=True)
    
    ## encode categorical features ##
    for col in list(X.columns):
      if X[col].dtypes == 'O':
        X[col] = enc.fit_transform(X[col])

    ### validate amount of features ###
    if len(list(X.columns)) > 100:
      X = pd.DataFrame(data=self.pca.fit_transform(X), columns=np.arange(30))

    ## scale numercial features ##
    for col in list(X.columns):
      if X[col].dtypes == 'int64' or X[col].dtypes == 'float64':
        X[col] = scaler.fit_transform(X[col].values.reshape(-1, 1))

    return X

  def train(self):

    """ 
    This function trains the model 

    Returns:
      -- trained model
    
    """

    model = self.model
    data = self.data 
    target = self.target
    features = self._get_features(data=data,target=target)
    good_features = self._feature_selection(data=data,features=features)
    X_train, X_test_binary, y_train, y_test_binary = self._process_data_for_training(data=data,features=good_features,target=target)
    
    model.fit(X_train,y_train)

    self.model = model
    self.X_test = X_test_binary
    self.y_test = y_test_binary

    return self

  def predict(self):

    """ 
    This function makes predictions

    Params:
        -- data (array-like or dataframe row)

    Returns:
        -- predictions
    
    """

    preds = self.model.predict(self.X_test)

    return preds

  def score(self, predictions):

    """ 
    This function scores the model

    Params:
        -- predictions
        -- actuals

    Returns:
        -- model accuray score
    
    """

    score = accuracy_score(predictions,self.y_test)
    self.accuracy = score

    return self

  def is_fitted(self, pca):
    try:
      n = pca.n_features_in_
      return True
    except:
      return False


##################################################### REGRESSION CLASS #####################################################
class Regression:

  """ 
  CUSTOM REGRESSION CLASS BUILT ON TOP OF SKLEARN 
  THIS CLASS CAN TAKE IN ARBITRARY DATA & RETURN A TRAINED REGRESSION MODEL
  REGRESSION --> RandomForestRegressor

  """

  def __init__(self, data, target, accuracy=0.0, model=RandomForestRegressor(), 
              scaler=StandardScaler(), encoder=LabelEncoder(), X_test=[], y_test=[],
              pca=PCA(n_components=30)):

    """ 
    Initialze the class with this constructor 
        Defines:
          -- > model data
          -- > model type
          -- > numerical data scaler
          -- > categorical data encoder
          -- > initialize X & y test as empty
          -- > initialzie accuracy as 0.0
          -- > PCA
    """

    self.data = data
    self.target = target
    self.accuracy = accuracy
    self.model = model
    self.scaler = scaler
    self.encoder = encoder
    self.X_test = X_test
    self.y_test = y_test
    self.pca = pca
    self.type = 'regression'

  def _get_features(self,data,target):

    """ 
    This function retrieves the model features 

    Params:
      -- model data
      -- model target

    Returns:
      -- list of features

    """

    cols = list(data.columns)
    features = [i for i in cols if i!=target]
    return features

  def _validate_date(self, string):

    """ 
    This function validates datetime format strings

    Params:
      -- string
    
    Returns:
      -- Bool
    """

    try:
        datetime.strptime(string, '%m/%d/%Y')
        return True
    except:
        return False

  def _feature_selection(self,data,features):

    """
    This function drops features with high cardinality

    Params:
      -- data
      -- features

    Returns:
      -- list of selected features 
    """

    good_features = []

    for f in features:
      cardinality = len(data[f].unique())
      null_count = data[f].isnull().sum()

      if cardinality > np.percentile(range(0,len(data)),85):
        continue
      elif cardinality < np.percentile(range(0,len(data)),0.1):
        continue
      elif null_count > np.percentile(range(0,len(data)),60):
        continue
      else:
        good_features.append(f)

    return good_features

  def _process_data_for_training(self,data,features,target):

    """ 
    MAIN FUNCTIONALITY 
    This function processes the data for the model
      -- scales numerical data
      -- encodes categorical data
      -- searches & validates datetime columns
      -- validates amount of features using the following method:
        -- > if the amount of features > 100
        -- > perform PCA on the data

    Params:
      -- model data
      -- model features
      -- model target

    Returns:
      -- X_train, X_test, y_train, y_test

    """

    scaler = self.scaler
    enc = self.encoder
    pca = self.pca

    for col in features:
      if data[col].dtypes == 'O':
          imputer = SimpleImputer(strategy='most_frequent')
          data[col] = imputer.fit_transform(data[col].values.reshape(-1, 1))
      elif data[col].dtypes == 'int64' or data[col].dtypes == 'float64':
          imputer = SimpleImputer(strategy='mean')
          data[col] = imputer.fit_transform(data[col].values.reshape(-1, 1))
    
    if data[target].dtypes == 'O':
      data[target] = enc.fit_transform(data[target])

    X = data[features].copy()
    y = data[target].copy()

    ## search & validate datetime columns ##
    dt_check = X.iloc[0]
    for i in range(0,len(dt_check)):
      if self._validate_date(str(dt_check[i])) == True:
        X.drop(columns=features[i], inplace=True)
        
    ## encode categorical features ##
    for col in list(X.columns):
      if X[col].dtypes == 'O':
        X[col] = enc.fit_transform(X[col])

    ### validate amount of features ###
    if len(list(X.columns)) > 100:
      X = pd.DataFrame(data=pca.fit_transform(X), columns=np.arange(30))
      self.pca = pca

    ## scale numercial features ##
    for col in list(X.columns):
      if X[col].dtypes == 'int64' or X[col].dtypes == 'float64':
        X[col] = scaler.fit_transform(X[col].values.reshape(-1, 1))

    X_train, X_test_reg, y_train, y_test_reg = train_test_split(X, y, test_size=0.2, random_state=2)

    return X_train, X_test_reg, y_train, y_test_reg

  def process_data_for_usage(self):
    """
    Public function created for model utilization

    -- This function takes in uploaded data and returns processed inputs
    -- This is extremely important to ensure all model input data is uniform
    """

    enc = self.encoder
    scaler = self.scaler
    data = self.data
    target = self.target

    data.dropna(inplace=True)
    data.drop(columns=target, inplace=True)

    features = list(data.columns)
    X = data.copy()

    ## search & validate datetime columns ##
    dt_check = X.iloc[0]
    for i in range(0,len(dt_check)):
      if self._validate_date(str(dt_check[i])) == True:
        X.drop(columns=features[i], inplace=True)
    
    ## encode categorical features ##
    for col in list(X.columns):
      if X[col].dtypes == 'O':
        X[col] = enc.fit_transform(X[col])

    ### validate amount of features ###
    if len(list(X.columns)) > 100:
      X = pd.DataFrame(data=self.pca.fit_transform(X), columns=np.arange(30))

    ## scale numercial features ##
    for col in list(X.columns):
      if X[col].dtypes == 'int64' or X[col].dtypes == 'float64':
        X[col] = scaler.fit_transform(X[col].values.reshape(-1, 1))

    return X

  def train(self):

    """ 
    This function trains the model 

    Returns:
      -- trained model
    
    """

    model = self.model
    data = self.data 
    target = self.target
    features = self._get_features(data=data,target=target)
    good_features = self._feature_selection(data=data,features=features)
    X_train, X_test_reg, y_train, y_test_reg = self._process_data_for_training(data=data,features=good_features,target=target)
    
    model.fit(X_train,y_train)

    self.model = model
    self.X_test = X_test_reg
    self.y_test = y_test_reg

    return self

  def predict(self, data):

    """ 
    This function makes predictions

    Params:
        -- data (array-like or dataframe row)

    Returns:
        -- predictions
    
    """

    preds = self.model.predict(data)

    return preds

  def score(self):

    """ 
    This function scores the model

    Params:
        -- predictions
        -- actuals

    Returns:
        -- model accuray score
    
    """

    score = self.model.score(self.X_test, self.y_test)
    self.accuracy = score

    return self

  def is_fitted(self, pca):
    try:
      n = pca.n_features_in_
      return True
    except:
      return False


##################################################### UTILIZE CLASS #####################################################
class Utilize:

  """ 
  CUSTOM CLASS FOR UTILIZING MODELS

  This class processes data to ensure model inputs are uniform

  """

  def __init__(self, data, target, scaler=StandardScaler(), encoder=LabelEncoder()):

    """ 
    Initialze the class with this constructor 
        Defines:
          -- > model data
          -- > numerical data scaler
          -- > categorical data encoder
    """

    self.data = data
    self.target = target
    self.scaler = scaler
    self.encoder = encoder

  def _validate_date(self, string):

    """ 
    This function validates datetime format strings

    Params:
      -- string
    
    Returns:
      -- Bool
    """

    try:
        datetime.strptime(string, '%m/%d/%Y')
        return True
    except:
        return False

  def _feature_selection(self,data,features):

    """
    This function drops features with high cardinality

    Params:
      -- data
      -- features

    Returns:
      -- list of selected features 
    """

    good_features = []

    for f in features:
      cardinality = len(data[f].unique())
      null_count = data[f].isnull().sum()

      if cardinality > np.percentile(range(0,len(data)),85):
        continue
      elif cardinality < np.percentile(range(0,len(data)),0.1):
        continue
      elif null_count > np.percentile(range(0,len(data)),60):
        continue
      else:
        good_features.append(f)

    return good_features

  def process_data_for_usage(self, pca=False, gm=False):
    """
    Public function created for model utilization

    -- This function takes in uploaded data and returns processed inputs
    -- This is extremely important to ensure all model input data is uniform
    """

    enc = self.encoder
    scaler = self.scaler
    data = self.data
    target = self.target

    try:
      data.drop(columns=target, inplace=True)
    except:
      print('NO TARGET IN "use_model" UPLOAD')

    features = self._feature_selection(data=data,features=list(data.columns))
    for col in features:
      if data[col].dtypes == 'O':
          imputer = SimpleImputer(strategy='most_frequent')
          data[col] = imputer.fit_transform(data[col].values.reshape(-1, 1))
      elif data[col].dtypes == 'int64' or data[col].dtypes == 'float64':
          imputer = SimpleImputer(strategy='mean')
          data[col] = imputer.fit_transform(data[col].values.reshape(-1, 1))

    X = data[features].copy()

    ## search & validate datetime columns ##
    dt_check = X.iloc[0]
    for i in range(0,len(dt_check)):
      if self._validate_date(str(dt_check[i])) == True:
        X.drop(columns=features[i], inplace=True)
    
    ## encode categorical features ##
    for col in list(X.columns):
      if X[col].dtypes == 'O':
        X[col] = enc.fit_transform(X[col])

    ### validate amount of features ###
    if len(list(X.columns)) > 100:
      X = pd.DataFrame(data=pca.transform(X), columns=np.arange(30))

    ## scale numercial features ##
    for col in list(X.columns):
      if X[col].dtypes == 'int64' or X[col].dtypes == 'float64':
        X[col] = scaler.fit_transform(X[col].values.reshape(-1, 1))

    return X