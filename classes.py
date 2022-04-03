import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError

##########################
###      Encoders      ###
##########################
                             
class Encoder(BaseEstimator, TransformerMixin):
    def check_fit(self):
        if not self.fitted:
            raise NotFittedError(f"This {__class__.__name__} instance is not fitted yet.\
                Call 'fit' with appropriate arguments before using this estimator.")
    @staticmethod
    def pd(X):
        if not isinstance(X,pd.DataFrame):
            return pd.DataFrame(X)
        else:
            return X

class FeatureTrans:
    ''' Collection of data-transformation features '''

    def __init__(self,data:pd.Series,n:int=5,n_step:int=5,n_max:int=50,empty:bool=False):
        '''Initiating object with data series'''
        if empty:
            return

        if str(data.dtype) == 'category':
            self.data = pd.Series(data.cat.codes, name=data.name, index = data.index )
        else:
            self.data = data

        dataunique = self.data.nunique()

        if dataunique < n_max:
            n_max = dataunique + 1
            n_step = 1
            n = 2

        datatype = str(self.data.dtype)

        dataint = datatype in ['int', 'int32', 'int64']
        datafloat = datatype in ['float', 'float64']
        datanum = dataint or datafloat

        math_func = ["cbrt", "sqrt", "log1p", "log2", "ln", "log10"] + [f"pow{i}" for i in range(2,n)] # TODO: separate variable n
        dict_math = {f"{func}":lambda X, func=func: __class__.math(X,func) for func in math_func}
        dict_math = {"dull":__class__.dull, **dict_math}

        dict_top = {f"top{i}":lambda X, i=i: __class__.top(X,i) for i in range(n,n_max,n_step)}

        dict_ncut = {f"cut{i}":lambda X, i=i: __class__.cut(X,i) for i in range(n,n_max,n_step)}
        dict_qcut = {f"qcut{i}":lambda X, i=i: __class__.qcut(X,i) for i in range(n,n_max,n_step)}
        dict_cut = {**dict_ncut, **dict_qcut}
        
        dict_cat = {
            "count":__class__.count,
            "freq":__class__.freq,
        }

        # Saving methods
        self.method = {
                **dict_cat,
                **dict_top,
            }

        if datanum:
            self.method = {
                **self.method,
                **dict_cut,
                **dict_math
            }

        if dataunique == 2:
            self.method = {
                "bilabel":__class__.bilabel,
                **self.method,
            }

        self.methods = list(self.method.keys())

    def __iter__(self):
        self.current_index = 0
        return self

    def __next__(self):
        i = self.current_index
        
        if i < len(self.methods):
            method_name = self.methods[i]
            method = self.method[method_name]
            x = method(self.data)
            self.current_index += 1
            return x
        raise StopIteration

    @staticmethod
    def dull(data):
        return data

    @staticmethod
    def bilabel(data:pd.Series):
        replaces = {'Yes':1, 'No':0}
        unique = data.unique()
        nonstandart = pd.Series([value not in replaces.keys() for value in unique]).sum()
        if nonstandart > 0 and len(unique) == 2:
            l1, l2 = unique
            replaces = {l1:0,l2:1}
        return pd.Series(data.replace(replaces),name=data.name + "_bilabel")

    @staticmethod
    def fuzzy_mean(data:pd.Series,astype=float) -> pd.Series:
        '''Filling nan values with fuzzy mean values of corresponding columns
           () '''
        data = data.copy()
        d_mean, d_std, d_min, d_max = data.agg(['mean','std','min','max'])
        d_nan = data.isna()
        data[d_nan] = np.random.normal(
            d_mean,
            d_std,
            d_nan.sum()
            ).clip(d_min, d_max)\
            .astype(astype)
        return data
    
    @staticmethod
    def robust(data:pd.Series,method:str) -> pd.Series:
        '''Filling nan values with mean/median'''
        data = data.copy()
        try:
            value = data.agg(method)[0]
        except TypeError:
            value = data.agg(method)
        return data.fillna(value)

    @staticmethod
    def fill(data:pd.Series,filler:str) -> pd.Series:
        '''Filling nan values with some str'''
        data = data.copy()
        data[data.isna()] = filler
        return data

    @staticmethod
    def count(data): 
        ''' Freq count '''
        return pd.Series(data.replace(data.value_counts().to_dict()),name=data.name+"_count")

    @staticmethod
    def freq(data): 
        ''' Freq share'''
        vcounts = data.value_counts()
        maxvc = vcounts.max()
        nunique = data.nunique()
        n = len(data)
        if maxvc < nunique:
            t = "u"
            result = data.replace((vcounts / nunique).to_dict())
        elif maxvc / n > 0.2:
            t = "n"
            result = data.replace((vcounts / n).to_dict())
        else:
            t = "o"
            result = data.replace((vcounts / maxvc).to_dict())
        return pd.Series(result,name=data.name+f"_freq{t}")

    @staticmethod
    def top(data,n):
        ''' Top rating - most frequent becomes 1, less becomes 2, 3 .. n'''
        return pd.Series(data.replace({
            v:i+1 if i < n else n 
            for i,v in enumerate(data.value_counts().index)}),
            name=data.name+f"_top{n}")

    @staticmethod
    def math(data:pd.Series,name:str):
        math_func = {
        "cbrt": {"func":np.cbrt,  "check": lambda X: True},
        "sqrt": {"func":np.sqrt,  "check": lambda X: (X >= 0).all()},
        "log1p":{"func":np.log1p, "check": lambda X: (X >  0).all()},
        "log2": {"func":np.log2,  "check": lambda X: (X >  0).all()},
        "ln":   {"func":np.log,   "check": lambda X: (X >  0).all()},
        "log10":{"func":np.log10, "check": lambda X: (X >  0).all()}
        }

        if name.startswith("pow"):
            pow = int(name[3:])
            return pd.Series( np.power(data,pow), name=data.name + f"_{name}", index = data.index )
        else:
            terms = math_func[name]
            check, func = terms["check"], terms["func"]

            if check(data):
                return pd.Series( func(data), name=data.name + f"_{name}", index = data.index  )
            else:
                return pd.Series( func(data.apply(abs)+1e-24), name=data.name + f"_{name}", index = data.index  ) # TODO: remove hack
    
    @staticmethod
    def qcut(data,n):
        return pd.Series( pd.qcut(data,n,labels=False,duplicates="drop"),name=data.name+f"_qcut{n}")

    @staticmethod
    def cut(data,n):
        return pd.Series( pd.cut(data,n,labels=False,duplicates="drop"),name=data.name+f"_cut{n}")


class NanFixer(Encoder,FeatureTrans):
    '''
    There is different (yet primitive) strategies for filling NAN values
    this class giving a possibility to grid search through them
    '''
    def __init__(self,lim:float=0.225,method:str="mean",filler:str="U"):
        '''Initiating object: copying parameters to self'''
        self.lim = lim
        self.plan = {}
        self.method = method
        self.filler = filler
        self.fitted = False
#         logging.debug("Initialized %s object with parameters:\
# \tmax share of NA in column to apply robust methods: %s,\
# \taggregation mode for robust method: %s,\
# \tfiller for NA values in other cases: %s",
#             __class__.__name__,lim,method,filler)
    
    def fit(self, X:pd.DataFrame,y=None):
        '''
        Making a plan for filling NA values depending on parameters
        and dtypes of data
        '''
        X = self.pd(X)

        for col in X.columns:
            data = X[col]
            datanan = data.isna().sum()/len(data)
            datatype = data.dtype
            datanum = datatype in [np.dtype(t) for t in (int, float, 'int64')]
            datafloat = datatype == np.dtype(float)
            if datanum:
                if datanan > self.lim:
                    self.plan[col] = {
                        "func":__class__.fuzzy_mean,
                        "param":{"astype":float if datafloat else int} 
                        }
                else:
                    self.plan[col] = {
                        "func":__class__.robust,
                        "param":{"method":self.method} 
                        }
            elif datatype == np.dtype(object):
                datamodefreq = data.value_counts()[data.mode()]/len(data)
                if datamodefreq[0] > 0.25: # TODO: in parameters?
                    self.plan[col] = {
                        "func":__class__.robust,
                        "param":{"method":"mode"} 
                        }
                else:     
                    self.plan[col] = {
                        "func":__class__.fill,
                        "param":{"filler":self.filler} 
                        }
        self.fitted = True
        # TODO: drop rows in X then y is NA
        return self

    def transform(self, X:pd.DataFrame):
        '''Following previously setted plan'''
        self.check_fit()

        X = self.pd(X)

        result = pd.DataFrame().reindex_like(X)
        for col in X:
            func, param = self.plan[col]["func"], self.plan[col]["param"]
            result[col] = func(X[col],**param)

        return result
        
class Digitalize(Encoder,FeatureTrans):
    '''
    There is a bunch of methods that can be used to transform data:
    vectorization, hashing, transform by frequency, in bins, by quantiles groups, etc...
    Here we will implement some of them in one class 
    with some logic of automatic application
    '''
    def __init__(self,final:bool=True,alim:int=10,mode:str="raw",ncut:int=5):
        '''Initiating object: copying parameters to self'''
        self.alim = alim
        self.ncut = ncut
        self.mode = mode
        self.final = final
        self.fitted = False
#         logging.debug("Initialized %s object with parameters:\
# \tfinal transform (result must be not whole dataset, but 'X, y'): %s,\
# \tmax additional features for one-hot transformation: %s,\
# \ttransformation mode for numeric features: %s,\
# \tn of top freq features to keep: %s",
#             __class__.__name__,final,alim,mode,ncut)
    
    def fit(self, X:pd.DataFrame,y=None):
        '''
        Making a plan for transforming values 
        depending on parameters and dtypes of data
        '''
        X = self.pd(X)

        self.features = []

        for col in X.columns:
            data = X[col]
            datatype = data.dtype
            datanum = datatype in [np.dtype(t) for t in (int, float, 'int64')]
            dataunique = data.nunique()
            if dataunique < self.alim:
                if dataunique < 3 and datanum: 
                    # logging.debug("Column %s (numeric) would be passed as is (raw)",col)
                    self.features.append({
                        'name':col,  
                        'func':__class__.dull,
                        'param': {}
                    })
                else:
                    # logging.debug("Column %s with %i columns would be one-hot encoded",col,dataunique)
                    self.features.append({
                        'name':col,  
                        'func':pd.get_dummies,
                        'param': {"prefix":col[:5]} #TODO: move it to variables
                        })
            elif datanum:
                if self.mode == "cut":
                    # logging.debug("Column %s (numeric) would be binned in %i equal-spaced bins",col,self.ncut)
                    self.features.append({
                        'name':col,  
                        'func':pd.cut,
                        'param': {"bins":self.ncut,"labels":False}
                    })
                elif self.mode == "qcut":
                    # logging.debug("Column %s (numeric) would be binned in %i equal-sized bins",col,self.ncut)
                    self.features.append({
                        'name':col,  
                        'func':pd.qcut,
                        'param': {"q":self.ncut,"labels":False,"duplicates":"drop"}
                    })
                else: # self.mode == "raw":
                    # logging.debug("Column %s (numeric) would be passed as is (raw)",col)
                    self.features.append({
                        'name':col,  
                        'func':lambda data:data,
                        'param': {}
                    })
            else:
                # logging.debug("Column %s (non-numeric) would be freq.coded with %i top variants",col,self.ncut)
                self.features.append({
                    'name':col,  
                    'func':__class__.freq,
                    'param': {"n":self.ncut}
                    })
        self.fitted = True
        return self

    def transform(self, X:pd.DataFrame):
        '''Following previously setted plan'''
        self.check_fit()
        
        X = self.pd(X)

        data_series = []
        # logging.debug("Transforming features by the plan")
        for feature in self.features:
            name, func, param = feature['name'], feature['func'], feature['param']
            # logging.debug("Column %s: Function %s applied with params %s",name,func,param)
            data_series.append(
                func(X[name],**param)
            )
        df_result = pd.concat(data_series,axis=1)

        return df_result

##########################
###     Estimators     ###
##########################

class SlowPolyLinearReg(FeatureTrans):
    '''
    This is just hand-made regression class 
    to see how it can be done from scratch (using numpy and pandas)
    '''
    # nonlinear functions to generate additional synthetic features
    genmap = {
        "cbrt": {"func":np.cbrt},
        "sqrt": {"func":np.sqrt,  "check": lambda X: (X >= 0).all()},
        "log2": {"func":np.log2,  "check": lambda X: (X >  0).all()},
        "ln":   {"func":np.log,   "check": lambda X: (X >  0).all()},
        "log10":{"func":np.log10, "check": lambda X: (X >  0).all()}
    }

    @staticmethod
    def gener():
        return list(SlowPolyLinearReg.genmap.keys())

    @staticmethod
    def norm():
        return list(SlowPolyLinearReg.normap.keys())

    # functions for saving or retrieving parameters and applying normalization
    def save_load(self, value, name):
        try:
            stored = getattr(self,name)
        except AttributeError:
            stored = None
        if type(stored) != type(None):
            return stored
        else:
            setattr(self,name,value)
            return value

    # min-max rescaling
    def minmax(self,X):
        X_min = SlowPolyLinearReg.save_load(self,X.min(),"norm_min")
        X_max = SlowPolyLinearReg.save_load(self,X.max(),"norm_max")
        return (X-X_min)/(X_max-X_min)

    # max normalization
    def n_max(self,X):
        X_max = SlowPolyLinearReg.save_load(self,X.max(),"norm_max")
        return X/X_max

    # Z-score normalization
    def n_std(self,X):
        X_mean = SlowPolyLinearReg.save_load(self,X.mean(),"norm_mean")
        X_std = SlowPolyLinearReg.save_load(self,X.std(),"norm_std")
        return (X-X_mean)/X_std

    # mean normalization    
    def n_mean(self,X):
        X_mean = SlowPolyLinearReg.save_load(self,X.mean(),"norm_mean")
        X_min = SlowPolyLinearReg.save_load(self,X.min(),"norm_min")
        X_max = SlowPolyLinearReg.save_load(self,X.max(),"norm_max")
        return (X-X_mean)/(X_max-X_min)

    # access to normalization functions from one dictionary
    normap = {
        None:       lambda _, X: X,
        "minmax":   minmax,
        "std":      n_std,
        "max":      n_max,
        "mean":     n_mean
    }
    
    # object creation
    def __init__(self,gener:list=[1],norm:str=None,alpha:float=None):
        '''
        Planning check and data generation for columns (synthetic features addon),
        than data normalization
        '''
        
        self.checks = {}
        self.generators = {}
        self.normalize = __class__.normap.get(norm,lambda _, X: print(f"Can't find {norm} normalization") or X)

        if alpha:
            self.regular = True
            self.alpha = alpha
            print(f"Will apply L2 regularization with coef {alpha:.3f}")
        else:
            self.regular = False

        for key in gener:
            if isinstance(key,str):
                param = __class__.genmap.get(key,{})
                if "func" in param:
                    self.generators[key] = param["func"]
                else:
                    raise ValueError(f"Can't find {key} function")
                if "check" in param:
                    self.checks[key] = param["check"]
            elif isinstance(key,int):
                self.generators[key] = lambda X, key=key: np.power(X,key)

    def __gen_cols(self,X):
        '''
        Generating column(s) 
        '''
        def namecol(data,name):
            data.columns = pd.MultiIndex.from_product([data.columns,[name]],names=['feature', 'function'])

        ndim = X.ndim
        rows, cols = X.shape
        result = pd.DataFrame()

        if X.size == 0:
            return result
        if ndim > 2:
            raise ValueError("Two-dimensional array max!")

        for key, generator in self.generators.items():
            # const hack
            if key == 0:
                addon = pd.DataFrame(np.ones((rows,1)))
                namecol(addon,0)
                result = pd.concat([addon,result],axis=1)
            else:
                addon = X.apply(generator)
                namecol(addon,key)
                result = pd.concat([result,addon],axis=1)

        return result

    def complete(self, X):
        '''
        Apply checks, column generation and normalization
        '''
        X = pd.DataFrame(X)
        for key, check in self.checks.items():
            if not X.apply(check).all():
                raise ValueError(f"Try different column generators, other than {key}")
        X_wide = self.__gen_cols(X)
        X_wide = self.normalize(self,X_wide).fillna(1)
        return X_wide

    def fit(self, X, y):
        '''
        Finding the most suitable weights for feature(s)
        '''
        X_wide = self.complete(X)

        XTX = X_wide.T @ X_wide
        XTY = X_wide.T @ y

        if self.regular:
            XTX = XTX + self.alpha * np.eye(XTX.shape[0])

        if np.linalg.det(XTX) == 0:
            print("Columns or rows are dependent!")
            # TODO: try removing some of the data to get rid of this effect
        elif np.linalg.cond(XTX) > 1e4:
            print("Ill-conditioned matrix!")
        try:
            # closed-form solution
            self.w = np.linalg.inv(XTX) @ (XTY)
        except np.linalg.LinAlgError:
            print("Can't fit the data")
            pass  # TODO: do numerical minimum finding       
    
    def predict(self, X):
        '''
        Applying previously fitted weights to new data
        '''
        X_wide = self.complete(X)
        return X_wide.dot(self.w)
    
    def score(self, X, y):
        '''
        R2-score of prediction
        '''
        X_wide = self.complete(X)
        y_pred = X_wide.dot(self.w)
        return np.corrcoef(y,y_pred)[0,1]**2