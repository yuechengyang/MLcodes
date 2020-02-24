from sklearn.base import BaseEstimator,BiclusterMixin          
        
class fill_na(BaseEstimator,BiclusterMixin):
    def __init__(self,fillna={'fillna_assign_str':None,'fillna_default_str':None}):
        self.fillna = fillna     
    def fit(self,X,y=None):
        return self
    def transform(self,X,convertList=True):
        print('-----------------begin-fillna--------------')
        if self.fillna['fillna_assign_str'] or self.fillna['fillna_default_str']:   
            if self.fillna['fillna_assign_str']:
                for f in self.fillna['fillna_assign_str'].split(','):
                    column = f.split(':')[0]
                    value = f.split(':')[1]
                    dtype = X[column].dtypes
                    if dtype=='int':
                        X[column] = X[column].fillna(int(value))
                    elif dtype=='float':
                        X[column] = X[column].fillna(float(value))
                    else:
                        X[column] = X[column].fillna(value)
            if self.fillna['fillna_default_str']:
                for c in X.columns:
                    if c not in X.select_dtypes('object').columns:
                        X = X.fillna(int(self.fillna['fillna_default_str']))  
                print('no_objective features have been filled with',self.fillna['fillna_default_str'])
        else:
            for o in X.select_dtypes('object').columns:
                X[o] = X[o].fillna('None')
            print('object features have been filled with "None"')  
            
        return X

class get_dummies(BaseEstimator,BiclusterMixin):
    import numpy as np
    def __init__(self,convertList_str=None,featuresList=None,n=10):
        self.n = n
        self.convertList_str = convertList_str
        self.featuresList = featuresList    
        self.convertList = None        
    def fit(self,X,y=None):
        return self
    def transform(self,X,re_convertList=False):
        if self.convertList_str:
            self.convertList = self.convertList_str.split(',')
        else:
            self.convertList = [c for c in self.featuresList if ('int' not in str(X[c].dtypes) and 'float' not in str(X[c].dtypes))]
        for o in self.convertList:
            print(o)
            if X[o].nunique()<=self.n:
                pass
            else:
                print('!!!column',o,'has more than',self.n,'values',',choose top',self.n,'values to get dummies')
                convert_values = list(X.groupby(o).size().sort_values(ascending=False).index[:self.n-1])
                X[o] = X.apply(lambda l:l[o] if l[o] in convert_values else np.nan,axis=1)
            X_tmp = pd.get_dummies(X[o])
            X_tmp.columns = [o+'_'+str(c) for c in X_tmp.columns]  
            print('add columns:',X_tmp.columns)
            X = X.merge(X_tmp,left_index=True,right_index=True) 
        print('------------------------------------conclusion----------------------------------')
        print(self.convertList,'have been converted to dummies')
        if re_convertList==True:
            return X,self.convertList
        else:
            return X        
        
class log(BaseEstimator,BiclusterMixin):
    import numpy as np
    def __init__(self,convertList_str=None,featuresList=None,skew=0.5):
        self.skew = skew
        self.convertList_str = convertList_str
        self.featuresList = featuresList    
        self.convertList = None        
    def fit(self,X,y=None):
        return self
    def transform(self,X,re_convertList=False):
        if self.convertList_str:
            self.convertList = self.convertList_str.split(',')
        if self.featuresList is not None:
            self.convertList = [c for c in self.featuresList if c not in X.select_dtypes('object').columns and X[c].skew()>self.skew]             
        print('add columns',end=': ')
        for c in self.convertList:
            X[c+'_log'] = np.log(X[c]+1)
            print(c+'_log',end=',')
        if re_convertList==True:
            return X,self.convertList
        else:
            return X         
        
class del_samples(BaseEstimator,BiclusterMixin):
    def __init__(self):
        pass      
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        tmp = X.shape[0]
        X.drop(X[:1460][X.MiscVal>=6000].index,inplace = True)
        X.drop(X[:1460][X.LotFrontage>=250].index,inplace = True)
        X.drop(X[:1460][X.BsmtFinSF1>=5000].index,inplace = True)
        X.drop(X[:1460][X.TotalBsmtSF>=5000].index,inplace = True)
        X.drop(X[:1460][X['1stFlrSF']>=4000].index,inplace = True)
        X.drop(X[:1460][(X.GrLivArea>4000)&(X.SalePrice<300000)].index,inplace = True)
        if tmp>X.shape[0]:
            print('remove',tmp-X.shape[0],'samples')
        return X             

class remove_high_relevance(BaseEstimator,BiclusterMixin):
    def __init__(self,featuresList,method='pearson',threshold=0.9):
        self.featuresList =   featuresList
        self.method = method
        self.threshold = threshold
    def fit(self,X,y=None):
        return self
    def transform(self,X,re_convertList=False):
        from tqdm import tqdm
        numeric_features = [c for c in self.featuresList if ('int' in str(X[c].dtypes) or 'float' in str(X[c].dtypes))]
        print('the length of numeric features is:',len(numeric_features))
        removeSet=set()
        iSet = set()
        for i in tqdm(numeric_features):
            if i not in removeSet:
                iSet.add(i)
                for j in numeric_features:
                    if j not in removeSet and i not in removeSet and j not in iSet:
                        pearsonr = (X[[i,j]].corr(method=self.method).loc[i,j])
                        if abs(pearsonr)>self.threshold:
                            print('pearsonr of',i,'and',j,'is',pearsonr,end='! ')
                            if X[i].count()>=X[j].count():
                                removeSet.add(j)
                                print('remove',j)
                            else:
                                removeSet.add(i)
                                print('remove',i)                                  
        featuresList = [c for c in X.columns if c not in removeSet]  
        X = X[featuresList]            
        if re_convertList==True:
            return X,list(removeSet)
        else:
            return X       
