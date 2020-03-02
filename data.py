import time
import sys
import dask.dataframe as dd
import pandas as pd
#from memory_profiler import profile

def timefunc(f):    
    def f_timer(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()        
        print ('... Time run ==>' ,f.__name__, 'took', round(end - start,4), 'seconds' )       
        return result
    return f_timer

def reduce_data_memory(data):
    for f in ['int','float','object']:
        if f=='object':
            data_object = data.select_dtypes(include=[f])
            for c in data_object.columns:
                if data_object[c].nunique()==1:
                    print('!!!object column '+c+' has only 1 value')
        else:
            data[data.select_dtypes(include=[f]).columns] = data.select_dtypes(include=[f]).apply(pd.to_numeric,downcast='unsigned')   
    return data

@timefunc
def read_data_with_cond(data_file_str,reduce_memory=False,cond_and_str=None,output_path_pre=None,sep='\t'): 
    import re     
    print('----------------begin------------------')    
    try:
        if sep=='\t':
            data = dd.read_table(data_file_str,low_memory=False,dtype={'uid': 'object'}).compute()
        if sep==',':
            data = dd.read_csv(data_file_str,low_memory=False,dtype={'uid': 'object'}).compute()  
    except:
        if sep=='\t':
            data = pd.read_table(data_file_str,low_memory=False,dtype={'uid': 'object'})
        if sep==',':
            data = pd.read_csv(data_file_str,low_memory=False,dtype={'uid': 'object'})     
    print('--initial')
    print(data.info())    
    if reduce_memory:
        print('--reduce_memory')
        data = reduce_data_memory(data)
        print(data.info())

    if cond_and_str:
        print('--cond')
        cnt=1
        for cond in cond_and_str.split(','):
            pattern = re.compile(r'^.*>=.*$')
            if pattern.match(cond):
                f,n = cond.split('>=')[0],int(cond.split('>=')[1])
                data = data[data[f]>=n]
                print('shape of data after cond',cnt,':',data.shape)
            pattern = re.compile(r'^.*==.*$')
            if pattern.match(cond):
                f,n = cond.split('==')[0],int(cond.split('==')[1])
                data = data[data[f]==n]
                print('shape of data after cond',cnt,':',data.shape)                
            pattern = re.compile(r'^.*isnull.*$')
            if pattern.match(cond):
                f = cond.split('.')[0]
                data = data[data[f].isnull()]
                print('shape of data after cond',cnt,':',data.shape)
            cnt+=1
        print(data.info())
    print('------------conclusion---------------')
    print('shape of dataset:',data.shape)  
    print('-------------outputs------------------')
    if output_path_pre:    
        columns = pd.DataFrame(data.dtypes)
        columns = columns.reset_index()
        columns.columns = ['feature_name','dtypes']
        columns.to_csv(output_path_pre+'columns.csv',index=False,header=True)
        print('column names and dtypes have been downloaded to ',output_path_pre+'columns.csv')        
    return data

@timefunc
def remove_high_relevance(train,featuresList,method='pearson',threshold=0.9):
    from tqdm import tqdm
    numeric_features = [c for c in featuresList if ('int' in str(train[c].dtypes) or 'float' in str(train[c].dtypes))]
    print('the length of numeric features is:',len(numeric_features))
    removeSet=set()
    iSet = set()
    for i in tqdm(numeric_features):
        if i not in removeSet:
            iSet.add(i)
            for j in numeric_features:
                if j not in removeSet and i not in removeSet and j not in iSet:
                    pearsonr = (train[[i,j]].corr(method=method).loc[i,j])
                    if abs(pearsonr)>threshold:
                        print('pearsonr of',i,'and',j,'is',pearsonr,end='! ')
                        if train[i].count()>=train[j].count():
                            removeSet.add(j)
                            print('remove',j)
                        else:
                            removeSet.add(i)
                            print('remove',i)  
    return list(removeSet)


@timefunc
def get_features_attr(data,idList_str=None,targetList_str=None,featuresList=None,missing_warn=0.99,remove={'list':None,'missing':False,'single':False},output_path_pre=None):
    import pickle
    import openpyxl
    removeSet = set()    

    if featuresList is None:
        idList = idList_str.split(',')
        targetList = targetList_str.split(',')
        featuresList = [c for c in data.columns if c not in idList and c not in targetList]
        
    if remove['list']:
        removeList = remove['list'].split(',')
        featuresList = [c for c in featuresList if c not in removeList]
        print('---------------------removelist-----------------')
        print('remove',len(removeList),'features')
        
    print('----------------missing-------------------')
    feature_missing = []
    remove_missing = []
    for i in featuresList:
        if data[i].count()/data.shape[0]<1:
            feature_missing.append([i,round(1-data[i].count()/data.shape[0],4),str(data[i].dtypes)])
            if 1-data[i].count()/data.shape[0]>missing_warn:
                print('!!!WARN missing_rate of',i,'is',round(1-data[i].count()/data.shape[0],4))
                if remove['missing']:                    
                    remove_missing.append(i)
                    del feature_missing[-1]        
                                        
    if len(feature_missing)>0:
        feature_missing = pd.DataFrame(feature_missing)
        feature_missing.columns = ['feature_name','missing_rate','dtype']      
        feature_missing.sort_values(by=['missing_rate'],ascending=False)
        if len(remove_missing)>0:
            print('remove',len(remove_missing),'missing features,','remaining',feature_missing.shape[0],'missing features')
        else:
            print('there are',feature_missing.shape[0],'missing features')  
    else:
        print('remove',len(remove_missing),'missing features,','remaining',0,'missing features')
        
    print('------------------single---------------------')
    remove_single=[]
    for i in featuresList:
        if data[i].nunique()<=1:
            print('!!!WARN nunique of',i,'is only',data[i].nunique())
            if remove['single']:
                remove_single.append(i)   
    if len(remove_single)>0:            
        print('remove',len(remove_single),'features with single value')
        
    print('--------------------totally---------------------')
    featuresList = [c for c in featuresList if c not in remove_missing and c not in remove_single]
    print('numbers of features:',len(featuresList))        
    dtypes_df = pd.DataFrame(data[featuresList].dtypes).reset_index().groupby([0]).count().reset_index()
    for i in dtypes_df[0]:
        print('--numbers of',i,'features are:',dtypes_df[dtypes_df[0]==i]['index'].values[0])     
        
    print('-------------outputs------------------')  
    if output_path_pre:
        with open(output_path_pre+'features.pkl','wb') as f:
            pickle.dump(featuresList,f)    
        print('feature list has been downloaded to ',output_path_pre+'features.pkl')    
               
    if output_path_pre and len(feature_missing)>0:
        feature_missing.to_csv(output_path_pre+'feature_missing.csv',index=False,header=True,float_format='%0.2f')    
        print('missing features has been downloaded to ',output_path_pre+'feature_missing.csv')
        
    if output_path_pre:
        writer = pd.ExcelWriter(output_path_pre+'feature_describe.xlsx')
        for d in data[featuresList].dtypes.unique():
            data[featuresList].select_dtypes(include=[d]).describe().T.to_excel(writer,str(d))
        writer.save()
        print('features describe has been downloaded to ',output_path_pre+'feature_describe.xlsx')
        
    return featuresList,feature_missing



@timefunc
def display_x_y(data,features_list,target_str,output_path_pre,target_type='category'):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pickle
    import re
    from matplotlib.backends.backend_pdf import PdfPages
    from tqdm import tqdm
    import numpy as np
    
    if isinstance(features_list,str):
        with open(features_list, 'rb') as f:
            featuresList = pickle.load(f)  
    else:
        featuresList = features_list
    target = target_str
    
    if target_type=='category':        
        # float
        pattern = re.compile(r'^.*float.*$')
        float_features = [c for c in featuresList if pattern.match(str(data[c].dtypes)) and data[c].nunique()>2]
        object_features = [c for c in featuresList if pattern.match(str(data[c].dtypes)) and data[c].nunique()<=2]
        if len(float_features)>0:
            print('--------exploring float features--------')
            with PdfPages(output_path_pre+'float_plot.pdf') as pdf:
                for i in tqdm(float_features):
                    fig = plt.figure(figsize=(8, 6))
                    sns.boxplot(x=target, y=i, data=pd.concat([data[target], data[i]], axis=1),showfliers=False)
                    pdf.savefig(fig)   
            print('plots have been saved at',output_path_pre+'float_plot.pdf')
        
        #int
        pattern = re.compile(r'^.*int.*$')
        int_features = [c for c in featuresList if pattern.match(str(data[c].dtypes))]
        if len(int_features)>0:
            print('-------------exploring int features----------')
            int_tab_features = [c for c in int_features if data[c].nunique()<=3]
            if len(int_tab_features)>0:
                with pd.ExcelWriter(output_path_pre+'int_tab.xlsx') as writer:
                    for i in tqdm(int_tab_features):
                        crosstab_num = data[[target,i]].pivot_table(index=[target],columns=i,aggfunc=len,margins=True)
                        crosstab_freq = pd.DataFrame(np.array(crosstab_num)/np.array(crosstab_num.loc['All',:]).reshape((1,-1)))
                        crosstab_freq.columns,crosstab_freq.index = crosstab_num.columns,crosstab_num.index
                        crosstab_freq.to_excel(writer,i)      
                print('tables have been saved at',output_path_pre+'int_tab.xlsx')
            int_plot_features = [c for c in int_features if data[c].nunique()>3]
            if len(int_plot_features)>0:
                with PdfPages(output_path_pre+'int_plot.pdf') as pdf:        
                    for i in tqdm(int_plot_features):        
                        fig = plt.figure(figsize=(8, 6))
                        sns.boxplot(x=target, y=i, data=pd.concat([data[target], data[i]], axis=1),showfliers=False)
                        pdf.savefig(fig)      
                print('plots have been saved at',output_path_pre+'int_plot.pdf')
        #object
        object_features += [c for c in featuresList if data[c].dtypes=='object']
        if len(object_features)>0:
            print('-------------exploring object features----------')
            with pd.ExcelWriter(output_path_pre+'object_tab.xlsx') as writer:
                for i in tqdm(object_features):
                    object_values = list(data.groupby(i).size().sort_values(ascending=False).index[:10])
                    corsstab_data = data[data[i].isin(object_values)].fillna(-999)
                    crosstab_num = corsstab_data[[target,i]].pivot_table(index=[target],columns=i,aggfunc=len,margins=True)
                    crosstab_freq = pd.DataFrame(np.array(crosstab_num)/np.array(crosstab_num.loc['All',:]).reshape((1,-1)))
                    crosstab_freq.columns,crosstab_freq.index = crosstab_num.columns,crosstab_num.index
                    crosstab_freq.to_excel(writer,i)     
            print('tables have been saved at',output_path_pre+'object_tab.xlsx')
            
    else:        
        # float
        pattern = re.compile(r'^.*float.*$')
        float_features = [c for c in featuresList if pattern.match(str(data[c].dtypes)) and data[c].nunique()>2]
        object_features = [c for c in featuresList if pattern.match(str(data[c].dtypes)) and data[c].nunique()<=2]
        if len(float_features)>0:
            print('--------exploring float features--------')
            with PdfPages(output_path_pre+'float_plot.pdf') as pdf:
                for i in tqdm(float_features):
                    fig = plt.figure(figsize=(8, 6))
                    plt.scatter(x=i, y=target, data=pd.concat([data[target], data[i]], axis=1))
                    plt.xlabel(i)
                    plt.ylabel(target)
                    pdf.savefig(fig)   
            print('plots have been saved at',output_path_pre+'float_plot.pdf')
        
        #int
        pattern = re.compile(r'^.*int.*$')
        int_features = [c for c in featuresList if pattern.match(str(data[c].dtypes))]
        print('-------------exploring int features----------')        
        if len(int_features)>0:
            with PdfPages(output_path_pre+'int_plot.pdf') as pdf:   
                for i in tqdm(int_features):
                    if data[i].nunique()>3:
                        fig = plt.figure(figsize=(8, 6))
                        plt.scatter(x=i, y=target, data=pd.concat([data[target], data[i]], axis=1))
                        plt.xlabel(i)
                        plt.ylabel(target)                        
                        pdf.savefig(fig)                        
                    else:
                        fig = plt.figure(figsize=(8, 6))
                        sns.boxplot(x=i, y=target, data=pd.concat([data[target], data[i]], axis=1),showfliers=False)
                        pdf.savefig(fig)
            print('plots have been saved at',output_path_pre+'int_plot.pdf')
            
        #object
        object_features += [c for c in featuresList if data[c].dtypes=='object']
        if len(object_features)>0:
            print('-------------exploring object features----------')
            with PdfPages(output_path_pre+'int_plot.pdf') as pdf: 
                for i in tqdm(object_features):
                    fig = plt.figure(figsize=(8, 6))
                    sns.boxplot(x=i, y=target, data=pd.concat([data[target], data[i]], axis=1),showfliers=False)
                    pdf.savefig(fig) 
            print('tables have been saved at',output_path_pre+'object_plot.pdf')        
