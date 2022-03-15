import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statistics as s
import itertools as it
from math import comb

#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  
#  #  #  #    Printing functions    #  #  #  #  
#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  

def informer(df:pd.DataFrame,target:str) -> dict:
    '''
    Function for initial dataset analysis - 
    returning dict with number of observations, features & classes,
    '''
    df_train = df[~df[target].isna()]
    classes = df_train[target].value_counts()
    tr_n, tr_f = df_train.shape
    te_n, _ = df[df[target].isna()].shape

    return {
        "classes": classes,
        "n_cls": len(classes),
        "features": tr_f,
        "train_obs": tr_n,
        "test_obs": te_n,
        }

def informer_print(info:dict) -> None:
    txt_classes = ", ".join([f"{i}: {n/(info['train_obs']):.1%}" 
                    for i, n in enumerate(info["classes"])])
    print(f'''\n\tFeatures:\t{info['features']}
    \tObservations:\t{info['train_obs']}/{info['test_obs']}
    \ttrain dataset:\t{info['train_obs']/(info['train_obs']+info['test_obs']):.1%}
    \tclasses:\t{txt_classes}''')

def imperfection(df:pd.DataFrame,numeric:bool=False,no_nan:bool=False) -> pd.DataFrame:
    '''
    Quick evaluation of missing values, types and variety of values in table columns
    '''
    def form_series(series:pd.Series,name:str) -> pd.Series:
        series.name = name
        return series
    
    missing_df = form_series(df.isnull().sum() / len(df),"NA Share")
    unique_df = form_series(df.apply(pd.unique),"Values")
    nunique_df = form_series(unique_df.apply(len),"Num. of unique")
    type_df = form_series(df.dtypes,"Type")

    full_df = pd.DataFrame([missing_df,nunique_df,type_df,unique_df]).T
    if not numeric:
        full_df = full_df.where(
            (full_df['Type'] != np.dtype('int64'))\
          & (full_df['Type'] != np.dtype('float64')))\
            .dropna()
    if no_nan:
        full_df = full_df.where(full_df['NA Share'] > 0).dropna()
    return full_df.sort_values(by=['NA Share','Num. of unique'], ascending=False)

def multicolumn(series,cols:int=5) -> pd.DataFrame:
    '''
    Work with pd.Series and slice of pd.DataFrame:
    Function for viewing long data series,
    useful with df.sample().T
    '''
    try:
        title = (series.columns.name, series.columns[0])
    except AttributeError:
        title = (series.dtype, series.name)
    n = len(series)
    rows= int(n//cols)
    if rows*cols < n:
        rows += 1
    col_sep = pd.Series(["│"] * rows, name="│")
    portions = []
    for i in range(cols):
        portion = series.iloc[rows*i:rows*(i+1)]
        if len(portion.index) == 0:
            break
        portion = portion.reset_index()
        portion.columns = [f"col_{i+1}",f"val_{i+1}"]
        portions.append(portion)
        portions.append(col_sep)
    return pd.concat(portions,axis=1).fillna("").style.set_caption("{}: {}".format(*title))

#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  
#  #  #  #    Plotting functions    #  #  #  #  
#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  

# For a visual assessment of the quality of the model, we will use the confusion matrix
yvg = list([(abs(np.cos(x-1)),abs(np.sin(x+1)),abs(np.sin(x*x))) for x in np.arange(-2,0,0.1)])
pal_yvg = sns.color_palette(yvg)
comb_2 = lambda n: comb(n,2)

def plot_some_scatters(X,y,name:str=None,s:float=5):
    n, m = X.shape
    ax_combs = tuple(it.combinations(range(m),r=2))
    k = comb_2(m)
    c = 1
    for i in range(2,6):
        if k % i == 0:
            c = i
    r = int(k / c)

    fig, axs = plt.subplots(r,c,figsize=(s*c,s*r))
    fig.suptitle(name)
    for i, ax in enumerate(np.ravel(axs)):
        x_idx, y_idx = ax_combs[i]
        ax.scatter(X[:,x_idx],X[:,y_idx],c=y)
        ax.set(xlabel=f"axis {x_idx}", ylabel=f"axis {y_idx}")
    plt.show()

def plot_conf_map(conf:np.ndarray,title:str=None) -> None:
    '''
    displaying the confusion matrix graphically with additional coefficients
    '''
    sq = lambda x: int(np.sqrt(x))
    q = sq(len(conf)) == 1 and 2 or sq(len(conf))
    conf_m = np.reshape(np.array(conf),(q,q))
    conf_share = conf_m/np.sum(conf)

    true_ans = np.diag(conf_m)
    p_list = true_ans / np.sum(conf_m, axis=0)
    r_list = true_ans / np.sum(conf_m, axis=1)

    # generalization of F1 metrix since we don't know in advance which class is the main one
    hm = s.harmonic_mean((*r_list,*p_list)) 

    coef_m = np.hstack((
            np.vstack( (conf_share,p_list) ),
            np.append(r_list,hm).reshape(-1,1) ))

    labels = np.asarray([f"{share:.1%}" for share in coef_m.ravel()]).reshape(q+1,q+1)
    labels[-1][-1] = f"H:{hm:.0%}"
    ax = sns.heatmap(coef_m, annot=labels, fmt='', cmap=pal_yvg, vmin=0, vmax=1)

    xlabels = [item.get_text() for item in ax.get_xticklabels()]
    ylabels = [item.get_text() for item in ax.get_yticklabels()]
    xlabels[-1] = 'Rate'
    ylabels[-1] = 'Prediction\nvalue'

    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels)
    
    if title is not None:
        plt.title(title)

    plt.show()

# Images transforms

# This fuctions break our picture
def img_breakdown(img_arr):
    return np.vstack((img_arr[:,:,0],img_arr[:,:,1],img_arr[:,:,2],img_arr[:,:,3]))

def img_breakright(img_arr):
    return np.hstack((img_arr[:,:,0],img_arr[:,:,1],img_arr[:,:,2],img_arr[:,:,3]))

# and these collect them back
def img_set_up(ldown_arr):
    q = ldown_arr.shape[0] // 4
    return np.dstack((ldown_arr[q*0:q*1,:],ldown_arr[q*1:q*2,:],ldown_arr[q*2:q*3,:],ldown_arr[q*3:q*4,:]))

def img_set_left(ldown_arr):
    q = ldown_arr.shape[1] // 4
    return np.dstack((ldown_arr[:,q*0:q*1],ldown_arr[:,q*1:q*2],ldown_arr[:,q*2:q*3],ldown_arr[:,q*3:q*4]))

def img_squeeze(img_arr,trans_class,n_components,conv_pair):
    '''
    Main process - we break data, transform it, 
    then inverse_transform and build in original form
    '''
    conv_break, conv_set = conv_pair
    data = conv_break(img_arr)
    transform = trans_class(n_components=n_components)
    data_trunc = transform.fit_transform(data)
    data_re = transform.inverse_transform(data_trunc)
    data_recol = conv_set(data_re)
    return np.clip(data_recol, 0, 1)

def img_no_ticks(axs):
    '''
    Options to display pictures
    '''
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    for ax in axs:
        ax.set_xticks([], [])
        ax.set_yticks([], [])
        
# of course we want to see the result
def img_framaker(img,n,trans_class): 
    '''
    Output function
    '''
    converting_pairs = ((img_breakdown, img_set_up), (img_breakright, img_set_left))

    fig, ax = plt.subplots(figsize=(10,5))
    # mixed image to reduce the number of artifacts
    new_img = img_squeeze(img,trans_class,n,converting_pairs[0])/2+img_squeeze(img,trans_class,n,converting_pairs[1])/2
    ax.imshow(new_img)
    ax.text(30,50,str(n))
    img_no_ticks([ax])
    
    return fig