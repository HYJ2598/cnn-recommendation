
# coding: utf-8

# # 个性化推荐
# 本项目使用文本卷积神经网络，并使用[`MovieLens`](https://grouplens.org/datasets/movielens/)数据集完成电影推荐的任务。
# 
# 

# 推荐系统在日常的网络应用中无处不在，比如网上购物、网上买书、新闻app、社交网络、音乐网站、电影网站等等等等，有人的地方就有推荐。根据个人的喜好，相同喜好人群的习惯等信息进行个性化的内容推荐。比如打开新闻类的app，因为有了个性化的内容，每个人看到的新闻首页都是不一样的。

# 这当然是很有用的，在信息爆炸的今天，获取信息的途径和方式多种多样，人们花费时间最多的不再是去哪获取信息，而是要在众多的信息中寻找自己感兴趣的，这就是信息超载问题。为了解决这个问题，推荐系统应运而生。

# 协同过滤是推荐系统应用较广泛的技术，该方法搜集用户的历史记录、个人喜好等信息，计算与其他用户的相似度，利用相似用户的评价来预测目标用户对特定项目的喜好程度。优点是会给用户推荐未浏览过的项目，缺点呢，对于新用户来说，没有任何与商品的交互记录和个人喜好等信息，存在冷启动问题，导致模型无法找到相似的用户或商品。

# 为了解决冷启动的问题，通常的做法是对于刚注册的用户，要求用户先选择自己感兴趣的话题、群组、商品、性格、喜欢的音乐类型等信息，比如豆瓣FM：
# <img src="assets/IMG_6242_300.PNG"/>

# ## 下载数据集
# 运行下面代码把[`数据集`](http://files.grouplens.org/datasets/movielens/ml-1m.zip)下载下来

# In[3]:


import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter
import tensorflow as tf

import os
import pickle
import re
from tensorflow.python.ops import math_ops


# In[7]:


from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import zipfile
import hashlib

def _unzip(save_path, _, database_name, data_path):
    """
    Unzip wrapper with the same interface as _ungzip
    :param save_path: The path of the gzip files
    :param database_name: Name of database
    :param data_path: Path to extract to
    :param _: HACK - Used to have to same interface as _ungzip
    """
    print('Extracting {}...'.format(database_name))
    with zipfile.ZipFile(save_path) as zf:
        zf.extractall(data_path)

def download_extract(database_name, data_path):
    """
    Download and extract database
    :param database_name: Database name
    """
    DATASET_ML1M = 'ml-1m'

    if database_name == DATASET_ML1M:
        url = 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'
        hash_code = 'c4d9eecfca2ab87c1945afe126590906'
        extract_path = os.path.join(data_path, 'ml-1m')
        save_path = os.path.join(data_path, 'ml-1m.zip')
        extract_fn = _unzip

    if os.path.exists(extract_path):
        print('Found {} Data'.format(database_name))
        return

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    if not os.path.exists(save_path):
        with DLProgress(unit='B', unit_scale=True, miniters=1, desc='Downloading {}'.format(database_name)) as pbar:
            urlretrieve(
                url,
                save_path,
                pbar.hook)

    assert hashlib.md5(open(save_path, 'rb').read()).hexdigest() == hash_code,         '{} file is corrupted.  Remove the file and try again.'.format(save_path)

    os.makedirs(extract_path)
    try:
        extract_fn(save_path, extract_path, database_name, data_path)
    except Exception as err:
        shutil.rmtree(extract_path)  # Remove extraction folder if there is an error
        raise err

    print('Done.')
    # Remove compressed data
#     os.remove(save_path)

class DLProgress(tqdm):
    """
    Handle Progress Bar while Downloading
    """
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        """
        A hook function that will be called once on establishment of the network connection and
        once after each block read thereafter.
        :param block_num: A count of blocks transferred so far
        :param block_size: Block size in bytes
        :param total_size: The total size of the file. This may be -1 on older FTP servers which do not return
                            a file size in response to a retrieval request.
        """
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


# In[9]:


data_dir = './'
download_extract('ml-1m', data_dir)


# ## 先来看看数据

# 本项目使用的是MovieLens 1M 数据集，包含6000个用户在近4000部电影上的1亿条评论。
# 
# 数据集分为三个文件：用户数据users.dat，电影数据movies.dat和评分数据ratings.dat。

# ### 用户数据
# 分别有用户ID、性别、年龄、职业ID和邮编等字段。
# 
# 数据中的格式：UserID::Gender::Age::Occupation::Zip-code
# 
# - Gender is denoted by a "M" for male and "F" for female
# - Age is chosen from the following ranges:
# 
# 	*  1:  "Under 18"
# 	* 18:  "18-24"
# 	* 25:  "25-34"
# 	* 35:  "35-44"
# 	* 45:  "45-49"
# 	* 50:  "50-55"
# 	* 56:  "56+"
# 
# - Occupation is chosen from the following choices:
# 
# 	*  0:  "other" or not specified
# 	*  1:  "academic/educator"
# 	*  2:  "artist"
# 	*  3:  "clerical/admin"
# 	*  4:  "college/grad student"
# 	*  5:  "customer service"
# 	*  6:  "doctor/health care"
# 	*  7:  "executive/managerial"
# 	*  8:  "farmer"
# 	*  9:  "homemaker"
# 	* 10:  "K-12 student"
# 	* 11:  "lawyer"
# 	* 12:  "programmer"
# 	* 13:  "retired"
# 	* 14:  "sales/marketing"
# 	* 15:  "scientist"
# 	* 16:  "self-employed"
# 	* 17:  "technician/engineer"
# 	* 18:  "tradesman/craftsman"
# 	* 19:  "unemployed"
# 	* 20:  "writer"
# 
# 

# In[14]:


users_title = ['UserID', 'Gender', 'Age', 'OccupationID', 'Zip-code']
users = pd.read_table('./ml-1m/users.dat', sep='::', header=None, names=users_title, engine = 'python')
users.head()


# 可以看出UserID、Gender、Age和Occupation都是类别字段，其中邮编字段是我们不使用的。

# ### 电影数据
# 分别有电影ID、电影名和电影风格等字段。
# 
# 数据中的格式：MovieID::Title::Genres
# 
# - Titles are identical to titles provided by the IMDB (including
# year of release)
# - Genres are pipe-separated and are selected from the following genres:
# 
# 	* Action
# 	* Adventure
# 	* Animation
# 	* Children's
# 	* Comedy
# 	* Crime
# 	* Documentary
# 	* Drama
# 	* Fantasy
# 	* Film-Noir
# 	* Horror
# 	* Musical
# 	* Mystery
# 	* Romance
# 	* Sci-Fi
# 	* Thriller
# 	* War
# 	* Western
# 

# In[15]:


coal_title = ['CoalID','CoalName','Calorific','Territory']
# movies = pd.read_table('./ml-1m/movies.dat', sep='::', header=None, names=movies_title, engine = 'python')
coals = pd.read_csv('F:\Github\jupyter\ml-1m\coal.csv', sep=',', names=coal_title,header=None,  engine = 'python')
coals.head()


# MovieID是类别字段，Title是文本，Genres也是类别字段

# ### 评分数据
# 分别有用户ID、电影ID、评分和时间戳等字段。
# 
# 数据中的格式：UserID::MovieID::Rating::Timestamp
# 
# - UserIDs range between 1 and 6040 
# - MovieIDs range between 1 and 3952
# - Ratings are made on a 5-star scale (whole-star ratings only)
# - Timestamp is represented in seconds since the epoch as returned by time(2)
# - Each user has at least 20 ratings

# In[18]:


ratings_title = ['UserID','MovieID', 'Rating', 'timestamps']
ratings = pd.read_table('F:\\Github\\jupyter\\ml-1m\\ratings.dat', sep='::', header=None, names=ratings_title, engine = 'python')
ratings.head()


# 评分字段Rating就是我们要学习的targets，时间戳字段我们不使用。

# ## 来说说数据预处理

# - UserID、Occupation和MovieID不用变。
# - Gender字段：需要将‘F’和‘M’转换成0和1。
# - Age字段：要转成7个连续数字0~6。
# - Genres字段：是分类字段，要转成数字。首先将Genres中的类别转成字符串到数字的字典，然后再将每个电影的Genres字段转成数字列表，因为有些电影是多个Genres的组合。
# - Title字段：处理方式跟Genres字段一样，首先创建文本到数字的字典，然后将Title中的描述转成数字的列表。另外Title中的年份也需要去掉。
# - Genres和Title字段需要将长度统一，这样在神经网络中方便处理。空白部分用‘< PAD >’对应的数字填充。

# ## 实现数据预处理

# In[19]:


def load_data():
    """
    Load Dataset from File
    """
    #读取User数据
    users_title = ['UserID', 'Gender', 'Age', 'JobID', 'Zip-code']
    users = pd.read_table('./ml-1m/users.dat', sep='::', header=None, names=users_title, engine = 'python')
    users = users.filter(regex='UserID|Gender|Age|JobID')
    users_orig = users.values
    #改变User数据中性别和年龄
    gender_map = {'F':0, 'M':1}
    users['Gender'] = users['Gender'].map(gender_map)

    age_map = {val:ii for ii,val in enumerate(set(users['Age']))}
    users['Age'] = users['Age'].map(age_map)

    #读取coal数据集
    # coal_title = ['MovieID', 'Title', 'Genres']
    coal_title = ['CoalID', 'CoalName', 'Calorific', 'Territory']
    # movies = pd.read_table('./ml-1m/movies.dat', sep='::', header=None, names=movies_title, engine = 'python')
    coals = pd.read_csv('F:\Github\jupyter\ml-1m\coal.csv', sep=',', names=coal_title, header=None, engine='python',dtype={'CoalID':int})
    print(coals.head())
    coals_orig = coals.values
    #将Title中的年份去掉
    # pattern = re.compile(r'^(.*)\((\d+)\)$')
    #
    # title_map = {val:pattern.match(val).group(1) for ii,val in enumerate(set(movies['Title']))}
    # coals['CoalID'] = coals['CoalID'].map(title_map)

    #煤炭名称转数字字典
    CoalName_set = set()

    # for val in coals['Calorific'].str.split('|'):
    #     genres_set.update(val)
    for val in coals['CoalName'].str.split():
        CoalName_set.update(val)
    print(CoalName_set)
    CoalName_set.add('<PAD>')
    CoalName2int = {val:ii for ii, val in enumerate(CoalName_set)}
    print(CoalName2int)
    #将电影类型转成等长数字列表，长度是18
    CoalName_map = {val:[CoalName2int[row] for row in val.split()] for ii,val in enumerate(set(coals['CoalName']))}
    print(CoalName_map)


    for key in CoalName_map:
        for cnt in range(max(CoalName2int.values()) - len(CoalName_map[key])):
            CoalName_map[key].insert(len(CoalName_map[key]) + cnt,CoalName2int['<PAD>'])
    
    coals['CoalName'] = coals['CoalName'].map(CoalName_map)
    print(CoalName_map)
    # 销售地区转数字字典
    Territory_set = set()
    # for val in coals['Calorific'].str.split('|'):
    #     genres_set.update(val)
    # print(coals['Territory'])
    # for val in coals['Territory'].str.split():
    for val in coals['Territory'].str.split():
    # for val in coals['Territory']:
    #     print(val)
    #     if val == None:
    #         val = ""
    #     else:
        if val != None:
            Territory_set.update(val)
            # Territory_set.add(str(val))
    Territory_set.add('<PAD>')
    print(Territory_set)

    Territory2int = {val: ii for ii, val in enumerate(Territory_set)}
    print('----------------')
    print(Territory2int)
    # print(set(coals['Territory']))

    # 将销售地区转成等长数字列表，长度是18
    Territory_map = {val : [Territory2int[row] for row in val.split()] for ii, val in enumerate(set(coals['Territory'])) if val != None}
    # Territory_map = {val: Territory2int[val] for ii, val in enumerate(set(coals['Territory'])) if val != None}

    for key in Territory_map:
        for cnt in range(max(Territory2int.values()) - len(Territory_map[key])):
            Territory_map[key].insert(len(Territory_map[key]) + cnt, Territory2int['<PAD>'])

    coals['Territory'] = coals['Territory'].map(Territory_map)
    print(Territory_map)
    #热量转换
    calorific_map = {val: ii for ii, val in enumerate(set(coals['Calorific']))}
    coals['Calorific'] = coals['Calorific'].map(calorific_map)
    print('-------------------calorific_map-----------------------')
    print(calorific_map)
    print('-------------------calorific_map-----------------------')
    # #电影Title转数字字典
    # title_set = set()
    # for val in movies['Title'].str.split():
    #     title_set.update(val)
    #
    # title_set.add('<PAD>')
    # title2int = {val:ii for ii, val in enumerate(title_set)}
    #
    # #将电影Title转成等长数字列表，长度是15
    # title_count = 15
    # title_map = {val:[title2int[row] for row in val.split()] for ii,val in enumerate(set(movies['Title']))}
    #
    # for key in title_map:
    #     for cnt in range(title_count - len(title_map[key])):
    #         title_map[key].insert(len(title_map[key]) + cnt,title2int['<PAD>'])
    #
    # movies['Title'] = movies['Title'].map(title_map)



    #读取评分数据集
    ratings_title = ['UserID','CoalID', 'ratings', 'timestamps']
    ratings = pd.read_csv('./ml-1m/ratings2.csv', sep=',', header=None, names=ratings_title, engine = 'python')
    ratings = ratings.filter(regex='UserID|CoalID|ratings')
    print(ratings)

    # ratings['UserID','CoalID']=ratings['UserID','CoalID'].apply(pd.to_numeric(),errors='ignore')
    # users['UserID']=users['UserID'].apply(pd.to_numeric(),errors='ignore')
    # coals['CoalID']=coals['CoalID'].apply(pd.to_numeric(),errors='ignore')
    # coals['CoalID']=pd.to_numeric(coals['CoalID'],errors='ignore')
    # coals['coalID']=coals['coalID'].apply(int)
    #合并三个表
    # data=pd.merge(ratings,users)
    # coals1=list.append(coals)
    data = pd.merge(pd.merge(ratings, users), coals)
    print('---------------data-------------------')
    print(data.head)
    print('---------------data-------------------')
    #将数据分成X和y两张表
    target_fields = ['ratings']
    features_pd, targets_pd = data.drop(target_fields, axis=1), data[target_fields]

    print('------------------features------------------')
    print(features_pd.head())
    print('------------------features------------------')

    features = features_pd.values
    targets_values = targets_pd.values
    print('------------------target------------------')
    print(targets_pd)
    print('------------------target------------------')
    
    return Territory2int,Territory_set,CoalName_set,CoalName2int,features, targets_values, ratings, users, coals, data, coals_orig, users_orig


# ### 加载数据并保存到本地

# - title_count：Title字段的长度（15）
# - title_set：Title文本的集合
# - genres2int：电影类型转数字的字典
# - features：是输入X
# - targets_values：是学习目标y
# - ratings：评分数据集的Pandas对象
# - users：用户数据集的Pandas对象
# - movies：电影数据的Pandas对象
# - data：三个数据集组合在一起的Pandas对象
# - movies_orig：没有做数据处理的原始电影数据
# - users_orig：没有做数据处理的原始用户数据

# In[20]:


Territory2int, Territory_set,CoalName_set, CoalName2int, features, targets_values, ratings, users, coals, data, coals_orig, users_orig = load_data()

pickle.dump((Territory2int, Territory_set,CoalName_set, CoalName2int, features, targets_values, ratings, users, coals, data, coals_orig, users_orig), open('preprocess.p', 'wb'))


# ### 预处理后的数据

# In[23]:


users.head()


# In[24]:


coals.head()


# In[25]:


coals.values[0]


# ### 从本地读取数据

# In[ ]:


Territory2int, Territory_set,CoalName_set, CoalName2int, features, targets_values, ratings, users, coals, data, coals_orig, users_orig = pickle.load(open('preprocess.p', mode='rb'))


# ## 模型设计

# <img src="assets/model.001.jpeg"/>

# 通过研究数据集中的字段类型，我们发现有一些是类别字段，通常的处理是将这些字段转成one hot编码，但是像UserID、MovieID这样的字段就会变成非常的稀疏，输入的维度急剧膨胀，这是我们不愿意见到的，毕竟我这小笔记本不像大厂动辄能处理数以亿计维度的输入：）
# 
# 所以在预处理数据时将这些字段转成了数字，我们用这个数字当做嵌入矩阵的索引，在网络的第一层使用了嵌入层，维度是（N，32）和（N，16）。
# 
# 电影类型的处理要多一步，有时一个电影有多个电影类型，这样从嵌入矩阵索引出来是一个（n，32）的矩阵，因为有多个类型嘛，我们要将这个矩阵求和，变成（1，32）的向量。
# 没有用\分割的电影类型处理部分
# 电影名的处理比较特殊，没有使用循环神经网络，而是用了文本卷积网络，下文会进行说明。这里煤炭名和销售地应该是两个类似于电影名的处理
# 
# 从嵌入层索引出特征以后，将各特征传入全连接层，将输出再次传入全连接层，最终分别得到（1，200）的用户特征和电影特征两个特征向量。
# 
# 我们的目的就是要训练出用户特征和电影特征，在实现推荐功能时使用。得到这两个特征以后，就可以选择任意的方式来拟合评分了。我使用了两种方式，一个是上图中画出的将两个特征做向量乘法，将结果与真实评分做回归，采用MSE优化损失。因为本质上这是一个回归问题，另一种方式是，将两个特征作为输入，再次传入全连接层，输出一个值，将输出值回归到真实评分，采用MSE优化损失。
# 
# 实际上第二个方式的MSE loss在0.8附近，第一个方式在1附近，5次迭代的结果。

# ## 文本卷积网络
# 网络看起来像下面这样

# <img src="assets/text_cnn.png"/>
# 图片来自Kim Yoon的论文：[`Convolutional Neural Networks for Sentence Classification`](https://arxiv.org/abs/1408.5882)
# 
# 将卷积神经网络用于文本的文章建议你阅读[`Understanding Convolutional Neural Networks for NLP`](http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/)

# 网络的第一层是词嵌入层，由每一个单词的嵌入向量组成的嵌入矩阵。下一层使用多个不同尺寸（窗口大小）的卷积核在嵌入矩阵上做卷积，窗口大小指的是每次卷积覆盖几个单词。这里跟对图像做卷积不太一样，图像的卷积通常用2x2、3x3、5x5之类的尺寸，而文本卷积要覆盖整个单词的嵌入向量，所以尺寸是（单词数，向量维度），比如每次滑动3个，4个或者5个单词。第三层网络是max pooling得到一个长向量，最后使用dropout做正则化，最终得到了电影Title的特征。
#所以需要有两个来得到Coalname和Territory
# ## 辅助函数

# In[27]:


import tensorflow as tf
import os
import pickle

def save_params(params):
    """
    Save parameters to file
    """
    pickle.dump(params, open('params.p', 'wb'))


def load_params():
    """
    Load parameters from file
    """
    return pickle.load(open('params.p', mode='rb'))


# ## 编码实现

# In[28]:


#嵌入矩阵的维度
embed_dim = 32
#用户ID个数
uid_max = max(features.take(0,1)) + 1 # 6040

#性别个数
gender_max = max(features.take(2,1)) + 1 # 1 + 1 = 2
#年龄类别个数
age_max = max(features.take(3,1)) + 1 # 6 + 1 = 7
#职业个数
job_max = max(features.take(4,1)) + 1# 20 + 1 = 21

#煤炭ID个数
coal_id_max = max(features.take(1,1)) + 1 # 3952
print('------------------coal_id_max------------------------')
print(coal_id_max)
print('------------------coal_id_max------------------------')
#煤炭名长度
coal_name_max = max(CoalName2int.values()) + 1 # 66+ 1 = 67
# coal_name_max1=max(features.take(5,1))
#煤炭名单词个数
coal_word_max = len(CoalName_set) # 5216
# print('------------------coal_name_max------------------------')
# print(coal_name_max1)
# print('------------------coal_name_max------------------------')

#对电影类型嵌入向量做加和操作的标志，考虑过使用mean做平均，但是没实现mean
combiner = "sum"
#销售地区长度
territory_max = max(Territory2int.values())+1
print('------------------territory_max------------------------')
print(territory_max)
print('------------------territory_max------------------------')
#销售地区单词个数
territory_word_max=max(Territory_set)
#发热量
calorific_max=max(features.take(6,1))+1
# calorific_max= max(map(int,calorific_maxlist))
# for i, v in enumerate(calorific_maxlist): calorific_maxlist[i] = int(v)
print('------------------calorific_max------------------------')
print(calorific_max)
print('------------------calorific_max------------------------')
#电影名长度
# sentences_size = title_count # = 15
#文本卷积滑动窗口，分别滑动2, 3, 4, 5个单词
window_sizes = {2, 3, 4, 5}
#文本卷积核数量
filter_num = 8

#电影ID转下标的字典，数据集中电影ID跟下标不一致，比如第5行的数据电影ID不一定是5
coalid2idx = {val[0]:i for i, val in enumerate(coals.values)}


# ### 超参

# In[29]:


# Number of Epochs
num_epochs = 5
# Batch Size
batch_size = 256

dropout_keep = 0.5
# Learning Rate
learning_rate = 0.0001
# Show stats for every n number of batches
show_every_n_batches = 20

save_dir = './save'


# ### 输入

# 定义输入的占位符

# In[30]:


def get_inputs():

    uid = tf.placeholder(tf.int32, [None, 1], name="uid")
    user_gender = tf.placeholder(tf.int32, [None, 1], name="user_gender")
    user_age = tf.placeholder(tf.int32, [None, 1], name="user_age")
    user_job = tf.placeholder(tf.int32, [None, 1], name="user_job")
    
    coal_id = tf.placeholder(tf.int32, [None, 1], name="coal_id")
    coal_name = tf.placeholder(tf.int32, [None, 66], name="coal_name")
    calorific = tf.placeholder(tf.int32,[None,1],name="coal_calorific")
    territory = tf.placeholder(tf.int32, [None, 100], name="coal_territory")
    targets = tf.placeholder(tf.int32, [None, 1], name="targets")
    LearningRate = tf.placeholder(tf.float32, name = "LearningRate")
    dropout_keep_prob = tf.placeholder(tf.float32, name = "dropout_keep_prob")
    return uid, user_gender, user_age, user_job, coal_id, coal_name,calorific, territory, targets, LearningRate, dropout_keep_prob


# ## 构建神经网络

# #### 定义User的嵌入矩阵

# In[31]:


def get_user_embedding(uid, user_gender, user_age, user_job):
    with tf.name_scope("user_embedding"):
        uid_embed_matrix = tf.Variable(tf.random_uniform([uid_max, embed_dim], -1, 1), name = "uid_embed_matrix")#返回用户数*32值在-1，1之间
        uid_embed_layer = tf.nn.embedding_lookup(uid_embed_matrix, uid, name = "uid_embed_layer")#在uid_embed_matrix中查找序号为uid的也是在-1,1之间
    
        gender_embed_matrix = tf.Variable(tf.random_uniform([gender_max, embed_dim // 2], -1, 1), name= "gender_embed_matrix")
        gender_embed_layer = tf.nn.embedding_lookup(gender_embed_matrix, user_gender, name = "gender_embed_layer")
        
        age_embed_matrix = tf.Variable(tf.random_uniform([age_max, embed_dim // 2], -1, 1), name="age_embed_matrix")
        age_embed_layer = tf.nn.embedding_lookup(age_embed_matrix, user_age, name="age_embed_layer")
        
        job_embed_matrix = tf.Variable(tf.random_uniform([job_max, embed_dim // 2], -1, 1), name = "job_embed_matrix")
        job_embed_layer = tf.nn.embedding_lookup(job_embed_matrix, user_job, name = "job_embed_layer")
    return uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer


# #### 将User的嵌入矩阵一起全连接生成User的特征

# In[32]:


def get_user_feature_layer(uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer):
    with tf.name_scope("user_fc"):
        #第一层全连接
        uid_fc_layer = tf.layers.dense(uid_embed_layer, embed_dim, name = "uid_fc_layer", activation=tf.nn.relu)#dense:全连接层 相当于添加一层
        gender_fc_layer = tf.layers.dense(gender_embed_layer, embed_dim, name = "gender_fc_layer", activation=tf.nn.relu)
        age_fc_layer = tf.layers.dense(age_embed_layer, embed_dim, name ="age_fc_layer", activation=tf.nn.relu)
        job_fc_layer = tf.layers.dense(job_embed_layer, embed_dim, name = "job_fc_layer", activation=tf.nn.relu)
        
        #第二层全连接
        user_combine_layer = tf.concat([uid_fc_layer, gender_fc_layer, age_fc_layer, job_fc_layer], 2)  #(?, 1, 128)
        user_combine_layer = tf.contrib.layers.fully_connected(user_combine_layer, 200, tf.tanh)  #(?, 1, 200)
    
        user_combine_layer_flat = tf.reshape(user_combine_layer, [-1, 200])
    return user_combine_layer, user_combine_layer_flat


# #### 定义Movie ID的嵌入矩阵

# In[33]:


def get_coal_id_embed_layer(coal_id):
    with tf.name_scope("coalID"):
        coal_id_embed_matrix = tf.Variable(tf.random_uniform([coal_id_max, embed_dim], -1, 1), name = "coal_id_embed_matrix")
        coal_id_embed_layer = tf.nn.embedding_lookup(coal_id_embed_matrix, coal_id, name = "coal_id_embed_layer")
    return coal_id_embed_layer

def get_calorific_layer(calorific):
    with tf.name_scope("calorific"):
        calorific_embed_matrix = tf.Variable(tf.random_uniform([calorific_max, embed_dim//2], -1, 1), name = "calorific_embed_matrix")
        calorific_embed_layer = tf.nn.embedding_lookup(calorific_embed_matrix, calorific, name = "calorific_embed_layer")
    return calorific_embed_layer
# #### 对电影类型的多个嵌入向量做加和

# In[34]:


def get_coal_name_layers(coal_name):
    with tf.name_scope("coal_name_layers"):
        #从嵌入矩阵得到煤炭名对应各个单词的嵌入向量
        coal_name_embed_matrix = tf.Variable(tf.random_uniform([coal_name_max, embed_dim], -1, 1), name = "coal_name_embed_matrix")
        coal_name_embed_layer = tf.nn.embedding_lookup(coal_name_embed_matrix, coal_name, name = "coal_name_embed_layer")
        # if combiner == "sum":
        #     movie_categories_embed_layer = tf.reduce_sum(movie_categories_embed_layer, axis=1, keep_dims=True)
    #     elif combiner == "mean":
        coal_name_embed_layer_expand = tf.expand_dims(coal_name_embed_layer,-1)

    #对文本嵌入层使用不同尺寸的卷积核做卷积和最大池化
    pool_layer_lst = []
    for window_size in window_sizes:
        with tf.name_scope("coal_txt_conv_maxpool_{}".format(window_size)):
            filter_weights = tf.Variable(tf.truncated_normal([window_size, embed_dim, 1, filter_num], stddev=0.1),
                                             name="filter_weights")
            filter_bias = tf.Variable(tf.constant(0.1, shape=[filter_num]), name="filter_bias")

            conv_layer = tf.nn.conv2d(coal_name_embed_layer_expand, filter_weights, [1, 1, 1, 1], padding="VALID",
                                          name="conv_layer")
            relu_layer = tf.nn.relu(tf.nn.bias_add(conv_layer, filter_bias), name="relu_layer")#leakyrele   elu  selu

            maxpool_layer = tf.nn.max_pool(relu_layer, [1, coal_name_max - window_size , 1, 1], [1, 1, 1, 1],
                                               padding="VALID", name="maxpool_layer")
            pool_layer_lst.append(maxpool_layer)

    # Dropout层
    with tf.name_scope("pool_dropout"):
            pool_layer = tf.concat(pool_layer_lst, 3, name="pool_layer")
            max_num = len(window_sizes) * filter_num
            pool_layer_flat = tf.reshape(pool_layer, [-1, 1, max_num], name="pool_layer_flat")

            dropout_layer1 = tf.nn.dropout(pool_layer_flat, dropout_keep_prob, name="dropout_layer")
    return pool_layer_flat, dropout_layer1

def get_coal_territory_layers(territory):
    with tf.name_scope("coal_territory_layers"):
        #从嵌入矩阵得到煤炭名对应各个单词的嵌入向量
        coal_territory_embed_matrix = tf.Variable(tf.random_uniform([territory_max, embed_dim], -1, 1), name = "coal_territory_embed_matrix")
        coal_territory_embed_layer = tf.nn.embedding_lookup(coal_territory_embed_matrix, territory, name = "coal_territory_embed_layer")
        # if combiner == "sum":
        #     movie_categories_embed_layer = tf.reduce_sum(movie_categories_embed_layer, axis=1, keep_dims=True)
    #     elif combiner == "mean":
        coal_territory_embed_layer_expand = tf.expand_dims(coal_territory_embed_layer,-1)

    #对文本嵌入层使用不同尺寸的卷积核做卷积和最大池化
    pool_layer_lst = []
    for window_size in window_sizes:
        with tf.name_scope("coal_txt_conv_maxpool_{}".format(window_size)):
            filter_weights = tf.Variable(tf.truncated_normal([window_size, embed_dim, 1, filter_num], stddev=0.1),
                                             name="filter_weights")
            filter_bias = tf.Variable(tf.constant(0.1, shape=[filter_num]), name="filter_bias")

            conv_layer = tf.nn.conv2d(coal_territory_embed_layer_expand, filter_weights, [1, 1, 1, 1], padding="VALID",
                                          name="conv_layer")
            print(tf.shape(conv_layer))
            #conv_layer2=
            relu_layer = tf.nn.relu(tf.nn.bias_add(conv_layer, filter_bias), name="relu_layer")

            maxpool_layer = tf.nn.max_pool(relu_layer, [1, territory_max - window_size , 1, 1], [1, 1, 1, 1],
                                               padding="VALID", name="maxpool_layer")
            pool_layer_lst.append(maxpool_layer)

    # Dropout层
    with tf.name_scope("pool_dropout"):
            pool_layer = tf.concat(pool_layer_lst, 3, name="pool_layer")
            max_num = len(window_sizes) * filter_num
            pool_layer_flat = tf.reshape(pool_layer, [-1, 1, max_num], name="pool_layer_flat")

            dropout_layer2 = tf.nn.dropout(pool_layer_flat, dropout_keep_prob, name="dropout_layer")
    return pool_layer_flat, dropout_layer2





# #### Movie Title的文本卷积网络实现

# In[35]:


# def get_coal_cnn_layer(movie_titles):
#     #从嵌入矩阵中得到电影名对应的各个单词的嵌入向量
#     with tf.name_scope("movie_embedding"):
#         movie_title_embed_matrix = tf.Variable(tf.random_uniform([movie_title_max, embed_dim], -1, 1), name = "movie_title_embed_matrix")
#         movie_title_embed_layer = tf.nn.embedding_lookup(movie_title_embed_matrix, movie_titles, name = "movie_title_embed_layer")
#         movie_title_embed_layer_expand = tf.expand_dims(movie_title_embed_layer, -1)
#
#     #对文本嵌入层使用不同尺寸的卷积核做卷积和最大池化
#     pool_layer_lst = []
#     for window_size in window_sizes:
#         with tf.name_scope("movie_txt_conv_maxpool_{}".format(window_size)):
#             filter_weights = tf.Variable(tf.truncated_normal([window_size, embed_dim, 1, filter_num],stddev=0.1),name = "filter_weights")
#             filter_bias = tf.Variable(tf.constant(0.1, shape=[filter_num]), name="filter_bias")
#
#             conv_layer = tf.nn.conv2d(movie_title_embed_layer_expand, filter_weights, [1,1,1,1], padding="VALID", name="conv_layer")
#             relu_layer = tf.nn.relu(tf.nn.bias_add(conv_layer,filter_bias), name ="relu_layer")
#
#             maxpool_layer = tf.nn.max_pool(relu_layer, [1,sentences_size - window_size + 1 ,1,1], [1,1,1,1], padding="VALID", name="maxpool_layer")
#             pool_layer_lst.append(maxpool_layer)
#
#     #Dropout层
#     with tf.name_scope("pool_dropout"):
#         pool_layer = tf.concat(pool_layer_lst, 3, name ="pool_layer")
#         max_num = len(window_sizes) * filter_num
#         pool_layer_flat = tf.reshape(pool_layer , [-1, 1, max_num], name = "pool_layer_flat")
#
#         dropout_layer = tf.nn.dropout(pool_layer_flat, dropout_keep_prob, name = "dropout_layer")
#     return pool_layer_flat, dropout_layer


# #### 将Movie的各个层一起做全连接

# In[36]:


def get_coal_feature_layer(coal_id_embed_layer, calorific_embed_layer, dropout_layer1,dropout_layer2):
    with tf.name_scope("coal_fc"):
        #第一层全连接
        coal_id_fc_layer = tf.layers.dense(coal_id_embed_layer, embed_dim, name = "coal_id_fc_layer", activation=tf.nn.relu)
        calorific_fc_layer = tf.layers.dense(calorific_embed_layer, embed_dim, name = "calorific_fc_layer", activation=tf.nn.relu)
    
        #第二层全连接
        coal_combine_layer = tf.concat([coal_id_fc_layer, calorific_fc_layer, dropout_layer1,dropout_layer2], 2)  #(?, 1, 96)#该tf.concat是矩阵在第二个维度上的连接操作
        coal_combine_layer = tf.contrib.layers.fully_connected(coal_combine_layer, 200, tf.tanh)  #(?, 1, 200)
    
        coal_combine_layer_flat = tf.reshape(coal_combine_layer, [-1, 200])
    return coal_combine_layer, coal_combine_layer_flat


# ## 构建计算图

# In[37]:


tf.reset_default_graph()
train_graph = tf.Graph()
with train_graph.as_default():
    #获取输入占位符
    uid, user_gender, user_age, user_job, coal_id, coal_name, calorific,territory, targets, lr, dropout_keep_prob = get_inputs()
    #获取User的4个嵌入向量
    uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer = get_user_embedding(uid, user_gender, user_age, user_job)
    #得到用户特征
    user_combine_layer, user_combine_layer_flat = get_user_feature_layer(uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer)
    #获取煤炭ID的嵌入向量
    coal_id_embed_layer = get_coal_id_embed_layer(coal_id)
    #获取发热量的嵌入向量
    calorific_embed_layer = get_calorific_layer(calorific)
    #获取煤炭名的特征向量
    pool_layer_flat1, dropout_layer1 = get_coal_name_layers(coal_name)
    # 获取销售地区的特征向量
    pool_layer_flat2, dropout_layer2 = get_coal_territory_layers(territory)
    #得到电影特征
    coal_combine_layer, coal_combine_layer_flat = get_coal_feature_layer(coal_id_embed_layer,
                                                                                calorific_embed_layer,
                                                                                dropout_layer1,
                                                                                dropout_layer2
                                                                            )
    #计算出评分，要注意两个不同的方案，inference的名字（name值）是不一样的，后面做推荐时要根据name取得tensor
    with tf.name_scope("inference"):
        # 将用户特征和电影特征作为输入，经过全连接，输出一个值的方案
        # inference_layer = tf.concat([user_combine_layer_flat, coal_combine_layer_flat], 1)  #(?, 200)
        # inference = tf.layers.dense(inference_layer, 1,
        #                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
        #                             kernel_regularizer=tf.nn.l2_loss, name="inference")
        # 简单的将用户特征和电影特征做矩阵乘法得到一个预测评分
       # inference = tf.matmul(user_combine_layer_flat, tf.transpose(movie_combine_layer_flat))
        inference = tf.reduce_sum(user_combine_layer_flat * coal_combine_layer_flat, axis=1)
        inference = tf.expand_dims(inference, axis=1)

    with tf.name_scope("loss"):
        # MSE损失，将计算值回归到评分
        cost = tf.losses.mean_squared_error(targets, inference )
        loss = tf.reduce_mean(cost)

    with tf.name_scope("accuracy"):
        acc = tf.metrics.accuracy(targets,inference)
        accuracy = tf.reduce_mean(acc)
    # 优化损失 
#     train_op = tf.train.AdamOptimizer(lr).minimize(loss)  #cost
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(lr)
    gradients = optimizer.compute_gradients(loss)  #cost
    train_op = optimizer.apply_gradients(gradients, global_step=global_step)
    


# In[41]:


inference


# ## 取得batch

# In[38]:


def get_batches(Xs, ys, batch_size):
    for start in range(0, len(Xs), batch_size):
        end = min(start + batch_size, len(Xs))
        yield Xs[start:end], ys[start:end]


# ## 训练网络

# In[39]:


# get_ipython().run_line_magic('matplotlib', 'inline')
# get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
import matplotlib.pyplot as plt
import time
import datetime

losses = {'train':[], 'test':[]}
accuracies = {'train':[], 'test':[]}


with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    #搜集数据给tensorBoard用
    # Keep track of gradient values and sparsity
    grad_summaries = []
    for g, v in gradients:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name.replace(':', '_')), g)
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name.replace(':', '_')), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)
        
    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))
     
    # Summaries for loss and accuracy
    loss_summary = tf.summary.scalar("loss", loss)
    accuracy_summary = tf.summary.scalar("accuracy",accuracy)

    # Train Summaries
    train_summary_op = tf.summary.merge([loss_summary, accuracy_summary,grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)


    # Inference summaries
    inference_summary_op = tf.summary.merge([loss_summary,accuracy_summary])
    inference_summary_dir = os.path.join(out_dir, "summaries", "inference")
    inference_summary_writer = tf.summary.FileWriter(inference_summary_dir, sess.graph)

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    for epoch_i in range(num_epochs):
        
        #将数据集分成训练集和测试集，随机种子不固定
        train_X,test_X, train_y, test_y = train_test_split(features,
                                                           targets_values,
                                                           test_size = 0.2,  
                                                           random_state = 0)  
        
        train_batches = get_batches(train_X, train_y, batch_size)
        test_batches = get_batches(test_X, test_y, batch_size)
    
        #训练的迭代，保存训练损失
        for batch_i in range(len(train_X) // batch_size):
            x, y = next(train_batches)

            coal_name1 = np.zeros([batch_size, 66])
            for i in range(batch_size):
                coal_name1[i] = x.take(5,1)[i]
            # print('--------------x.take(6,1)---------------')
            # print(x.take(6,1))
            # print('--------------x.take(6,1)---------------')
            # calorific1 = np.zeros([batch_size,1])
            # for i in range(batch_size):
            #     calorific1[i]=x.take(6,1)[i]
                # print(calorific1[i])
            territory1 = np.zeros([batch_size, 100])
            for i in range(batch_size):
                territory1[i] = x.take(7,1)[i]

            feed = {
                uid: np.reshape(x.take(0,1), [batch_size, 1]),
                user_gender: np.reshape(x.take(2,1), [batch_size, 1]),
                user_age: np.reshape(x.take(3,1), [batch_size, 1]),
                user_job: np.reshape(x.take(4,1), [batch_size, 1]),
                coal_id: np.reshape(x.take(1,1), [batch_size, 1]),
                calorific:np.reshape(x.take(6,1),[batch_size,1]),
                coal_name: coal_name1,  #x.take(6,1)
                territory: territory1,  #x.take(5,1)
                targets: np.reshape(y, [batch_size, 1]),
                dropout_keep_prob: dropout_keep, #dropout_keep
                lr: learning_rate}

            step, train_loss,train_accuracy, summaries, _ = sess.run([global_step, loss,accuracy, train_summary_op, train_op], feed)  #cost
            losses['train'].append(train_loss)
            accuracies['train'].append(train_accuracy)
            train_summary_writer.add_summary(summaries, step)  #
            
            # Show every <show_every_n_batches> batches
            if (epoch_i * (len(train_X) // batch_size) + batch_i) % show_every_n_batches == 0:
                time_str = datetime.datetime.now().isoformat()
                print('{}: Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                    time_str,
                    epoch_i,
                    batch_i,
                    (len(train_X) // batch_size),
                    train_loss))
                
        #使用测试数据的迭代
        for batch_i  in range(len(test_X) // batch_size):
            x, y = next(test_batches)
            
            coal_name2 = np.zeros([batch_size, 66])
            for i in range(batch_size):
                coal_name2[i] = x.take(5,1)[i]

            territory2 = np.zeros([batch_size, 100])
            for i in range(batch_size):
                territory2[i] = x.take(7,1)[i]

            # calorific2 = np.zeros([batch_size,1])
            # for i in range(batch_size):
            #     calorific2[i] = x.take(6,1)[i]
            feed = {
                uid: np.reshape(x.take(0,1), [batch_size, 1]),
                user_gender: np.reshape(x.take(2,1), [batch_size, 1]),
                user_age: np.reshape(x.take(3,1), [batch_size, 1]),
                user_job: np.reshape(x.take(4,1), [batch_size, 1]),
                coal_id: np.reshape(x.take(1,1), [batch_size, 1]),
                calorific: np.reshape(x.take(6,1),[batch_size,1]),  #x.take(6,1)
                coal_name: coal_name2,
                territory: territory2,
                targets: np.reshape(y, [batch_size, 1]),
                dropout_keep_prob: 1,
                lr: learning_rate}
            
            step, test_loss, summaries = sess.run([global_step, loss, inference_summary_op], feed)  #cost

            #保存测试损失
            losses['test'].append(test_loss)
            inference_summary_writer.add_summary(summaries, step)  #

            time_str = datetime.datetime.now().isoformat()
            if (epoch_i * (len(test_X) // batch_size) + batch_i) % show_every_n_batches == 0:
                print('{}: Epoch {:>3} Batch {:>4}/{}   test_loss = {:.3f}'.format(
                    time_str,
                    epoch_i,
                    batch_i,
                    (len(test_X) // batch_size),
                    test_loss))

    # Save Model
    saver.save(sess, save_dir)  #, global_step=epoch_i
    print('Model Trained and Saved')


# ## 在 TensorBoard 中查看可视化结果

# tensorboard --logdir /PATH_TO_CODE/runs/1513402825/summaries/
# 
# <img src="assets/loss.png"/>

# ## 保存参数
# 保存`save_dir` 在生成预测时使用。

# In[40]:


save_params((save_dir))

load_dir = load_params()


# ## 显示训练Loss

# In[42]:


plt.plot(losses['train'], label='Training loss')
plt.legend()
_ = plt.ylim()

plt.plot(accuracies['train'], label='Training accuracy')
plt.legend()
_ = plt.ylim()


# ## 显示测试Loss
# 迭代次数再增加一些，下降的趋势会明显一些

# In[43]:


plt.plot(losses['test'], label='Test loss')
plt.legend()
_ = plt.ylim()

plt.plot(accuracies['test'], label='Test accuracy')
plt.legend()
_ = plt.ylim()

# ## 获取 Tensors
# 使用函数 [`get_tensor_by_name()`](https://www.tensorflow.org/api_docs/python/tf/Graph#get_tensor_by_name)从 `loaded_graph` 中获取tensors，后面的推荐功能要用到。

# In[44]:


def get_tensors(loaded_graph):

    uid = loaded_graph.get_tensor_by_name("uid:0")
    user_gender = loaded_graph.get_tensor_by_name("user_gender:0")
    user_age = loaded_graph.get_tensor_by_name("user_age:0")
    user_job = loaded_graph.get_tensor_by_name("user_job:0")
    coal_id = loaded_graph.get_tensor_by_name("coal_id:0")
    coal_name = loaded_graph.get_tensor_by_name("coal_name:0")
    calorific = loaded_graph.get_tensor_by_name("coal_calorific:0")
    territory = loaded_graph.get_tensor_by_name("coal_territory:0")
    targets = loaded_graph.get_tensor_by_name("targets:0")
    dropout_keep_prob = loaded_graph.get_tensor_by_name("dropout_keep_prob:0")
    lr = loaded_graph.get_tensor_by_name("LearningRate:0")
    #两种不同计算预测评分的方案使用不同的name获取tensor inference
#     inference = loaded_graph.get_tensor_by_name("inference/inference/BiasAdd:0")
    inference = loaded_graph.get_tensor_by_name("inference/ExpandDims:0") # 之前是MatMul:0 因为inference代码修改了 这里也要修改 感谢网友 @清歌 指出问题
    coal_combine_layer_flat = loaded_graph.get_tensor_by_name("coal_fc/Reshape:0")
    user_combine_layer_flat = loaded_graph.get_tensor_by_name("user_fc/Reshape:0")
    return uid, user_gender, user_age, user_job, coal_id, coal_name,calorific,territory,  targets, lr, dropout_keep_prob, inference, coal_combine_layer_flat, user_combine_layer_flat


# ## 指定用户和电影进行评分
# 这部分就是对网络做正向传播，计算得到预测的评分

# In[49]:


def rating_movie(user_id_val, coal_id_val):
    loaded_graph = tf.Graph()  #
    with tf.Session(graph=loaded_graph) as sess:  #
        # Load saved model
        loader = tf.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)
    
        # Get Tensors from loaded model
        uid, user_gender, user_age, user_job, coal_id, coal_name, calorific,territory, targets, lr, dropout_keep_prob, inference,_, __ = get_tensors(loaded_graph)  #loaded_graph
    

        coal_name3 = np.zeros([1, 66])#电影类型转成了18
        # for i in range(66):
        coal_name3[0] = coals.values[coalid2idx[coal_id_val]][1]#movieid2++++++++++++++++++++++++++++++++++++++++++...................................idx是电影id转下标的字典
    
        territory3 = np.zeros([1, 100])#sentences_size=title_count=15
        # for i in range(100):
        territory3[0] = coals.values[coalid2idx[coal_id_val]][3]#movieid2idx是电影id转下标的字典

        # calorific = np.zeros([1,1])
        # calorific[0] = coals.values[coalid2idx[coal_id_val]][2]
        feed = {
              uid: np.reshape(users.values[user_id_val-1][0], [1, 1]),
              user_gender: np.reshape(users.values[user_id_val-1][1], [1, 1]),
              user_age: np.reshape(users.values[user_id_val-1][2], [1, 1]),
              user_job: np.reshape(users.values[user_id_val-1][3], [1, 1]),
              coal_id: np.reshape(coals.values[coalid2idx[coal_id_val]][0], [1, 1]),
              coal_name: coal_name3,  #x.take(6,1)
              calorific:np.reshape(coals.values[coalid2idx[coal_id_val]][2],[1,1]),
              territory: territory3,  #x.take(5,1)
              dropout_keep_prob: 1}
    
        # Get Prediction
        inference_val = sess.run([inference], feed)  
    
        return (inference_val)


# In[54]:


print(rating_movie(234, 100))


# ## 生成Movie特征矩阵
# 将训练好的电影特征组合成电影特征矩阵并保存到本地

# In[55]:


loaded_graph = tf.Graph()  #
coal_matrics = []
with tf.Session(graph=loaded_graph) as sess:  #
    # Load saved model
    loader = tf.train.import_meta_graph(load_dir + '.meta')
    loader.restore(sess, load_dir)

    # Get Tensors from loaded model
    uid, user_gender, user_age, user_job, coal_id, coal_name, calorific,territory, targets, lr, dropout_keep_prob, _, coal_combine_layer_flat, __ = get_tensors(loaded_graph)  #loaded_graph

    for item in coals.values:
        coal_name4 = np.zeros([1, 66])
        coal_name4[0] = item.take(1)

        territory4 = np.zeros([1, 100])
        territory4[0] = item.take(3)

        feed = {
            coal_id: np.reshape(item.take(0), [1, 1]),
            coal_name: coal_name4,  #x.take(6,1)
            calorific:np.reshape(item.take(2), [1, 1]),
            territory: territory4,  #x.take(5,1)
            dropout_keep_prob: 1}

        coal_combine_layer_flat_val = sess.run([coal_combine_layer_flat], feed)
        coal_matrics.append(coal_combine_layer_flat_val)
        #print('----------------coal_matrix--------------------')
       # print(coal_matrics)
        #print('----------------coal_matrix--------------------')

pickle.dump((np.array(coal_matrics).reshape(-1, 200)), open('coal_matrics.p', 'wb'))
coal_matrics = pickle.load(open('coal_matrics.p', mode='rb'))


# In[57]:


coal_matrics = pickle.load(open('coal_matrics.p', mode='rb'))


# ## 生成User特征矩阵
# 将训练好的用户特征组合成用户特征矩阵并保存到本地

# In[56]:


loaded_graph = tf.Graph()  #
users_matrics = []
with tf.Session(graph=loaded_graph) as sess:  #
    # Load saved model
    loader = tf.train.import_meta_graph(load_dir + '.meta')
    loader.restore(sess, load_dir)

    # Get Tensors from loaded model
    uid, user_gender, user_age, user_job, coal_id, coal_name, calorific,territory, targets, lr, dropout_keep_prob, _, __,user_combine_layer_flat = get_tensors(loaded_graph)  #loaded_graph

    for item in users.values:

        feed = {
            uid: np.reshape(item.take(0), [1, 1]),
            user_gender: np.reshape(item.take(1), [1, 1]),
            user_age: np.reshape(item.take(2), [1, 1]),
            user_job: np.reshape(item.take(3), [1, 1]),
            dropout_keep_prob: 1}

        user_combine_layer_flat_val = sess.run([user_combine_layer_flat], feed)  
        users_matrics.append(user_combine_layer_flat_val)
        #print('---------------users_matrics---------------')
       # print(users_matrics)
        #print('---------------users_matrics---------------')
pickle.dump((np.array(users_matrics).reshape(-1, 200)), open('users_matrics.p', 'wb'))
users_matrics = pickle.load(open('users_matrics.p', mode='rb'))


# In[58]:


users_matrics = pickle.load(open('users_matrics.p', mode='rb'))


# ## 开始推荐电影
# 使用生产的用户特征矩阵和电影特征矩阵做电影推荐

# ### 推荐同类型的电影
# 思路是计算当前看的电影特征向量与整个电影特征矩阵的余弦相似度，取相似度最大的top_k个，这里加了些随机选择在里面，保证每次的推荐稍稍有些不同。

# In[59]:


def recommend_same_type_movie(coal_id_val, top_k = 20):
    
    loaded_graph = tf.Graph()  #
    with tf.Session(graph=loaded_graph) as sess:  #
        # Load saved model
        loader = tf.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)
        
        norm_coal_matrics = tf.sqrt(tf.reduce_sum(tf.square(coal_matrics), 1, keep_dims=True))
        normalized_coal_matrics = coal_matrics / norm_coal_matrics#这种煤炭特征矩阵/相对煤炭特征

        #推荐同类型的电影
        probs_embeddings = (coal_matrics[coalid2idx[coal_id_val]]).reshape([1, 200])#把煤炭矩阵转化成1行200列的
        probs_similarity = tf.matmul(probs_embeddings, tf.transpose(normalized_coal_matrics))#1行200列的当前煤炭特征向量乘整个煤炭特征矩阵
        # print('----------维度-----------------------')
      #  print(probs_similarity)
        # print('----------维度-----------------------')
        sim = (probs_similarity.eval())
       # print(sim)
    #     results = (-sim[0]).argsort()[0:top_k]
    #     print(results)
        
        print("您购买的煤种是：{}".format(coals_orig[coalid2idx[coal_id_val]]))
        print("以下是给您的推荐：")
        p = np.squeeze(sim)
        p[np.argsort(p)[:-top_k]] = 0
        # print('----------p-----------------------')
        # print(p)
        # print('----------p-----------------------')
        p = p / np.sum(p)

        results = set()
        while len(results) != 5:
            c = np.random.choice(201, 1, p=p)[0]
            results.add(c)
        for val in (results):
            print(val)
            print(coals_orig[val])
        
        return results


# In[60]:


recommend_same_type_movie(100, 10)


# ### 推荐您喜欢的电影
# 思路是使用用户特征向量与电影特征矩阵计算所有电影的评分，取评分最高的top_k个，同样加了些随机选择部分。

# In[61]:


def recommend_your_favorite_movie(user_id_val, top_k = 10):

    loaded_graph = tf.Graph()  #
    with tf.Session(graph=loaded_graph) as sess:  #
        # Load saved model
        loader = tf.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)

        #推荐您喜欢的电影
        probs_embeddings = (users_matrics[user_id_val-1]).reshape([1, 200])

        probs_similarity = tf.matmul(probs_embeddings, tf.transpose(coal_matrics))#tf.transpose更换维度序列tf.matmul矩阵a*b而tf.multiply矩阵各元素相乘
        sim = (probs_similarity.eval())#.eval返回表达式的值
    #     print(sim.shape)
    #     results = (-sim[0]).argsort()[0:top_k]
    #     print(results)
        
    #     sim_norm = probs_norm_similarity.eval()
    #     print((-sim_norm[0]).argsort()[0:top_k])
    
        print("以下是给您的推荐：")
        p = np.squeeze(sim)#np.squeeze把shape中为1的维度去掉
        p[np.argsort(p)[:-top_k]] = 0#np.argsort(p)将矩阵按照axis排序返回排序后下标
        p = p / np.sum(p)
        results = set()
        while len(results) != 5:
            c = np.random.choice(201, 1, p=p)[0]
            results.add(c)
        for val in (results):
            print(val)
            print(coals_orig[val])

        return results


# In[62]:


recommend_your_favorite_movie(234, 10)


# ### 看过这个电影的人还看了（喜欢）哪些电影
# - 首先选出喜欢某个电影的top_k个人，得到这几个人的用户特征向量。
# - 然后计算这几个人对所有电影的评分
# - 选择每个人评分最高的电影作为推荐
# - 同样加入了随机选择

# In[63]:


import random

def recommend_other_favorite_movie(coal_id_val, top_k = 20):
    loaded_graph = tf.Graph()  #
    with tf.Session(graph=loaded_graph) as sess:  #
        # Load saved model
        loader = tf.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)

        probs_coal_embeddings = (coal_matrics[coalid2idx[coal_id_val]]).reshape([1, 200])
        probs_user_favorite_similarity = tf.matmul(probs_coal_embeddings, tf.transpose(users_matrics))
        favorite_user_id = np.argsort(probs_user_favorite_similarity.eval())[0][-top_k:]
    #     print(normalized_users_matrics.eval().shape)
    #     print(probs_user_favorite_similarity.eval()[0][favorite_user_id])
    #     print(favorite_user_id.shape)
    
        print("您购买的煤种是：{}".format(coals_orig[coalid2idx[coal_id_val]]))
        
        print("买过这个煤种的人是：{}".format(users_orig[favorite_user_id-1]))
        probs_users_embeddings = (users_matrics[favorite_user_id-1]).reshape([-1, 200])
        probs_similarity = tf.matmul(probs_users_embeddings, tf.transpose(coal_matrics))
        sim = (probs_similarity.eval())
    #     results = (-sim[0]).argsort()[0:top_k]
    #     print(results)
    
    #     print(sim.shape)
    #     print(np.argmax(sim, 1))
        p = np.argmax(sim, 1)
        print("买过这个煤种的人还喜欢浏览：")

        results = set()
        while len(results) != 5:
            c = p[random.randrange(top_k)]
            results.add(c)
        for val in (results):
            print(val)
            print(coals_orig[val])
        
        return results


# In[64]:


recommend_other_favorite_movie(100, 20)


# # 结论
# 
# 以上就是实现的常用的推荐功能，将网络模型作为回归问题进行训练，得到训练好的用户特征矩阵和电影特征矩阵进行推荐。

# ## 扩展阅读
# 如果你对个性化推荐感兴趣，以下资料建议你看看：

# - [`Understanding Convolutional Neural Networks for NLP`](http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/)
# 
# - [`Convolutional Neural Networks for Sentence Classification`](https://github.com/yoonkim/CNN_sentence)
# 
# - [`利用TensorFlow实现卷积神经网络做文本分类`](http://www.jianshu.com/p/ed3eac3dcb39?from=singlemessage)
# 
# - [`Convolutional Neural Network for Text Classification in Tensorflow`](https://github.com/dennybritz/cnn-text-classification-tf)
# 
# - [`SVD Implement Recommendation systems`](https://github.com/songgc/TF-recomm)

# 今天的分享就到这里，请多指教！
