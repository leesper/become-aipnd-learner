
# coding: utf-8

# ## 探索电影数据集
# 
# 在这个项目中，你将尝试使用所学的知识，使用 `NumPy`、`Pandas`、`matplotlib`、`seaborn` 库中的函数，来对电影数据集进行探索。
# 
# 下载数据集：
# [TMDb电影数据](https://s3.cn-north-1.amazonaws.com.cn/static-documents/nd101/explore+dataset/tmdb-movies.csv)
# 

# 
# 数据集各列名称的含义：
# <table>
# <thead><tr><th>列名称</th><th>id</th><th>imdb_id</th><th>popularity</th><th>budget</th><th>revenue</th><th>original_title</th><th>cast</th><th>homepage</th><th>director</th><th>tagline</th><th>keywords</th><th>overview</th><th>runtime</th><th>genres</th><th>production_companies</th><th>release_date</th><th>vote_count</th><th>vote_average</th><th>release_year</th><th>budget_adj</th><th>revenue_adj</th></tr></thead><tbody>
#  <tr><td>含义</td><td>编号</td><td>IMDB 编号</td><td>知名度</td><td>预算</td><td>票房</td><td>名称</td><td>主演</td><td>网站</td><td>导演</td><td>宣传词</td><td>关键词</td><td>简介</td><td>时常</td><td>类别</td><td>发行公司</td><td>发行日期</td><td>投票总数</td><td>投票均值</td><td>发行年份</td><td>预算（调整后）</td><td>票房（调整后）</td></tr>
# </tbody></table>
# 

# **请注意，你需要提交该报告导出的 `.html`、`.ipynb` 以及 `.py` 文件。**

# 
# 
# ---
# 
# ---
# 
# ## 第一节 数据的导入与处理
# 
# 在这一部分，你需要编写代码，使用 Pandas 读取数据，并进行预处理。

# 
# **任务1.1：** 导入库以及数据
# 
# 1. 载入需要的库 `NumPy`、`Pandas`、`matplotlib`、`seaborn`。
# 2. 利用 `Pandas` 库，读取 `tmdb-movies.csv` 中的数据，保存为 `movie_data`。
# 
# 提示：记得使用 notebook 中的魔法指令 `%matplotlib inline`，否则会导致你接下来无法打印出图像。

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


movie_data = pd.read_csv('tmdb-movies.csv')


# ---
# 
# **任务1.2: ** 了解数据
# 
# 你会接触到各种各样的数据表，因此在读取之后，我们有必要通过一些简单的方法，来了解我们数据表是什么样子的。
# 
# 1. 获取数据表的行列，并打印。
# 2. 使用 `.head()`、`.tail()`、`.sample()` 方法，观察、了解数据表的情况。
# 3. 使用 `.dtypes` 属性，来查看各列数据的数据类型。
# 4. 使用 `isnull()` 配合 `.any()` 等方法，来查看各列是否存在空值。
# 5. 使用 `.describe()` 方法，看看数据表中数值型的数据是怎么分布的。
# 
# 

# In[3]:


print('movie data is of {} rows and {} columns'.format(*movie_data.shape))


# In[4]:


movie_data.sample()


# In[5]:


movie_data.head()


# In[6]:


movie_data.tail()


# In[8]:


movie_data.dtypes


# In[9]:


print('NaNs in data')
movie_data.isnull().any()


# In[10]:


movie_data.describe()


# ---
# 
# **任务1.3: ** 清理数据
# 
# 在真实的工作场景中，数据处理往往是最为费时费力的环节。但是幸运的是，我们提供给大家的 tmdb 数据集非常的「干净」，不需要大家做特别多的数据清洗以及处理工作。在这一步中，你的核心的工作主要是对数据表中的空值进行处理。你可以使用 `.fillna()` 来填补空值，当然也可以使用 `.dropna()` 来丢弃数据表中包含空值的某些行或者列。
# 
# 任务：使用适当的方法来清理空值，并将得到的数据保存。

# In[35]:


print('missing values of each column')
fields = movie_data.isnull().sum().sort_values(ascending=False)
missing_fields = fields[fields > 0]
base_color = sns.color_palette()[0]
sns.barplot(missing_fields, missing_fields.index.values, color=base_color)


# 从上图可以看出，homepage字段的缺失值最多，其次是tagline，然后是keywords，homepage和imdb_id对后面的分析用处不大，可以直接删掉这两列，其他可能有用的字段缺失值就统一标记为missing吧。

# In[36]:


# homepage缺失项最多，且对后续分析没什么用处，imdb_id也是，直接drop掉这2列
movie_data_cleaned = movie_data.drop(['imdb_id', 'homepage'], axis=1)
# 缺失的tagline, keywords和production_companies统一标记为'missing'
value = {
    'tagline': 'missing',
    'keywords': 'missing',
    'production_companies': 'missing',
}
movie_data_cleaned.fillna(value=value, inplace=True)
# 剩下占比不多的空值直接drop掉，以免影响后续分析
movie_data_cleaned.dropna(inplace=True)
print('movie data after cleaned in {} rows and {} columns', *movie_data_cleaned.shape)
movie_data_cleaned.isnull().sum()


# ---
# 
# ---
# 
# ## 第二节 根据指定要求读取数据
# 
# 
# 相比 Excel 等数据分析软件，Pandas 的一大特长在于，能够轻松地基于复杂的逻辑选择合适的数据。因此，如何根据指定的要求，从数据表当获取适当的数据，是使用 Pandas 中非常重要的技能，也是本节重点考察大家的内容。
# 
# 

# ---
# 
# **任务2.1: ** 简单读取
# 
# 1. 读取数据表中名为 `id`、`popularity`、`budget`、`runtime`、`vote_average` 列的数据。
# 2. 读取数据表中前1～20行以及48、49行的数据。
# 3. 读取数据表中第50～60行的 `popularity` 那一列的数据。
# 
# 要求：每一个语句只能用一行代码实现。

# In[37]:


movie_data_cleaned[['id', 'popularity', 'budget', 'runtime', 'vote_average']].head()


# In[38]:


movie_data_cleaned.iloc[list(range(20)) + [48, 49]]


# In[39]:


movie_data_cleaned.iloc[list(range(50, 61))]['popularity']


# ---
# 
# **任务2.2: **逻辑读取（Logical Indexing）
# 
# 1. 读取数据表中 **`popularity` 大于5** 的所有数据。
# 2. 读取数据表中 **`popularity` 大于5** 的所有数据且**发行年份在1996年之后**的所有数据。
# 
# 提示：Pandas 中的逻辑运算符如 `&`、`|`，分别代表`且`以及`或`。
# 
# 要求：请使用 Logical Indexing实现。

# In[40]:


movie_data_cleaned[movie_data_cleaned['popularity'] > 5]


# In[41]:


movie_data_cleaned[(movie_data_cleaned['popularity'] > 5) & (movie_data_cleaned['release_year'] > 1996)]


# ---
# 
# **任务2.3: **分组读取
# 
# 1. 对 `release_year` 进行分组，使用 [`.agg`](http://pandas.pydata.org/pandas-docs/version/0.22/generated/pandas.core.groupby.DataFrameGroupBy.agg.html) 获得 `revenue` 的均值。
# 2. 对 `director` 进行分组，使用 [`.agg`](http://pandas.pydata.org/pandas-docs/version/0.22/generated/pandas.core.groupby.DataFrameGroupBy.agg.html) 获得 `popularity` 的均值，从高到低排列。
# 
# 要求：使用 `Groupby` 命令实现。

# In[42]:


movie_data_cleaned.groupby('release_year')['revenue'].agg(['mean'])


# In[43]:


movie_data_cleaned.groupby('director')['popularity'].agg('mean').sort_values(ascending=False)


# ---
# 
# ---
# 
# ## 第三节 绘图与可视化
# 
# 接着你要尝试对你的数据进行图像的绘制以及可视化。这一节最重要的是，你能够选择合适的图像，对特定的可视化目标进行可视化。所谓可视化的目标，是你希望从可视化的过程中，观察到怎样的信息以及变化。例如，观察票房随着时间的变化、哪个导演最受欢迎等。
# 
# <table>
# <thead><tr><th>可视化的目标</th><th>可以使用的图像</th></tr></thead><tbody>
#  <tr><td>表示某一属性数据的分布</td><td>饼图、直方图、散点图</td></tr>
#  <tr><td>表示某一属性数据随着某一个变量变化</td><td>条形图、折线图、热力图</td></tr>
#  <tr><td>比较多个属性的数据之间的关系</td><td>散点图、小提琴图、堆积条形图、堆积折线图</td></tr>
# </tbody></table>
# 
# 在这个部分，你需要根据题目中问题，选择适当的可视化图像进行绘制，并进行相应的分析。对于选做题，他们具有一定的难度，你可以尝试挑战一下～

# **任务3.1：**对 `popularity` 最高的20名电影绘制其 `popularity` 值。

# In[45]:


movies_top_20_pop = movie_data_cleaned.sort_values(by=['popularity'], ascending=False).head(20)
base_color = sns.color_palette()[0]
sns.barplot(data=movies_top_20_pop, x='popularity', y='original_title', color=base_color)
plt.xlabel('Popularity')
plt.ylabel('Title');


# 这里采用的是条形图来展示流行度最高的前20名电影，因为电影标题属于称名量表（nominal）而非顺序量表（ordinal），因此这里按流行度进行排序，最流行的是《侏罗纪世界》，其次是《疯狂的Max》，第三名则是《星级穿越》。

# ---
# **任务3.2：**分析电影净利润（票房-成本）随着年份变化的情况，并简单进行分析。

# In[79]:


movie_data_cleaned['net_profit'] = movie_data_cleaned['revenue'] - movie_data_cleaned['budget']
cnt_movies_by_year = movie_data_cleaned['original_title'].groupby(movie_data_cleaned['release_year']).count()
cnt_movies_by_year.plot()
plt.xlabel('Year')
plt.ylabel('Movies Released');


# In[80]:


cnt_movies_by_year = movie_data_cleaned['net_profit'].groupby(movie_data_cleaned['release_year']).sum()
cnt_movies_by_year.plot()
plt.xlabel('Year')
plt.ylabel('Total Profit');


# In[82]:


net_profit_by_year = movie_data_cleaned.groupby(['release_year'])['net_profit'].agg('mean')
net_profit_by_year.plot(kind='line')
plt.xlabel('Year')
plt.ylabel('Avg Profit');


# 电影的发行量和总利润每年是逐步升高的，特别是2000年以后增长很快；再结合平均利润来看，某些年份会有回落，如果某一年电影发行量比较少的, 那么每部电影对平均利润的影响会更大；如果某一年电影发行量较多, 每部电影对平均利润的影响就会更少。

# ---
# 
# **[选做]任务3.3：**选择最多产的10位导演（电影数量最多的），绘制他们排行前3的三部电影的票房情况，并简要进行分析。

# In[143]:


movie_data_split = movie_data_cleaned['director'].str.split('|', expand=True).stack().reset_index(level=0).set_index('level_0').rename(columns={0:'director'}).join(movie_data_cleaned[['revenue', 'original_title']])

top_10_directors = movie_data_split['original_title'].groupby(movie_data_split['director']).count().sort_values(ascending=False)[:10].index

top_director_movies = movie_data_split[movie_data_split['director'].isin(top_10_directors)]
top_3_movies = top_director_movies.sort_values(by='revenue', ascending=False).groupby(['director']).head(3)


# In[158]:


def plot_top_3_movies(data, directors):
    plt.figure(figsize=(20, 40))
    for index, director in enumerate(directors):
        plt.subplot(len(directors), 1, index+1)
        dd = data[data['director'] == director]
        plt.bar(x=dd['original_title'], height=dd['revenue'])
        plt.ylabel(director)


# In[160]:


plot_top_3_movies(top_3_movies, top_10_directors.values)


# 最多产的10位导演中，前三甲分别是Woody Allen，Clint Eastwood和Steven Spielberg。大部分导演排前3的电影票房差距还是比较大的，名次越靠前，所执导的电影票房收入越高。

# ---
# 
# **[选做]任务3.4：**分析1968年~2015年六月电影的数量的变化。

# In[170]:


after1968 = movie_data_cleaned['release_year'] >= 1968
before2015 = movie_data_cleaned['release_year'] <= 2015
inJune = movie_data_cleaned['release_date'].str.startswith('6/')
movies_in_year = movie_data_cleaned[(after1968) & (before2015) & (inJune)]
# movies_in_year.groupby(['release_year'])['original_title'].count().plot(kind='bar')
plt.figure(figsize=(10, 8))
sns.countplot(data=movies_in_year, y='release_year', color=base_color)


# 从图像上看，1968年到2015年以来，每年6月发行的电影量总体趋势上是呈上升趋势的，上世纪90年代有小幅回落，千禧年之后又有较大规模增长。

# ---
# 
# **[选做]任务3.5：**分析1968年~2015年六月电影 `Comedy` 和 `Drama` 两类电影的数量的变化。

# In[175]:


comedy = movies_in_year['genres'].str.contains('Comedy')
drama = movies_in_year['genres'].str.contains('Drama')
movies_drama = movies_in_year[drama]
movies_comedy = movies_in_year[comedy]
plt.figure(figsize=(10, 8))
sns.countplot(data=movies_comedy, y='release_year', color=base_color)
plt.xlabel('Comedy Movies');


# 1968年到2015年期间每年6月喜剧电影发行量总体呈上升趋势，20世纪80年代有过一次小规模的爆发，到了2000年以后，喜剧电影发行量每年6月都超过了10部。

# In[176]:


plt.figure(figsize=(10, 8))
sns.countplot(data=movies_drama, y='release_year', color=base_color)
plt.xlabel('Drama Movies');


# 1968年到2015年期间每年6月发行的戏剧总体也是呈上升趋势的，从1999年以后戏剧在每年6月发行量有较大增长，大部分都超过了10部。

# > 注意: 当你写完了所有的代码，并且回答了所有的问题。你就可以把你的 iPython Notebook 导出成 HTML 文件。你可以在菜单栏，这样导出**File -> Download as -> HTML (.html)、Python (.py)** 把导出的 HTML、python文件 和这个 iPython notebook 一起提交给审阅者。
