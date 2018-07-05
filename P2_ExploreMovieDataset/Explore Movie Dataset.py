
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

# In[123]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[124]:


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

# In[125]:


print('movie data is of {} rows and {} columns'.format(*movie_data.shape))


# In[126]:


movie_data.sample()


# In[127]:


movie_data.head()


# In[128]:


movie_data.tail()


# In[129]:


movie_data.dtypes


# In[130]:


print('NaNs in data')
movie_data.isnull().any()


# In[131]:


movie_data.describe()


# ---
# 
# **任务1.3: ** 清理数据
# 
# 在真实的工作场景中，数据处理往往是最为费时费力的环节。但是幸运的是，我们提供给大家的 tmdb 数据集非常的「干净」，不需要大家做特别多的数据清洗以及处理工作。在这一步中，你的核心的工作主要是对数据表中的空值进行处理。你可以使用 `.fillna()` 来填补空值，当然也可以使用 `.dropna()` 来丢弃数据表中包含空值的某些行或者列。
# 
# 任务：使用适当的方法来清理空值，并将得到的数据保存。

# In[132]:


print('missing values of each column')
movie_data.isnull().sum()


# In[133]:


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

# In[134]:


movie_data_cleaned[['id', 'popularity', 'budget', 'runtime', 'vote_average']].head()


# In[135]:


movie_data_cleaned.iloc[lambda _: list(range(20)) + [48, 49]]


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

# In[136]:


movie_data_cleaned[movie_data_cleaned['popularity'] > 5]


# In[137]:


movie_data_cleaned[(movie_data_cleaned['popularity'] > 5) & (movie_data_cleaned['release_year'] > 1996)]


# ---
# 
# **任务2.3: **分组读取
# 
# 1. 对 `release_year` 进行分组，使用 [`.agg`](http://pandas.pydata.org/pandas-docs/version/0.22/generated/pandas.core.groupby.DataFrameGroupBy.agg.html) 获得 `revenue` 的均值。
# 2. 对 `director` 进行分组，使用 [`.agg`](http://pandas.pydata.org/pandas-docs/version/0.22/generated/pandas.core.groupby.DataFrameGroupBy.agg.html) 获得 `popularity` 的均值，从高到低排列。
# 
# 要求：使用 `Groupby` 命令实现。

# In[138]:


movie_data_cleaned.groupby('release_year')['revenue'].agg(['mean'])


# In[139]:


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

# In[140]:


movies_top_20_pop = movie_data_cleaned.sort_values(by=['popularity'], ascending=False).head(20)
bin_edges = np.arange(0, movies_top_20_pop['popularity'].max()+1, 2)
plt.hist(data=movies_top_20_pop, x='popularity', bins=bin_edges)


# ---
# **任务3.2：**分析电影净利润（票房-成本）随着年份变化的情况，并简单进行分析。

# In[141]:


movie_data_cleaned['net_profit'] = movie_data_cleaned['revenue'] - movie_data_cleaned['budget']
net_profit_by_year = movie_data_cleaned.groupby(['release_year'])['net_profit'].agg('mean')
net_profit_by_year.plot(kind='line')


# 从净利润随年份变化的折线图上可以看到，净利润虽然在某些年份上会有回落，但总体上呈曲折上升的趋势。在1977年左右突然达到了一个最高值。

# ---
# 
# **[选做]任务3.3：**选择最多产的10位导演（电影数量最多的），绘制他们排行前3的三部电影的票房情况，并简要进行分析。

# In[142]:


def plot_top_3_movies(df):
    plt.figure(figsize=(16, 40))
    directors = list(set(df['director']))
    for index, director in enumerate(directors):
        plt.subplot(10, 1, index+1)
        data = df[df['director'] == director]
        plt.bar(x=data['original_title'], height=data['revenue'])
        plt.ylabel(director)


# In[143]:


directors_top_10 = movie_data_cleaned.groupby(['director'], as_index=False).count().sort_values(by=['original_title'], ascending=False)['director'].head(10)
movies_by_top_10_directors = movie_data_cleaned[movie_data_cleaned['director'].isin(directors_top_10)][['revenue', 'original_title', 'director']]
movies_by_top_3_revenues = movies_by_top_10_directors.sort_values(by=['revenue'], ascending=False).groupby(['director']).head(3)
plot_top_3_movies(movies_by_top_3_revenues)


# 最多产的10位导演中，各自票房排前3的电影票房之间的差距还是比较大的，只有斯皮尔伯格导演的电影和韦斯·克拉文的《惊声尖叫》3部曲做到了每部电影票房都很高。

# ---
# 
# **[选做]任务3.4：**分析1968年~2015年六月电影的数量的变化。

# In[144]:


after1968 = movie_data_cleaned['release_year'] >= 1968
before2015 = movie_data_cleaned['release_year'] <= 2015
inJune = movie_data_cleaned['release_date'].str.startswith('6/')
movies_in_year = movie_data_cleaned[(after1968) & (before2015) & (inJune)]
movies_in_year.groupby(['release_year'])['original_title'].count().plot(kind='bar')


# 从图像上看，1968年到2015年以来，每年6月发行的电影量总体趋势上是呈上升趋势的，上世纪90年代有小幅回落，千禧年之后又有较大规模增长。

# ---
# 
# **[选做]任务3.5：**分析1968年~2015年六月电影 `Comedy` 和 `Drama` 两类电影的数量的变化。

# In[145]:


comedy = movies_in_year['genres'].str.contains('Comedy')
drama = movies_in_year['genres'].str.contains('Drama')
movies_drama_comedy = movies_in_year[comedy | drama]
movies_drama_comedy.groupby(['release_year'])['original_title'].count().plot(kind='bar')


# 1968年到2015年六月电影中Comedy和Drama两类电影数量总体上呈上升趋势的，仍然在上世纪90年代有小幅回落，但千禧年后又有较大发展。

# > 注意: 当你写完了所有的代码，并且回答了所有的问题。你就可以把你的 iPython Notebook 导出成 HTML 文件。你可以在菜单栏，这样导出**File -> Download as -> HTML (.html)、Python (.py)** 把导出的 HTML、python文件 和这个 iPython notebook 一起提交给审阅者。
