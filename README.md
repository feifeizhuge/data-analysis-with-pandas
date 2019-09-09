# data-analysis-with-pandas
## what is it? 

pandas是python的一个包，可以很方便的处理表格文件，分析数据。但是其函数非常的多，有时候会因为经常不用，而不够熟练。因此将平时分析过的案例记录在这里，方便学习和熟悉pandas的使用

## Let's do it
### 1.读取数据
ToDo: 怎么读取数据，怎么宏观的观察数据
### 2.数据清洗
读取完数据后，我们就需要处理的分析数据了，也就是最消耗时间的数据清洗，虽然这个过程很繁琐，但是再厉害的算法也离不开，干净整洁的数据😁
#### 2.1 数据太多怎么办？
在处理一些工业记录的数据的时候，会经常发现一个问题就是数据点太多了，比如一秒采集30个数据点，这对于时间跨度长达一天或者一周的总体数据来说，这个采样频率实在是太高了。   
在E-bus的数据中，读入的原始数据有(2928756, 87),查看时间戳这一列，发现数据非常的冗余
```python
print(X['Time [Europe/Amsterdam]'][0:40])
# Output:
0     2016-02-10 03:51:26:227
1     2016-02-10 03:51:26:228
2     2016-02-10 03:51:26:231
3     2016-02-10 03:51:26:232
4     2016-02-10 03:51:26:233
5     2016-02-10 03:51:26:235
6     2016-02-10 03:51:26:243
7     2016-02-10 03:51:26:245
8     2016-02-10 03:51:26:252
9     2016-02-10 03:51:26:264
...
```
观察可以发现，毫秒级别的数据几乎毫无意义，所以我们要进行降采样，把数据量减少。思路就是：   
1. 对`'Time [Europe/Amsterdam]'`这列数据，进行切片`X = X[0::20]`，每隔20个数据采集一个点，输出如下，发现时间戳变得稀疏了，似乎index还有些问题，我们可以最后处理完再调整
```python
print(X['Time [Europe/Amsterdam]'][0:40])
0      2016-02-10 03:51:26:227
20     2016-02-10 03:51:26:284
40     2016-02-10 03:51:26:506
60     2016-02-10 03:51:26:908
80     2016-02-10 03:51:27:511
100    2016-02-10 03:51:28:106
120    2016-02-10 03:51:28:723
140    2016-02-10 03:51:29:255
160    2016-02-10 03:51:29:907
...
```
2. 我们可以看到时间列的精度之前是`毫秒[ms]`,我们现在想变成`秒[s]`。从列`'Unix Timestamp [ms]'`入手，   

数据  | 类型 |
--------- | --------|
Unix Timestamp [ms] | int64
Time [Europe/Amsterdam] | object

`object`数据类型相当于pandas 里面的字符串类型，不好直接进行处理，而`Unix Timestamp [ms]`恰好是`int64`方便进行运算处理。
```python
X['Unix Timestamp [ms]'][0:2]
Out[8]: 
0     1455072686227
20    1455072686284

X['Time [Europe/Amsterdam]'][0:2]
Out[9]: 
0     2016-02-10 03:51:26:227
20    2016-02-10 03:51:26:284
```
然后他们二者之间可以相互转换，通过函数`pd.to_datetime(1455072686227,unit='ms')`
```python
pd.to_datetime(1455072686227,unit='ms')
Out[10]: Timestamp('2016-02-10 02:51:26.227000')
```
这个数值和之前的字符串时间格式一摸一样，就是慢了一个小时，不过没关系我们下一步进行调整
```python
pd.to_datetime(1455072686227//1000 + 60*60,unit='s')
Out[13]: Timestamp('2016-02-10 03:51:26')
```
这样就完成了，时间精度的降低，大大降低了数据的冗余。下一步就可以删除时间戳中的重复项。
```python
# 删除重复的数字，保存第一个，原地修改数据
X.drop_duplicates('Time [Europe/Amsterdam]','first',inplace=True)
修改前：
0      2016-02-10 03:51:26
20     2016-02-10 03:51:26
40     2016-02-10 03:51:26
60     2016-02-10 03:51:26
80     2016-02-10 03:51:27
100    2016-02-10 03:51:28
120    2016-02-10 03:51:28
140    2016-02-10 03:51:29
160    2016-02-10 03:51:29
修改后：
0      2016-02-10 03:51:26
80     2016-02-10 03:51:27
100    2016-02-10 03:51:28
140    2016-02-10 03:51:29
```
至此保存修改后的数据，就能得到降采样后的数据了(1.3GB ---> 21MB)
#### 2.2 处理异常值
由于公交车传感器的启动问题，通常原始数据的第一行都是Nan，为了后续处理的方便，我们先将第一行数据删除掉：
```python
# 删除第一行的数据，原地修改
X.drop(X.index[0], inplace=True)
```
接下来获取，原始数据中的特征列表：
```python
# X.columns 返回一个series，通过value属性获得一个array
Features_name = X.columns.values
```
初步筛选有价值的特征，把全为零的特征都删除掉：
```python
# print name of feature, which doesn't have value(Nan)
Feature_Nan = Features_name[X.isnull().all().values]
# drop the all columns which value are all Nan
X.dropna(axis=1, how='all', inplace=True)
```
在剩下非Nan的特征中，将值为Nan的数字都替换成零：
```python
# replace the Nan value with naught, because Nan value means no meansure happend
X.fillna(value=0, inplace=True)
```
将之前因为修改而变得混乱的index重新排列：
```python
# drop=True，表示不保留之前的index；原地修改
X.reset_index(drop=True, inplace=True)
```

#### 2.3 数据分段
接下来就是重要的数据分段，因为公交车数据的特殊性，我们要将原始数据，分段成一圈一圈的行驶路程。通过观察所有的特征，我们发现信号`'Primove - Pick-up position control'` 和分段有很大的关联，下降沿表示车停靠终点站开始充电，上升沿

接下来就是重要的数据分段，因为公交车数据的特殊性，我们要将原始数据，分段成一圈一圈的行驶路程。通过观察所有的特征，我们发现信号`'Primove - Pick-up position control'` 和分段有很大的关联，下降沿表示车停靠终点站开始充电，上升样表示充电结束，进入下一次的行驶。
![Primove - Pick-up position control](https://github.com/feifeizhuge/data-analysis-with-pandas/blob/master/data-cleaning/control_signal_segmentation.png Pick-up position control)
