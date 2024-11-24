import os
import pandas as pd
from numpy import mean, median, ptp, var, std
from scipy.stats import mode
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

# 创建好用以保存图片的文件夹
output_dir_pic = './pic'
os.makedirs(output_dir_pic, exist_ok=True)

# 创建好用以保存数据结果的文件夹
output_dir_result = './result'
os.makedirs(output_dir_result, exist_ok=True)

# 读取文件
Claim = pd.read_spss("data/Claim.sav")
Policy_Holder = pd.read_spss("data/Policy_Holder_new.sav")
Provider = pd.read_spss("data/Provider_new.sav")

# 缺失值处理
Claim_null = any(Claim.isnull())  # 返回缺失值矩阵（在后文中并不会使用）
Claim = Claim.dropna()  # 删除所有含有缺失值的行

'''----------特征提取----------'''
# 计算每个人的理赔频次
claim_freq = Claim['ClaimID'].value_counts().reset_index()
claim_freq.columns = ['ClaimID', 'claim_count']  # 对列进行重命名

# 计算每个人的保险理赔描述性统计特征
amount_stat = Claim.groupby('ClaimID')['TotalAllowed'].agg(['mean', 'median', 'max', 'min']).reset_index()

# 计算住院时间间隔的特征
Claim['FirstDayOfStay'] = pd.to_datetime(Claim['FirstDayOfStay'], errors='coerce')  # 先将日期转换为datetime格式，才能使用.dt
Claim = Claim.sort_values(by=['ClaimID', 'FirstDayOfStay'])
Claim['time_diff'] = Claim.groupby('ClaimID')['FirstDayOfStay'].diff().dt.days
time_interval = Claim.groupby('ClaimID')['time_diff'].mean().reset_index(name='avg_time_interval')

# 合并特征
features = claim_freq.merge(amount_stat, on='ClaimID').merge(time_interval, on='ClaimID')
Claim = Claim.merge(claim_freq, on='ClaimID').merge(amount_stat, on='ClaimID').merge(time_interval, on='ClaimID')

# 缺失值处理(某些人只进行过一次理赔，因此其avg_time_interval的值为nan),在此将其填充为均值
mean_value = features['avg_time_interval'].mean()
features['avg_time_interval'].fillna(mean_value, inplace=True)  # 用均值填充缺失值
mean_value = Claim['avg_time_interval'].mean()
Claim['avg_time_interval'].fillna(mean_value, inplace=True)  # 用均值填充缺失值

'''----------利用Isolation Forest进行异常点检测----------'''
def Isolation_Forest_detectfeatures_1(features,outliers_fraction):

    # 定义Isolation Forest的模型，并选择参数

    isolation_forest = IsolationForest(contamination=outliers_fraction)

    # 训练
    # features['anomaly'] = isolation_forest.fit_predict(features[['claim_count', 'mean', 'median', 'max', 'min', 'avg_time_interval']])
    features['anomaly'] = isolation_forest.fit_predict(features[['claim_count', 'mean', 'avg_time_interval']])

    # 导出得分
    # scores_pred = isolation_forest.decision_function(features[['claim_count', 'mean', 'median', 'max', 'min', 'avg_time_interval']])
    scores_pred = isolation_forest.decision_function(features[['claim_count', 'mean', 'avg_time_interval']])
    # 导出异常点信息
    anomalies = features[features['anomaly'] == -1]
    anomalies.to_csv('./result/detect_IsolationForest.csv', index=False)

Isolation_Forest_detectfeatures_1(features,outliers_fraction = 0.005)


# 热图绘制
isolation_forest = IsolationForest(contamination=0.05)
Claim['anomaly'] = isolation_forest.fit_predict(Claim[['claim_count', 'mean', 'median', 'max', 'min', 'avg_time_interval']])
Claim['AnomalyScore'] = isolation_forest.decision_function(Claim[['claim_count', 'mean', 'median', 'max', 'min', 'avg_time_interval']])

# 异常率统计
anomaly_rate = Claim['anomaly'].mean()

# 绘制异常分数分布
plt.figure(figsize=(10, 6))
sns.histplot(Claim["AnomalyScore"], kde=True, bins=20, color="skyblue", edgecolor="black")
plt.axvline(x=0, color="red", linestyle="--", label="Anomaly Threshold")
plt.title("Distribution of Anomaly Scores")
plt.xlabel("Anomaly Score")
plt.ylabel("Frequency")
plt.legend()
plt.savefig(output_dir_pic+'/IsolationForest.jpg', dpi=600)
plt.show()

print("end")
