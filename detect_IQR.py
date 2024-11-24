import pandas as pd
# from sklearn.ensemble import IsolationForest
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# 读取文件
Claim = pd.read_spss("data/Claim.sav")
Policy_Holder = pd.read_spss("data/Policy_Holder_new.sav")
Provider = pd.read_spss("data/Provider_new.sav")

# 缺失值处理
Claim_null = any(Claim.isnull())  # 返回缺失值矩阵（在后文中并不会使用）
Claim = Claim.dropna()  # 删除所有含有缺失值的行
# Claim = Claim[["Policy_HolderStatus", "ProviderCategoryService", "DIAG", "Procedure", "LOS", "PlaceOfService", "TotalPaid"]]
# 定义检测异常的函数
def detect_outliers_iqr(group):
    Q1 = group['TotalPaid'].quantile(0.25)  # 第一四分位数
    Q3 = group['TotalPaid'].quantile(0.75)  # 第三四分位数
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR  # 上限
    # 标记异常点
    group['is_outlier'] = group['TotalPaid'] > upper_bound
    return group

# 对每个诊断代码分组并检测异常
Claim = Claim.groupby('DIAG').apply(detect_outliers_iqr)

# 统计每个 claim_id 作为异常点出现的次数
outlier_counts = Claim[Claim['is_outlier']].groupby('ClaimID').size().reset_index(name='outlier_count')

# 将 outlier_count 列设为整数类型，避免 Categorical 问题
outlier_counts['outlier_count'] = outlier_counts['outlier_count'].astype(int)

'''
# 将异常点计数结果合并回原始数据，填充缺失值为 0
Claim = Claim.merge(outlier_counts, on='ClaimID', how='left')
Claim['outlier_count'] = Claim['outlier_count'].astype(int)

# 筛选出异常点
# outliers = Claim_new[Claim_new['is_outlier']]
'''

# 筛选出异常次数多于1的点
result_detect = outlier_counts[outlier_counts['outlier_count'] > 1]

# 保存
result_detect.to_csv('./result/dectet_IQR.csv', index=True)