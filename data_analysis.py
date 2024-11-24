import os
import pandas as pd
from numpy import mean, median, ptp, var, std
from scipy.stats import mode,chi2_contingency
import statsmodels.api as sm# 做线性回归
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt
import seaborn as sns
import plot
import itertools
import analysis
import pingouin as pg
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS

'''
----------题目----------
针对某保险公司收集的医疗保险数据，已支付金额（TotalPaid）为结果变量，进行探索性分析。探究哪些因素会引起支付金额的变化，找到、分析和解释这些重要影响因素。
所用的统计方法分为5个大类，列表如下：
1. 描述性统计分析；
2. 可视化技术；
3. 相关分析，列联分析（因支付金额是连续型变量，不适合列联分析，可寻找两个离散型变量完成列联分析）；
4. 假设检验，方差分析；
5．多元线性回归：全子集法、逐步回归法；
6. 保险公司探查到有骗保行为，请寻找到3个可疑人员，并说明理由。
'''

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

'''----------分析过程----------'''
print('*'+'-'*10+'分析过程'+'-'*10+'*')
# 将已支付金额对投保人状态进行分类，并在类间做独立变量的t检验
'''result_t_test_Policy_HolderStatus = analysis.t_test_Policy_HolderStatus(Claim=Claim)
result_t_test_Policy_HolderStatus.to_csv(output_dir_result+'/result_t_test_Policy_HolderStatus.csv', index=False)
print(result_t_test_Policy_HolderStatus)'''

# 对投保人状态做方差分析,并生成方差分析表
anova_model_PH = smf.ols('TotalPaid ~ C(Policy_HolderStatus)', data=Claim).fit()
anova_table_PH = anova_lm(anova_model_PH)
anova_table_PH_df = pd.DataFrame(anova_table_PH)
anova_table_PH_df.to_csv("./result/anova_table_PH.csv", index=True)

# 将已支付金额对医疗保险机构服务类别进行分类，并在类间做独立变量的t检验
'''result_t_test_ProviderCategoryService = analysis.t_test_ProviderCategoryService(Claim=Claim)
result_t_test_ProviderCategoryService.to_csv(output_dir_result+'/result_t_test_ProviderCategoryService.csv', index=False)
print(result_t_test_ProviderCategoryService)'''


# 对医疗保险机构服务类别做方差分析,并生成方差分析表
anova_model_PH = smf.ols('TotalPaid ~ C(ProviderCategoryService)', data=Claim).fit()
anova_table_PH = anova_lm(anova_model_PH)
anova_table_PH_df = pd.DataFrame(anova_table_PH)
anova_table_PH_df.to_csv("./result/anova_table_PCS.csv", index=True)

# 对已诊断进行描述性统计分析（将已支付金额对诊断进行分组，计算平均值和中位数并重新排列）
'''result_describe_TotalPaid_DIAG = Claim.groupby('DIAG')['TotalPaid'].agg(['mean', 'median']).reset_index()
result_describe_TotalPaid_DIAG.to_csv(output_dir_result+'/result_describe_TotalPaid_DIAG.csv')
print(result_describe_TotalPaid_DIAG)
result_describe_TotalPaid_DIAG_sorted=result_describe_TotalPaid_DIAG.sort_values(by='mean')
result_describe_TotalPaid_DIAG_sorted.to_csv(output_dir_result+'/result_describe_TotalPaid_DIAG_sorted.csv')
print('result_describe_TotalPaid_DIAG_sorted')'''

# 将已支付金额对处理过程代码进行分类，并在类间做独立变量的t检验,并计算Cohen's d效应
result_t_test_Procedure = analysis.t_test_Procedure(Claim=Claim)
result_t_test_Procedure.to_csv(output_dir_result+'/result_t_test_Procedure.csv', index=False)
print(result_t_test_Procedure)
# 获取每个 Procedure 下的所有唯一组
unique_groups = Claim['Procedure'].unique()

# 用于存储结果
effect = []
# 预备一个矩阵，用于绘制热图
effect_matrix = np.zeros((len(unique_groups), len(unique_groups)))

for i in range(len(unique_groups)):
    for j in range(len(unique_groups)):
        group1 = Claim[Claim['Procedure'] == unique_groups[i]]['TotalPaid']
        group2 = Claim[Claim['Procedure'] == unique_groups[j]]['TotalPaid']

        # 计算 Cohen's d
        d = pg.compute_effsize(group1, group2, eftype='cohen')
        effect.append({
            "Group_1": unique_groups[i],
            "Group_2": unique_groups[j],
            "Cohen_d": d
        })
        effect_matrix[i, j] = d
        effect_matrix[j, i] = -d

# 转换为 DataFrame
effect_df = pd.DataFrame(effect)
print(effect_df)
# 保存结果为 CSV 文件
effect_df.to_csv("./result/Procedures_cohen_d.csv", index=False)

# 将矩阵转化为DataFrame，并设置行列索引为Group
effect_matrix_df = pd.DataFrame(effect_matrix, index=unique_groups, columns=unique_groups)

# 绘制热图
# 设置热图的外观
plt.figure(figsize=(8, 6))

# 绘制热图，去掉数字标注
sns.set(style='white', palette='muted')  # 设置主题风格
heatmap = sns.heatmap(effect_matrix_df, annot=False, cmap='coolwarm', fmt='.2f',
                      vmin=-1, vmax=1, center=0,
                      cbar_kws={'shrink': 0.8, 'label': "Cohen's d"},
                      linewidths=0.5, linecolor='gray')

# 标题和标签
plt.title("Heatmap of Cohen's d for Different Procedures", fontsize=16, weight='bold')
plt.xlabel('Procedures', fontsize=14)
plt.ylabel('Procedures', fontsize=14)

# 调整热图外观
plt.tight_layout()
plt.savefig("./pic/Heatmap_Procedures_cohen_d.jpg", dpi=600)
plt.show()

# 将已支付金额对医疗保险机构服务类别进行分类，并在类间做独立变量的t检验
'''result_t_test_Procedure = analysis.t_test_Procedure(Claim=Claim)
result_t_test_Procedure.to_csv(output_dir_result+'/result_t_test_Procedure.csv', index=False)
print(result_t_test_Procedure)

# 对处理过程代码做方差分析,并生成方差分析表
anova_model_P = smf.ols('TotalPaid ~ C(Procedure)', data=Claim).fit()
anova_table_P = anova_lm(anova_model_P)
anova_table_P_df = pd.DataFrame(anova_table_P)
anova_table_P_df.to_csv("./result/anova_table_P.csv", index=True)'''

# 对因素：投保人状态，医疗保险机构服务类别，处理过程进行方差检验
'''anova_model_PH_PCS_P = smf.ols('TotalPaid ~ C(Policy_HolderStatus) * C(ProviderCategoryService) * C(Procedure)', data=Claim).fit()

# 生成方差分析表
anova_table_PH_PCS_P = anova_lm(anova_model_PH_PCS_P)
anova_table_PH_PCS_P_df = pd.DataFrame(anova_table_PH_PCS_P)
anova_table_PH_PCS_P_df.to_csv("./result/anova_table_PH_PCS_P.csv", index=True)'''

# 对不同治疗措施代码做t检验
result_t_test_MEDcode=analysis.t_test_MEDcode(Policy_Holder)

'''----------对住院时长与已支付金额做相关性检验----------'''
# Pearson 相关性检验
pearson_corr_TotalPaid_LOS = Claim['LOS'].corr(Claim['TotalPaid'], method='pearson')
print(f"Pearson correlation coefficient: {pearson_corr_TotalPaid_LOS}")

# Spearman 相关性检验
spearman_corr_TotalPaid_LOS = Claim['LOS'].corr(Claim['TotalPaid'], method='spearman')
print(f"Spearman correlation coefficient: {spearman_corr_TotalPaid_LOS}")

'''----------对保费覆盖额和支付金额做线性回归----------'''
X_TotalAllowed = Claim['TotalAllowed']
Y_TotalPaid = Claim['TotalPaid']

# 增加常数项（截距项），这是 statsmodels 所必需的
X_TotalAllowed = sm.add_constant(X_TotalAllowed)

# 创建 OLS 模型并拟合
model = sm.OLS(Y_TotalPaid, X_TotalAllowed).fit()

# 提取回归结果的系数表为 DataFrame
results_regression_TotalAllowed_TotalPaid = model.summary2().tables[1]  # 提取包含系数的表

# 保存为 CSV 文件
results_regression_TotalAllowed_TotalPaid.to_csv("./result/regression_TotalAllowed_TotalPaid.csv", index=True)

# 输出回归结果
print(model.summary())

'''----------对医疗保险机构服务类别和投保人状态做列联表分析----------'''
def contingency_ProviderCategoryService_Policy_HolderStatus(Claim):
    print('----------对医疗保险机构服务类别和投保人状态做列联表分析----------')
    # 生成列联表
    contingency_table_ProviderCategoryService_Policy_HolderStatus = pd.crosstab(Claim["ProviderCategoryService"], Claim["Policy_HolderStatus"])
    print("列联表：")
    print(contingency_table_ProviderCategoryService_Policy_HolderStatus)

    # 卡方检验
    chi2, p, dof, expected = chi2_contingency(contingency_table_ProviderCategoryService_Policy_HolderStatus)
    print("\n卡方检验结果：")
    print(f"卡方值: {chi2:.4f}")
    print(f"p值: {p:.4f}")
    print(f"自由度: {dof}")
    print("期望频数：")
    print(expected)

    contingency_table_ProviderCategoryService_Policy_HolderStatus.to_csv("./result/contingency_table_ProviderCategoryService_Policy_HolderStatus.csv", index=True)

    # 可视化列联表
    plt.figure(figsize=(11, 10))
    plt.subplots_adjust(bottom=0.2, left=0.2)  # 手动调整边距
    sns.heatmap(contingency_table_ProviderCategoryService_Policy_HolderStatus, annot=True, fmt="d", cmap="Blues")
    plt.title("Contingency Table Heatmap")
    plt.ylabel("Insurance Type")
    plt.xlabel("Policy_HolderStatus")
    plt.savefig(output_dir_pic+'/contingency_table_ProviderCategoryService_Policy_HolderStatus.jpg', dpi=600)
    plt.show()

contingency_ProviderCategoryService_Policy_HolderStatus(Claim)

'''----------对账单金额和支付金额做线性回归----------'''
'''X_TotalBilled = Claim['TotalBilled']
Y_TotalPaid = Claim['TotalPaid']

# 增加常数项（截距项），这是 statsmodels 所必需的
X_TotalBilled = sm.add_constant(X_TotalBilled)

# 创建 OLS 模型并拟合
model = sm.OLS(Y_TotalPaid, X_TotalBilled).fit()

# 提取回归结果的系数表为 DataFrame
results_regression_TotalBilled_TotalPaid = model.summary2().tables[1]  # 提取包含系数的表

# 保存为 CSV 文件
results_regression_TotalBilled_TotalPaid.to_csv("./result/regression_TotalBilled_TotalPaid.csv", index=True)

# 输出回归结果
print(model.summary())'''

'''----------对保费覆盖额，账单金额和支付金额做多元线性回归----------'''
'''X_TotalBilled_TotalAllowed = Claim[['TotalBilled', 'TotalAllowed']]
Y_TotalPaid = Claim['TotalPaid']

# 增加常数项（截距项），这是 statsmodels 所必需的
X_TotalBilled = sm.add_constant(X_TotalBilled_TotalAllowed)

# 创建 OLS 模型并拟合
model = LinearRegression()

efs = EFS(
    estimator=model,                            # 回归模型
    min_features=1,                             # 子集最少包含的特征数
    max_features=len(X_TotalBilled_TotalAllowed.columns),    # 子集最多包含的特征数
    scoring='r2',                               # 评价指标
    cv=0                                        # 不进行交叉验证
)

# 执行特征选择
efs = efs.fit(X_TotalBilled_TotalAllowed, Y_TotalPaid)

# 打印结果
print("最佳特征组合:", efs.best_feature_names_)
print("对应 R^2:", efs.best_score_)

# 全部子集的结果
subset_results = pd.DataFrame.from_dict(efs.subsets_).T
print(subset_results)

# 提取特征数量和 R^2
subset_results["n_features"] = subset_results["feature_idx"].apply(len)
subset_results = subset_results.sort_values(by="n_features")'''

print("end")
