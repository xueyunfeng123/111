import os
import pandas as pd
from numpy import mean, median, ptp, var, std
from scipy.stats import mode
import statsmodels.api as sm# 做线性回归
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt
import seaborn as sns
import plot
# import itertools
import analysis

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
Policy_Holder_null = any(Policy_Holder.isnull())  # 返回缺失值矩阵（在后文中并不会使用）
Policy_Holder = Policy_Holder.dropna()  # 删除所有含有缺失值的行
Provider_null = any(Provider.isnull())  # 返回缺失值矩阵（在后文中并不会使用）
Provider = Provider.dropna()  # 删除所有含有缺失值的行

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用 SimHei 字体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

'''----------可视化处理----------'''

print('*'+'-'*10+'数据可视化'+'-'*10+'*')
# 绘制已支付金额的直方图
plot.histogram_TotalPaid(Claim=Claim)
plt.savefig(output_dir_pic+'/Histogram_TotalPaid.jpg', dpi=600, transparent=False, facecolor='white')
# transparent=False禁用透明背景, facecolor='white'设置白色背景：这是为了防止保存高dpi图片时出现黑白格
plt.show()

# 使用 describe() 以获得该数据的描述性统计分析
desc = Claim['TotalPaid'].describe()

# 输出中位数、均值和方差
print("Descriptive Statistics:")
print(f"Median: {desc['50%']}")
print(f"Mean: {desc['mean']}")
print(f"Variance: {Claim['TotalPaid'].var()}")

'''
# 绘制箱型图：以投保人状态为x轴，以已支付金额为y轴
# sns.set(style="white")  # 使用纯白色背景防止黑白格问题
plt.figure(figsize=(10, 6))
sns.boxplot(x='Policy_HolderStatus', y='TotalPaid', data=Claim)
plt.savefig(output_dir_pic+'/Boxplot_Policy_HolderStatus_TotalPaid.jpg', dpi=600, bbox_inches='tight', facecolor='white', edgecolor='white')
plt.show()
'''
# 绘制箱型图(带小提琴图)：以投保人状态为x轴，以已支付金额为y轴
# sns.set(style="white")  # 使用纯白色背景防止黑白格问题
plt.figure(figsize=(10, 6))
# width箱体宽度，fliersize异常值点大小，palette箱型图颜色，linewidth箱体边框粗细，boxprops箱体透明度
sns.boxplot(x='Policy_HolderStatus', y='TotalPaid', data=Claim, width=0.3, fliersize=2.5, palette="Set2", hue="Policy_HolderStatus",linewidth=1.5, boxprops=dict(alpha=0.8))
sns.violinplot(x="Policy_HolderStatus", y="TotalPaid", data=Claim, inner=None, color="lightblue", alpha=0.7)
# 在小提琴图中添加平均值
means = Claim.groupby("Policy_HolderStatus")["TotalPaid"].mean()  # 计算每个类别的平均值
for i, mean in enumerate(means):
    plt.hlines(y=mean, xmin=i - 0.25, xmax=i + 0.25, colors="red", linestyles="--", linewidth=2, label=f"Mean ({mean:.2f})" if i == 0 else None)

plt.savefig(output_dir_pic+'/Boxplot1_Policy_HolderStatus_TotalPaid.jpg', dpi=600, bbox_inches='tight', facecolor='white', edgecolor='white')
plt.show()

'''
# 绘制箱型图：以医疗保险机构服务类别为x轴，以已支付金额为y轴
# sns.set(style="white")  # 使用纯白色背景防止黑白格问题
plt.figure(figsize=(16, 6))
sns.boxplot(x='ProviderCategoryService', y='TotalPaid', data=Claim)
plt.xticks(rotation=45)
plt.savefig(output_dir_pic+'/Boxplot_ProviderCategoryService_TotalPaid.jpg', dpi=600, bbox_inches='tight', facecolor='white', edgecolor='white')
plt.show()
'''

# 绘制箱型图(带小提琴图)：以医疗保险机构服务类别为x轴，以已支付金额为y轴
# sns.set(style="white")  # 使用纯白色背景防止黑白格问题
plt.figure(figsize=(13, 6))
# width箱体宽度，fliersize异常值点大小，palette箱型图颜色，linewidth箱体边框粗细，boxprops箱体透明度
sns.boxplot(x='ProviderCategoryService', y='TotalPaid', data=Claim, width=0.3, fliersize=2.5, palette="Set2", hue="ProviderCategoryService",linewidth=1.5, boxprops=dict(alpha=0.8))
sns.violinplot(x="ProviderCategoryService", y="TotalPaid", data=Claim, inner=None, color="lightblue", alpha=0.7)
plt.xticks(rotation=45)
# 在小提琴图中添加平均值
means = Claim.groupby("ProviderCategoryService")["TotalPaid"].mean()  # 计算每个类别的平均值
for i, mean in enumerate(means):
    plt.hlines(y=mean, xmin=i - 0.25, xmax=i + 0.25, colors="red", linestyles="--", linewidth=2, label=f"Mean ({mean:.2f})" if i == 0 else None)

plt.savefig(output_dir_pic+'/Boxplot1_Policy_ProviderCategoryService_TotalPaid.jpg', dpi=600, bbox_inches='tight', facecolor='white', edgecolor='white')
plt.show()

'''
# 绘制箱型图：以处理过程代码类别为x轴，以已支付金额为y轴
# sns.set(style="white")  # 使用纯白色背景防止黑白格问题
plt.figure(figsize=(16, 6))
sns.boxplot(x='Procedure', y='TotalPaid', data=Claim)
plt.savefig(output_dir_pic+'/Boxplot_Procedure_TotalPaid.jpg', dpi=600, bbox_inches='tight', facecolor='white', edgecolor='white')
plt.show()
'''

# 绘制箱型图(带小提琴图)：以不同处理过程为x轴，以已支付金额为y轴
# sns.set(style="white")  # 使用纯白色背景防止黑白格问题
plt.figure(figsize=(13, 6))
# width箱体宽度，fliersize异常值点大小，palette箱型图颜色，linewidth箱体边框粗细，boxprops箱体透明度
sns.boxplot(x='Procedure', y='TotalPaid', data=Claim, width=0.22, fliersize=2.5, palette="Set2", hue="Procedure",linewidth=1.5, boxprops=dict(alpha=0.8))
sns.violinplot(x="Procedure", y="TotalPaid", data=Claim, width=1.5, inner=None, color="lightblue", alpha=0.7)
plt.xticks(rotation=45)
# 在小提琴图中添加平均值
means = Claim.groupby("Procedure")["TotalPaid"].mean()  # 计算每个类别的平均值
for i, mean in enumerate(means):
    plt.hlines(y=mean, xmin=i - 0.25, xmax=i + 0.25, colors="red", linestyles="--", linewidth=2, label=f"Mean ({mean:.2f})" if i == 0 else None)
plt.savefig(output_dir_pic+'/Boxplot1_Procedure_TotalPaid.jpg', dpi=600, bbox_inches='tight', facecolor='white', edgecolor='white')
plt.show()

# 绘制核密度散点图：以住院时长为x轴，以已支付金额为y轴
sns.kdeplot(x=Claim['LOS'], y=Claim['TotalPaid'], cmap='Blues', fill=True, thresh=0.1)
plt.title('Scatter Plot of LOS vs TotalPaid with KDE')
plt.xlabel('LOS')
plt.ylabel('TotalPaid')
# plt.figure(figsize=(9, 6))
plt.savefig(output_dir_pic+'/KDE_LOS_TotalPaid.jpg', dpi=600, bbox_inches='tight', facecolor='white', edgecolor='white')
plt.show()

# 绘制核密度散点图：以账单金额为x轴，以已支付金额为y轴
sns.kdeplot(x=Claim['TotalBilled'], y=Claim['TotalPaid'], cmap='Blues', fill=True, thresh=0.1)
plt.title('Scatter Plot of TotalBilled vs TotalPaid with KDE')
plt.xlabel('TotalBilled')
plt.ylabel('TotalPaid')
# plt.figure(figsize=(9, 6))
plt.savefig(output_dir_pic+'/KDE_TotalBilled_TotalPaid.jpg', dpi=600, bbox_inches='tight', facecolor='white', edgecolor='white')
plt.show()

# 绘制核密度散点图：以保费覆盖额为x轴，以已支付金额为y轴
sns.kdeplot(x=Claim['TotalAllowed'], y=Claim['TotalPaid'], cmap='Blues', fill=True, thresh=0.1)
plt.title('Scatter Plot of TotalAllowed vs TotalPaid with KDE')
plt.xlabel('TotalAllowed')
plt.ylabel('TotalPaid')
# plt.figure(figsize=(9, 6))
plt.savefig(output_dir_pic+'/KDE_TotalAllowed_TotalPaid.jpg', dpi=600, bbox_inches='tight', facecolor='white', edgecolor='white')
plt.show()

# 绘制箱型图(带散点图)：以MEDcode为x轴，以已支付金额为y轴
# sns.set(style="white")  # 使用纯白色背景防止黑白格问题
plt.figure(figsize=(11, 6))
# width箱体宽度，fliersize异常值点大小，palette箱型图颜色，linewidth箱体边框粗细，boxprops箱体透明度
sns.boxplot(x='MEDcode', y='TotalPaid', data=Policy_Holder, width=0.3, fliersize=2.5, palette="Set2", hue="MEDcode",linewidth=1.5, boxprops=dict(alpha=0.8))
sns.violinplot(x="MEDcode", y="TotalPaid", data=Policy_Holder, width=0.8, inner=None, color="lightblue", alpha=0.7)
sns.swarmplot(x="MEDcode", y="TotalPaid", data=Policy_Holder, color="black", alpha=0.6, size=4)

# 在图中添加平均值及样本量
means = Policy_Holder.groupby("MEDcode")["TotalPaid"].mean()  # 计算每个类别的平均值
for i, mean in enumerate(means):
    plt.hlines(y=mean, xmin=i - 0.25, xmax=i + 0.25, colors="red", linestyles="--", linewidth=2, label=f"Mean ({mean:.2f})" if i == 0 else None)
group_sizes = Policy_Holder.groupby("MEDcode").size()  # 计算每组样本量
for i, category in enumerate(group_sizes.index):
    size = group_sizes[category]
    plt.text(i, Policy_Holder['TotalPaid'].max() + 0.5, f"n={size}", ha='center', fontsize=12, color='black')

plt.savefig(output_dir_pic+'/Boxplot2_MEDcode_TotalPaid.jpg', dpi=600, bbox_inches='tight', facecolor='white', edgecolor='white')
plt.show()

'''
# 绘制箱型图(带散点图)：以ProviderType为x轴，以已支付金额为y轴
# sns.set(style="white")  # 使用纯白色背景防止黑白格问题
plt.figure(figsize=(20, 6))
# 过滤掉ProviderType中size小于10的组
group_sizes = Provider.groupby("ProviderType").size()
valid_groups = group_sizes[group_sizes >= 10].index
ProviderType_filtered = Provider[Provider["ProviderType"].isin(valid_groups)]
ProviderType_filtered.loc[:, 'ProviderType'] = pd.Categorical(ProviderType_filtered['ProviderType'])
ProviderType_filtered['ProviderType'].cat.remove_unused_categories()
print("有效分组:", ProviderType_filtered['ProviderType'].cat.categories)

# width箱体宽度，fliersize异常值点大小，palette箱型图颜色，linewidth箱体边框粗细，boxprops箱体透明度
sns.boxplot(x='ProviderType', y='TotalPaid', data=ProviderType_filtered, width=0.3, fliersize=2.5, palette="Set2", linewidth=1.5, boxprops=dict(alpha=0.8))
sns.violinplot(x="ProviderType", y="TotalPaid", data=ProviderType_filtered, width=0.8, inner=None, color="lightblue", alpha=0.7)
sns.swarmplot(x="ProviderType", y="TotalPaid", data=ProviderType_filtered, color="black", alpha=0.6, size=4)
plt.show()
# 在图中添加平均值及样本量
means = ProviderType_filtered.groupby("ProviderType")["TotalPaid"].mean()  # 计算每个类别的平均值
for i, mean in enumerate(means):
    plt.hlines(y=mean, xmin=i - 0.25, xmax=i + 0.25, colors="red", linestyles="--", linewidth=2, label=f"Mean ({mean:.2f})" if i == 0 else None)
group_sizes = ProviderType_filtered.groupby("ProviderType").size()  # 计算每组样本量
for i, category in enumerate(group_sizes.index):
    size = group_sizes[category]
    plt.text(i, ProviderType_filtered['TotalPaid'].max() + 0.5, f"n={size}", ha='center', fontsize=12, color='black')

plt.savefig(output_dir_pic+'/Boxplot2_ProviderType_TotalPaid.jpg', dpi=600, bbox_inches='tight', facecolor='white', edgecolor='white')
plt.show()
'''

''''# 绘制箱型图：以医疗保健机构大类为x轴，以已支付金额为y轴
# sns.set(style="white")  # 使用纯白色背景防止黑白格问题
plt.figure(figsize=(15, 6))
sns.boxplot(x='ProviderSpecialty', y='TotalPaid', data=Provider)
plt.xticks(rotation=45, ha='right')
plt.savefig(output_dir_pic+'/Boxplot_ProviderSpecialty_TotalPaid.jpg', dpi=600, bbox_inches='tight', facecolor='white', edgecolor='white')
plt.show()'''

# 绘制箱型图：以保险条款为x轴，以已支付金额为y轴
# sns.set(style="white")  # 使用纯白色背景防止黑白格问题
plt.figure(figsize=(11, 6))
# width箱体宽度，fliersize异常值点大小，palette箱型图颜色，linewidth箱体边框粗细，boxprops箱体透明度
sns.boxplot(x='ProgramCode', y='TotalPaid', data=Policy_Holder, width=0.3, fliersize=2.5, palette="Set2", hue="ProgramCode",linewidth=1.5, boxprops=dict(alpha=0.8))
sns.violinplot(x="ProgramCode", y="TotalPaid", data=Policy_Holder, width=0.8, inner=None, color="lightblue", alpha=0.7)
sns.swarmplot(x="ProgramCode", y="TotalPaid", data=Policy_Holder, color="black", alpha=0.6, size=4)

# 在图中添加平均值及样本量
means = Policy_Holder.groupby("ProgramCode")["TotalPaid"].mean()  # 计算每个类别的平均值
for i, mean in enumerate(means):
    plt.hlines(y=mean, xmin=i - 0.25, xmax=i + 0.25, colors="red", linestyles="--", linewidth=2, label=f"Mean ({mean:.2f})" if i == 0 else None)
group_sizes = Policy_Holder.groupby("ProgramCode").size()  # 计算每组样本量
for i, category in enumerate(group_sizes.index):
    size = group_sizes[category]
    plt.text(i, Policy_Holder['TotalPaid'].max() + 0.5, f"n={size}", ha='center', fontsize=12, color='black')

plt.savefig(output_dir_pic+'/Boxplot2_ProgramCode_TotalPaid.jpg', dpi=600, bbox_inches='tight', facecolor='white', edgecolor='white')
plt.show()


