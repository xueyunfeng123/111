import pandas as pd
import numpy as np
from scipy import stats
import itertools
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns

# 将已支付金额对投保人状态进行分类，并在类间做独立变量的t检验
def t_test_Policy_HolderStatus(Claim):
    # 创建 DataFrame
    df = Claim[["Policy_HolderStatus", "TotalPaid"]]

    # 按 'category' 分组
    grouped = df.groupby('Policy_HolderStatus')['TotalPaid']

    # 获取所有类别的组合（即每一对组的配对）
    category_combinations = itertools.combinations(df['Policy_HolderStatus'].unique(), 2)

    # 创建空的结果 DataFrame
    results = []

    # 对每对类别组进行 t 检验
    for cat1, cat2 in category_combinations:
        group1 = df[df['Policy_HolderStatus'] == cat1]['TotalPaid']
        group2 = df[df['Policy_HolderStatus'] == cat2]['TotalPaid']

        t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)  # 如果方差不齐，可以设置equal_var=False

        results.append({'Group1': cat1, 'Group2': cat2, 'T-statistic': t_stat, 'P-value': p_value})

        print(f"Comparing {cat1} and {cat2}:")
        print(f"T-statistic: {t_stat}, P-value: {p_value}")
        if p_value < 0.05:
            print(f"{cat1} 与 {cat2} 间存在显著差异")
        print("-" * 50)

    results_df = pd.DataFrame(results)

    return results_df

def t_test_ProviderCategoryService(Claim):
    # 创建 DataFrame
    df = Claim[["ProviderCategoryService", "TotalPaid"]]

    # 按 'category' 分组
    grouped = df.groupby('ProviderCategoryService')['TotalPaid']

    # 获取所有类别的组合（即每一对组的配对）
    category_combinations = itertools.combinations(df['ProviderCategoryService'].unique(), 2)

    # 创建空的结果 DataFrame
    results = []

    # 对每对类别组进行 t 检验
    for cat1, cat2 in category_combinations:
        group1 = df[df['ProviderCategoryService'] == cat1]['TotalPaid']
        group2 = df[df['ProviderCategoryService'] == cat2]['TotalPaid']

        t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)  # 如果方差不齐，可以设置equal_var=False

        results.append({'Group1': cat1, 'Group2': cat2, 'T-statistic': t_stat, 'P-value': p_value})

        print(f"Comparing {cat1} and {cat2}:")
        print(f"T-statistic: {t_stat}, P-value: {p_value}")
        if p_value < 0.05:
            print(f"{cat1} 与 {cat2} 间存在显著差异")
        print("-" * 50)

    results_df = pd.DataFrame(results)

    return results_df

def t_test_Procedure(Claim):
    # 创建 DataFrame
    df = Claim[["Procedure", "TotalPaid"]]

    # 按 'category' 分组
    grouped = df.groupby('Procedure')['TotalPaid']

    # 获取所有类别的组合（即每一对组的配对）
    category_combinations = itertools.combinations(df['Procedure'].unique(), 2)

    # 创建空的结果 DataFrame
    results = []

    # 对每对类别组进行 t 检验
    for cat1, cat2 in category_combinations:
        group1 = df[df['Procedure'] == cat1]['TotalPaid']
        group2 = df[df['Procedure'] == cat2]['TotalPaid']

        t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)  # 如果方差不齐，可以设置equal_var=False

        results.append({'Group1': cat1, 'Group2': cat2, 'T-statistic': t_stat, 'P-value': p_value})


        if p_value < 0.05:
            print("-" * 50)
            print(f"{cat1} 与 {cat2} 间存在显著差异")
            print(f"Comparing {cat1} and {cat2}:")
            print(f"T-statistic: {t_stat}, P-value: {p_value}")


    results_df = pd.DataFrame(results)

    return results_df

def Cohen_d_Procedure(Claim):
    unique_groups = Claim['Procedure'].unique()

    effect = []# 用于存储结果

    effect_matrix = np.zeros((len(unique_groups), len(unique_groups)))# 预备一个矩阵，用于绘制热图

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

    return effect_df

def t_test_MEDcode(Policy_Holder):
    # 创建 DataFrame
    df = Policy_Holder[["MEDcode", "TotalPaid"]]

    # 创建空的结果 DataFrame
    results = []

    # 对每对类别组进行 t 检验
    group1 = df[df['MEDcode'] == 'MCD']['TotalPaid']
    group2 = df[df['MEDcode'] == 'Undocumented']['TotalPaid']

    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)  # 如果方差不齐，可以设置equal_var=False
    d = pg.compute_effsize(group1, group2, eftype='cohen')
    results.append({'Group1': 'MCD', 'Group2': 'Undocumented', 'T-statistic': t_stat, 'P-value': p_value, 'Cohen_d': d})

    print("Comparing MCD and Undocumented:")
    print(f"T-statistic: {t_stat}, P-value: {p_value}, Cohen_d: {d}")

    results_df = pd.DataFrame(results)

    return results_df