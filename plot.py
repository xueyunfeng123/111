import matplotlib.pyplot as plt


def histogram_TotalPaid(Claim):  # 绘制已支付金额的直方图
    plt.style.use('ggplot')  # 设置绘图风格

    # plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 处理中文乱码

    plt.rcParams['axes.unicode_minus'] = False  # 坐标轴负号的处理

    Claim.TotalPaid.plot(kind='hist', bins=20, color='steelblue', edgecolor='black', density=True,
                         label='Histogram')  # 绘制直方图

    Claim.TotalPaid.plot(kind='kde', color='red', label='KDE')  # 绘制核密度图

    plt.xlabel('TotalPaid($)')
    plt.ylabel('Frequency')  # 添加x轴和y轴标签

    # 添加标题
    plt.title('Histogram of TotalPaid')
    # 显示图例
    plt.legend()
