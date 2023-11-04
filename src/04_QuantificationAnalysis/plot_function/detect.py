import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scikitplot.metrics import plot_confusion_matrix,confusion_matrix


def confusion_matrix_plot(plot_data):
    
    # 获取数据
    y_true = plot_data['True Release Volume']
    y_pred = plot_data['Qube Release Volume']

    # 绘制混淆矩阵
    plt.figure(figsize=(10,10))
    
    plot_confusion_matrix(y_true=y_true, y_pred=y_pred, cmap='Reds',
                          text_fontsize='large',title='Counfusion matrix of the Hour Release',
                          hide_counts=True)

    plt.annotate(text='TN', xy=[-0.15, -0.1], fontsize=12)
    plt.annotate(text='FP', xy=[0.85, -0.1], fontsize=12)
    plt.annotate(text='FN', xy=[-0.15, 0.92], fontsize=12)
    plt.annotate(text='TP', xy=[0.85, 0.92], fontsize=12)

    # 绘制百分比
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize='all')
    cm = np.round(cm, 4)
    cm = cm *100

    TP_RATE = '%.2f'%cm[0,0] + '%'
    FP_RATE = '%.2f'%cm[0,1] + '%'
    FN_RATE = '%.2f'%cm[1,0] + '%'
    TN_RATE = '%.2f'%cm[1,1] + '%'

    plt.annotate(text=TP_RATE, xy=[-0.15, 0.15], fontsize=12)
    plt.annotate(text=FP_RATE, xy=[0.85, 0.15], fontsize=12)
    plt.annotate(text=FN_RATE, xy=[-0.15, 1.15], fontsize=12)
    plt.annotate(text=TN_RATE, xy=[0.85, 1.14], fontsize=12)
    
    
    # 绘制数量
    cn = confusion_matrix(y_true=y_true, y_pred=y_pred)
    
    TP_NUM = cn[0,0]
    FP_NUM = cn[0,1]
    FN_NUM = cn[1,0]
    TN_NUM = cn[1,1]
    
    plt.annotate(text=TP_NUM, xy=[-0.15, 0.03], fontsize=12)
    plt.annotate(text=FP_NUM, xy=[0.85, 0.03], fontsize=12)
    plt.annotate(text=FN_NUM, xy=[-0.15, 1.03], fontsize=12)
    plt.annotate(text=TN_NUM, xy=[0.85, 1.03], fontsize=12)
    

    plt.ylabel('True Release Volume')
    plt.xlabel('Qube Release Volume')
    
    
    plt.show()
    
    
def detection_rate_bin_plot(data, n_bins, threshold):
    
    # 拷贝数据
    plot_data = data.copy()
    # 分箱区间
    bins = np.linspace(start=0, stop=threshold, num=n_bins+1)

    # 计算区间中值
    bins_df = pd.Series(data=bins)
    bins_median = bins_df.rolling(window=2).mean().dropna().values

    # 分箱以及统计
    plot_data['bins']= pd.cut(plot_data['True Release Rate (Kg/hour)'], bins=bins, precision=0)
    stat_table = pd.pivot_table(data=plot_data, index='bins', values='Qube Release Volume', aggfunc=['mean','sum','count'])

    # 柱状图X轴位置
    x = bins_median

    # 柱状图Y轴位置
    height = stat_table['mean'].values.reshape(-1)

    # 柱子宽度
    width = threshold/(n_bins+1)-0.5

    # 检测成功样本数量
    suc_counts = stat_table['sum'].values.reshape(-1)

    # 总观察样本数量
    det_counts = stat_table['count'].values.reshape(-1)

    # 圆圈散点数据
    scatter_data = plot_data.loc[plot_data['True Release Rate (Kg/hour)']<=threshold, :]
    scatter_x = scatter_data['True Release Rate (Kg/hour)'].values.reshape(-1)
    scatter_y = scatter_data['Qube Release Volume'].astype('int').values.reshape(-1)
    
    
    # find the lower and uppder bound defined by two sigma
    ps = stat_table['mean'].values.reshape(-1)
    ns = stat_table['count'].values.reshape(-1)

    sigmas = np.sqrt(ps*(1-ps)/ns)
    two_sigmas = 2*sigmas

    two_sigma_lowers = 2*sigmas
    two_sigma_uppers = 2*sigmas

    two_sigma_uppers[(two_sigmas + ps)>1] = (1-ps)[(two_sigmas + ps)>1]
    two_sigma_lowers[(ps- two_sigmas)<0] = ps[(ps- two_sigmas)<0]
    
    
    # 设置图大小
    plt.figure(figsize=(10,8))

    # 绘制柱状图
    plt.bar(x=bins_median,
            height=height,
            yerr=[two_sigma_lowers,two_sigma_uppers],
            error_kw=dict(lw=2, capsize=3, capthick=1,alpha=0.3),
            width=width,
            alpha=0.6,
            color='#8c1515',
            ecolor='black',
            capsize=2
           )

    # 绘制散点图
    plt.scatter(x=scatter_x, y=scatter_y, marker='o', edgecolor="black", facecolors='white', s=100, linewidths=1.5)

    # 绘制标签值
    for i in range(n_bins):
        plt.annotate('%d / %d' %(suc_counts[i],det_counts[i]), [x[i]- width*0.45, 0.03],fontsize=10)

    plt.ylabel('Proportion detected',fontsize=16)
    plt.xlabel('True Release Rate (Kg/hour)',fontsize=16)
    plt.yticks(ticks=[0.0,0.2,0.4,0.6,0.8,1.0], fontsize=12)
    plt.xticks(fontsize=12)
    
    #plt.show()