import random as rd
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import matplotlib.pyplot as plt

def linreg_results_no_intercept(x,y):

    """
Regression to output all values to be potentially used in plotting:
n = number of points in scatter plot;
pearson_corr = peason's correlation coefficient;
slope = regression parameters;
r_value = R (not R_squared);
x_lim = (min&max) of x;
y_pred = (min&max) of y_predction computed with slope, intercept, and x_lim;
lower_CI, upper_CI are bounds of 95% confidence interval for the fit line;
lower_PI, upper_CI are bounds of 95% prediction interval for predictions;
see reference: http://www2.stat.duke.edu/~tjl13/s101/slides/unit6lec3H.pdf
    """

    n = len(x)

    model = sm.OLS(y,x)
    result = model.fit()
    slope = result.params[0]
    r_squared = result.rsquared
    std_err = result.bse[0]

    x_lim = np.array([0,max(x)])
    y_pred = slope*x
    residual = y - y_pred
    dof = n - 1                               # degree of freedom
    t_score = stats.t.ppf(1-0.025, df=dof)    # one-sided t-test
    
    # sort x from smallest to largest for ease of plotting
    df = pd.DataFrame()
    df['x'] = x
    df['y'] = y
    df = df.sort_values('x')
    x = df.x.values
    y = df.y.values
    
    y_hat = slope*x             
    x_mean = np.mean(x)
    S_yy = np.sum(np.power(y-y_hat,2))      # total sum of error in y
    S_xx = np.sum(np.power(x-x_mean,2))     # total sum of variation in x

    # find lower and upper bounds of CI and PI
    lower_CI = y_hat - t_score * np.sqrt(S_yy/dof * (1/n+np.power(x-x_mean,2)/S_xx))
    upper_CI = y_hat + t_score * np.sqrt(S_yy/dof * (1/n+np.power(x-x_mean,2)/S_xx))
    lower_PI = y_hat - t_score * np.sqrt(S_yy/dof * (1/n+1+np.power(x-x_mean,2)/S_xx))
    upper_PI = y_hat + t_score * np.sqrt(S_yy/dof * (1/n+1+np.power(x-x_mean,2)/S_xx))
    
    return n,slope,r_squared,x_lim,y_pred,lower_CI,upper_CI,lower_PI,upper_PI,residual,std_err


def linreg_results(x,y):

    """
Regression to output all values to be potentially used in plotting:
n = number of points in scatter plot;
pearson_corr = peason's correlation coefficient;
slope, intercept = regression parameters;
r_value = R (not R_squared);
x_lim = (min&max) of x;
y_pred = (min&max) of y_predction computed with slope, intercept, and x_lim;
lower_CI, upper_CI are bounds of 95% confidence interval for the fit line;
lower_PI, upper_CI are bounds of 95% prediction interval for predictions;
see reference: http://www2.stat.duke.edu/~tjl13/s101/slides/unit6lec3H.pdf
    """

    n = len(x)
    pearson_corr, _ = stats.pearsonr(x, y)    # Pearson's correlation coefficient
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    x_lim = np.array([0,max(x)])
    y_pred = intercept + slope*x
    residual = y - (intercept+slope*x)
    dof = n - 2                               # degree of freedom
    t_score = stats.t.ppf(1-0.025, df=dof)    # one-sided t-test
    
    # sort x from smallest to largest for ease of plotting
    df = pd.DataFrame()
    df['x'] = x
    df['y'] = y
    df = df.sort_values('x')
    x = df.x.values
    y = df.y.values
    
    y_hat = intercept + slope*x             
    x_mean = np.mean(x)
    S_yy = np.sum(np.power(y-y_hat,2))      # total sum of error in y
    S_xx = np.sum(np.power(x-x_mean,2))     # total sum of variation in x

    # find lower and upper bounds of CI and PI
    lower_CI = y_hat - t_score * np.sqrt(S_yy/dof * (1/n+np.power(x-x_mean,2)/S_xx))
    upper_CI = y_hat + t_score * np.sqrt(S_yy/dof * (1/n+np.power(x-x_mean,2)/S_xx))
    lower_PI = y_hat - t_score * np.sqrt(S_yy/dof * (1/n+1+np.power(x-x_mean,2)/S_xx))
    upper_PI = y_hat + t_score * np.sqrt(S_yy/dof * (1/n+1+np.power(x-x_mean,2)/S_xx))
    
    return n,pearson_corr,slope,intercept,r_value,x_lim,y_pred,lower_CI,upper_CI,lower_PI,upper_PI,residual,std_err




# plot the parity chart
def parity_plot(ax, plot_data, force_intercept_origin=0, plot_interval=['confidence'],legend_loc='lower right'):

  """
plot parity chart: scatter plot with a parity line, a regression line, and a confidence interval

INPUTS
- ax is the subplot ax to plot on
- plot_data is the processed data
- force_intercept_origin decides which regression to use
- plot_interval can be ['confidence','prediction'] or either one of those in the list
- plot_lim is the limit of the x and y axes
- legend_loc is the location of the legend

OUTPUT
- ax is the plotted parity chart
  """
  # shape list
  shape_list = [".",",","o","v","^","<",">","1","2","3","4","8","s","p","P","*",
                       "h","H","+","x","X","D","d","|","_",0,1,2,3,4,5,6,7,8,9,10,11]
  # color list
  color_list = ['black', 'blue', 'red', 'green', 'yellow', 'gray',
           'orange', 'deeppink', 'violet', 'chocolate', 'cyan', 'purple',
           'dodgerblue', 'tomato', 'gold', 'navy', 'tan', 'rosybrown',
           'lime', 'seagreen', 'sandybrown', 'skyblue', 'lightsalmon', 'teal'
           ]
  
  
  # label name
  xlabel = 'True Release Rate (Kg/h)'
  ylabel = 'Reported Release Rate (Kg/h)'
  
  # x_lim, y_lim
  xlim = plot_data.iloc[:, 0].max() * 1.2
  ylim = plot_data.iloc[:, 1].max() * 1.2
  
  # set up plot
  ax.set_xlabel(xlabel,fontsize=13)
  ax.set_ylabel(ylabel,fontsize=13)
  ax.set_xlim([0, xlim])
  ax.set_ylim([0, xlim])

  # parity line
  x_lim = np.array([0, xlim])
  y_lim = np.array([0, xlim])
  ax.plot(x_lim, y_lim, color='black',label = 'Parity line')

  # define x and y data points
  x = plot_data.iloc[:,0].values
  y = plot_data.iloc[:,1].fillna(0).values

  # regression
  if force_intercept_origin == 0:
    n,pearson_corr,slope,intercept,r_value,x_lim,y_pred,lower_CI,upper_CI,lower_PI,upper_PI,residual,std_err = linreg_results(x,y)
  elif force_intercept_origin == 1:
    n,slope,r_squared,x_lim,y_pred,lower_CI,upper_CI,lower_PI,upper_PI,residual,std_err = linreg_results_no_intercept(x,y)

    
  # plot regression line
  if force_intercept_origin == 0:
    if intercept<0:   # label differently depending on intercept
      ax.plot(x,y_pred,'-',color='#8c1515',linewidth=2.5,
              label = 'Best fit $R^{2}$ = %0.2f \n$y$ = %0.2f$x$%0.2f' % (r_value**2,slope,intercept))
    elif intercept>=0:
      ax.plot(x,y_pred,'-',color='#8c1515',linewidth=2.5,
              label = 'Best fit $R^{2}$ = %0.2f \n$y$ = %0.2f$x+$%0.2f' % (r_value**2,slope,intercept))
  elif force_intercept_origin ==1:
      ax.plot(x,y_pred,'-',color='#8c1515',linewidth=2.5,
            label = 'Best fit $R^{2}$ = %0.2f \n$y$ = %0.2f$x$' % (r_squared,slope))

  # plot intervals
  if 'confidence' in plot_interval:
    ax.plot(np.sort(x),upper_CI,':',color='black', label='95% CI')
    ax.plot(np.sort(x),lower_CI,':',color='black')
    ax.fill_between(np.sort(x), lower_CI, upper_CI, color='black', alpha=0.05)
  if 'prediction' in plot_interval:
    ax.plot(np.sort(x),upper_PI,':',color='#8c1515', label='95% PI')
    ax.plot(np.sort(x),lower_PI,':',color='#8c1515')
    ax.fill_between(np.sort(x), lower_PI, upper_PI, color='black', alpha=0.05)


  # scatter plots
  z = plot_data.iloc[:,2].unique()
  z.sort()
  len_z = len(z)
  plot_shape = rd.sample(shape_list, len_z)
  plot_color = rd.sample(color_list, len_z)
  
  # sample size
  ax.scatter(0, 0, marker='.', color='w', label='Sample Size:{}'.format(n))
  
  loc = 0
  for label in z:
    x_ = plot_data.loc[plot_data.iloc[:,2]==label,:].iloc[:,0]
    y_ = plot_data.loc[plot_data.iloc[:,2]==label,:].iloc[:,1]
    ax.scatter(x_, y_, marker=plot_shape[loc], color=plot_color[loc], label=label)
    loc +=1


  ax.legend(loc=legend_loc, bbox_to_anchor=(1.6, 0.62),fontsize=12)   # legend box on the right
  #ax.legend(loc=legend_loc,fontsize=12)   # legend box within the plot
  
 
  # plt.close()

  return ax
  
  
  
# plot the parity chart
def regression_plot(ax, plot_data, force_intercept_origin=0, plot_interval=['confidence'],legend_loc='lower right'):

  """
    plot parity chart: scatter plot with a parity line, a regression line, and a confidence interval

    INPUTS
    - ax is the subplot ax to plot on
    - plot_data is the processed data
    - force_intercept_origin decides which regression to use
    - plot_interval can be ['confidence','prediction'] or either one of those in the list
    - plot_lim is the limit of the x and y axes
    - legend_loc is the location of the legend

    OUTPUT
    - ax is the plotted parity chart
  """
      
  # 重置绘图数据索引
  plot_data.reset_index(drop=True, inplace=True)
  
  # label name
  xlabel = 'True Release Rate (kg/h)'
  ylabel = 'Reported Release Rate (kg/h)'
  
  # x_lim, y_lim
  xlim = plot_data.loc[:, xlabel].max() * 1.2
  ylim = plot_data.loc[:, ylabel].max() * 1.2
  
  # set up plot
  ax.set_xlabel(xlabel,fontsize=13)
  ax.set_ylabel(ylabel,fontsize=13)
  ax.set_xlim([0, xlim])
  ax.set_ylim([0, xlim])

  # parity line
  x_lim = np.array([0, xlim])
  y_lim = np.array([0, xlim])
  ax.plot(x_lim, y_lim, color='black',label = 'Parity line')

  # define x and y data points
  x = plot_data.loc[:, xlabel].values
  y = plot_data.loc[:, ylabel].values

  # regression
  if force_intercept_origin == 0:
    n,pearson_corr,slope,intercept,r_value,x_lim,y_pred,lower_CI,upper_CI,lower_PI,upper_PI,residual,std_err = linreg_results(x,y)
  elif force_intercept_origin == 1:
    n,slope,r_squared,x_lim,y_pred,lower_CI,upper_CI,lower_PI,upper_PI,residual,std_err = linreg_results_no_intercept(x,y)


  # plot regression line
  if force_intercept_origin == 0:
    if intercept<0:   # label differently depending on intercept
      ax.plot(x,y_pred,'-',color='#8c1515',linewidth=2.5,
              label = 'Best fit $R^{2}$ = %0.2f \n$y$ = %0.2f$x$%0.2f' % (r_value**2,slope,intercept))
    elif intercept>=0:
      ax.plot(x,y_pred,'-',color='#8c1515',linewidth=2.5,
              label = 'Best fit $R^{2}$ = %0.2f \n$y$ = %0.2f$x+$%0.2f' % (r_value**2,slope,intercept))
  elif force_intercept_origin ==1:
      ax.plot(x,y_pred,'-',color='#8c1515',linewidth=2.5,
            label = 'Best fit $R^{2}$ = %0.2f \n$y$ = %0.2f$x$' % (r_squared,slope))

  # plot intervals
  if 'confidence' in plot_interval:
    ax.plot(np.sort(x),upper_CI,':',color='black', label='95% CI')
    ax.plot(np.sort(x),lower_CI,':',color='black')
    ax.fill_between(np.sort(x), lower_CI, upper_CI, color='black', alpha=0.05)
  if 'prediction' in plot_interval:
    ax.plot(np.sort(x),upper_PI,':',color='#8c1515', label='95% PI')
    ax.plot(np.sort(x),lower_PI,':',color='#8c1515')
    ax.fill_between(np.sort(x), lower_PI, upper_PI, color='black', alpha=0.05)


  # scatter plots
  ax.scatter(x, y, marker='o', color='#B83A4B', facecolors='none', label='Data Point')
  
  # sample size
  ax.scatter(0, 0, marker='.', color='w', label='Sample Size:{}'.format(n))
  
  # errorbar
  y_u = plot_data['EmissionRateUpper'] - y
  y_l = y - plot_data['EmissionRateLower']
  yerr = [y_l, y_u]
  
  plt.errorbar(x=x, y=y, yerr=yerr, linestyle="none")
  
  
  ax.legend(loc=legend_loc, bbox_to_anchor=(1.6, 0.62),fontsize=12)   # legend box on the right


  return ax
  

def CS_timeSeries_plot(start_obs, end_obs, EmissionRate_obs ,start_det, end_det, EmissionRate_det, TailDuration, HeadDuration, true_data, title, save_path):
    """
    start_obs:观测开始时间
    end_obs:观测结束时间
    EmissionRate_obs:观测气体释放速度
    start_det:设备监测开始时间
    end_det:设备监测结束时间
    EmissionRate_det:设备监测气体释放速度
    true_data:真实气体释放数据
    wind_data:真实风速数据
    """
    
    ### 提取真实气体释放数据 ###
    
    # 计算提取时间范围
    left_time = min(start_obs, end_obs, start_det, end_det)
    right_time = max(start_obs, end_obs, start_det, end_det)
    Extended_time = (right_time - left_time) * 1
    shift_Stime = left_time - Extended_time
    shift_Etime = right_time + Extended_time
    
    # 筛选提取范围内的数据
    true_index = true_data['Datetime (UTC)'].between(shift_Stime, shift_Etime, inclusive='both')
    true_data = true_data.loc[true_index, ['Datetime (UTC)','Release Rate (kg/h)']]
    true_data.set_index('Datetime (UTC)', inplace=True)
    
    
    ### 生成设备监测数据 ###
    detect_data = pd.Series(index=[start_det, end_det], data=[EmissionRate_det, EmissionRate_det])
    
    # 绘制时间延迟
    plt.plot(detect_data, c='white', lw=0.1, label='Tail Duaration:%.2f min'%TailDuration)
    
    if HeadDuration>=0:
      plt.plot(detect_data, c='white', lw=0.1, label='before detected:%.2f min'%HeadDuration)
    else:
      plt.plot(detect_data, c='white', lw=0.1, label='after detected:%.2f min'%(-HeadDuration))
    
    # 绘制时序图
    plt.plot(detect_data, c='blue', lw=1, label='Reported Release Rate:%.2f (Kg/h)'%EmissionRate_det)
    plt.plot(true_data, c='black', lw=1, label='True Release Rate:%.2f (Kg/h)'%EmissionRate_obs)
  
    # 标注时间起点线
    plt.axvline(start_obs, ls=":", c="g", lw=1, label='start time')
    plt.axvline(end_obs, ls=":", c="r", lw=1, label='end time')
    
    # 图标签处理
    plt.ylabel('True Release Rate (Kg/h)',fontsize=12)
    plt.xlabel('Datetime (UTC)',fontsize=12)
    plt.xticks(rotation=90)
    plt.legend(loc='upper right', bbox_to_anchor=(1.55, 1),fontsize=8)
    plt.title(title)
    plt.savefig(save_path + title + '.jpg', dpi=200, bbox_inches = 'tight')
    plt.close()
    
    

def timeSeries_plot(start_obs, end_obs, EmissionRate_obs ,start_det, end_det, EmissionRate_det, true_data, title, save_path):
    """
    start_obs:观测开始时间
    end_obs:观测结束时间
    EmissionRate_obs:观测气体释放速度
    start_det:设备监测开始时间
    end_det:设备监测结束时间
    EmissionRate_det:设备监测气体释放速度
    true_data:真实气体释放数据
    wind_data:真实风速数据
    """
    
    ### 提取真实气体释放数据 ###
    
    # 计算提取时间范围
    left_time = min(start_obs, end_obs, start_det, end_det)
    right_time = max(start_obs, end_obs, start_det, end_det)
    Extended_time = (right_time - left_time) * 1
    shift_Stime = left_time - Extended_time
    shift_Etime = right_time + Extended_time
    
    # 筛选提取范围内的数据
    true_index = true_data['Datetime (UTC)'].between(shift_Stime, shift_Etime, inclusive='both')
    true_data = true_data.loc[true_index, ['Datetime (UTC)','True Release Rate (kg/h)']]
    true_data.set_index('Datetime (UTC)', inplace=True)
    
    ### 生成设备监测数据 ###
    detect_data = pd.Series(index=[start_det, end_det], data=[EmissionRate_det, EmissionRate_det])
    
    
    # 绘制时序图
    plt.plot(detect_data, c='blue', lw=1, label='Reported Release Rate:%.2f (kg/h)'%EmissionRate_det)
    plt.plot(true_data, c='black', lw=1, label='True Release Rate:%.2f (kg/h)'%EmissionRate_obs)
  
    # 标注时间起点线
    plt.axvline(start_obs, ls=":", c="g", lw=1, label='start time')
    plt.axvline(end_obs, ls=":", c="r", lw=1, label='end time')
    
    # 图标签处理
    plt.ylabel('True Release Rate (kg/h)',fontsize=12)
    plt.xlabel('Datetime (UTC)',fontsize=12)
    plt.xticks(rotation=90)
    plt.legend(loc='upper right', bbox_to_anchor=(1.55, 1),fontsize=8)
    plt.title(title)
    plt.savefig(save_path + title + '.jpg', dpi=200, bbox_inches = 'tight')
    plt.close()