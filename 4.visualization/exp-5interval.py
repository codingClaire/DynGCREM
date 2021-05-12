import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
plt.style.use('seaborn-whitegrid')
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus']=False
xlabel=[15,30,45,60]

g4={
"AUC":[0.574178, 0.575535,0.560191,0.555020],
"Recall":[0.148718,0.151396,0.120798, 0.110541],
"Precision":[0.546672,0.632452,0.463724,0.388263],
"F1":[0.229083,0.236534,0.188096,0.166738]
}

df4=pd.DataFrame(g4,index=xlabel)


palette = sns.color_palette("colorblind",4)
print(palette)
ax=sns.lineplot(data=df4,
            markers=True, dashes=False,
            linewidth=2.5,markersize=13,
            palette=palette
            )
plt.xlabel("动态图时间间隔划分",fontsize=15)
plt.ylabel("Score",fontsize=15)
plt.title('S2',fontsize='16',fontweight='bold') 
plt.setp(ax.get_legend().get_texts(), fontsize='15') # for legend text
plt.show()

