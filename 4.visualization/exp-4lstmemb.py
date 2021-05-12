import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
plt.style.use('seaborn-whitegrid')
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus']=False
xlabel=[4,8,16,32,64]



   
g4={
    "embsize":[4,8,16,32,64,4,8,16,32,64,4,8,16,32,64,4,8,16,32,64],
    "cate":["AUC","AUC","AUC","AUC","AUC","Recall","Recall","Recall","Recall","Recall",
    "Precision","Precision","Precision","Precision","Precision",
    "F1","F1","F1","F1","F1"],
    "score":[0.573474,0.5755,0.577347,0.564986,0.562297,
0.147293,0.1514,0.155100,0.130342,0.125071,
0.624753,0.6325,0.574686,0.586790,0.476706,
0.234094,0.2365,0.239546,0.203236,0.193379]
}


df4=pd.DataFrame(g4)


t='S2'
data=df4

ax=sns.barplot(x="embsize", 
            y="score", 
            hue="cate", 
            data=data,
            #palette= palette
            palette=sns.color_palette("colorblind",4),
            saturation=1
            )

plt.xlabel("LSTM自编码网络嵌入层大小",fontsize=15)
plt.ylabel("Score",fontsize=15)
plt.title(t,fontsize='16',fontweight='bold',loc='left') 
plt.setp(ax.get_legend().get_texts(), fontsize='17') # for legend text
plt.legend(bbox_to_anchor=(1.00, 1.1), loc='upper right', borderaxespad=0,ncol=2)

plt.show()


'''

sns.set_theme(style="whitegrid")

penguins = sns.load_dataset("penguins")
print(penguins)
# Draw a nested barplot by species and sex
g = sns.catplot(
    data=penguins, kind="bar",
    x="species", y="body_mass_g", hue="sex",
    ci="sd", palette="dark", alpha=.6, height=6
)
g.despine(left=True)
g.set_axis_labels("", "Body mass (g)")
g.legend.set_title("")
plt.show()
'''