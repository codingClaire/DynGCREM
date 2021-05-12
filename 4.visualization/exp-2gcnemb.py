import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
plt.style.use('seaborn-whitegrid')
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus']=False
xlabel=[4,8,16,32,64]
g3={
    "embsize":[4,8,16,32,64,4,8,16,32,64,4,8,16,32,64,4,8,16,32,64],
    "cate":["AUC","AUC","AUC","AUC","AUC","Recall","Recall","Recall","Recall","Recall",
    "Precision","Precision","Precision","Precision","Precision",
    "F1","F1","F1","F1","F1"],
    "score":[0.5442635556029177, 0.5579637040852651, 0.5509561555888393, 0.5616475461154575, 0.5495894866210127,
    0.08877470355731225, 0.11644268774703556, 0.10221343873517788, 0.12363636363636363, 0.09944664031620554,
0.6279999999999999, 0.48978499278499277, 0.579142857142857, 0.6394761904761905, 0.5609999999999999,
0.1502346736277771, 0.1808936566497909, 0.16671975130707056, 0.2001766655212706, 0.16514830252761287]
}

g4={
    "embsize":[4,8,16,32,64,4,8,16,32,64,4,8,16,32,64,4,8,16,32,64],
    "cate":["AUC","AUC","AUC","AUC","AUC","Recall","Recall","Recall","Recall","Recall",
    "Precision","Precision","Precision","Precision","Precision",
    "F1","F1","F1","F1","F1"],
    "score":[0.5620047025386349, 0.5885954033916386, 0.5717400544233713, 0.5553129518403995, 0.5589430128117187,
0.12433048433048435, 0.17777777777777776, 0.14393162393162393, 0.11099715099715099, 0.11834757834757834,
0.570051948051948, 0.5141489621489621, 0.5115091575091575, 0.526975468975469, 0.4654920634920635,
0.20000212109229354, 0.2592650742280723, 0.2192541593881301, 0.1786465376797342, 0.1863431280613444]
}

g5={
    "embsize":[4,8,16,32,64,4,8,16,32,64,4,8,16,32,64,4,8,16,32,64],
    "cate":["AUC","AUC","AUC","AUC","AUC","Recall","Recall","Recall","Recall","Recall",
    "Precision","Precision","Precision","Precision","Precision",
    "F1","F1","F1","F1","F1"],
    "score":[0.5688551208300584, 0.5635848270657993, 0.5647354665452616, 0.570603770515723, 0.5696199539984137,
0.13825210084033615, 0.12783193277310925, 0.13001680672268906, 0.14178151260504201, 0.13969747899159662,
0.48900504071092293, 0.42859505371014317, 0.4660038850038849, 0.5071047775753659, 0.5203780663780664,
0.2115035480194984, 0.1928358727377722, 0.200612202535336, 0.21529114872053262, 0.21749630837733794]
}

df3=pd.DataFrame(g3)
df4=pd.DataFrame(g4)
df5=pd.DataFrame(g5)

t='S1'
data=df3

ax=sns.barplot(x="embsize", 
            y="score", 
            hue="cate", 
            data=data,
            #palette= palette
            palette=sns.color_palette("colorblind",4),
            saturation=1
            )

plt.xlabel("GCN嵌入层大小",fontsize=15)
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