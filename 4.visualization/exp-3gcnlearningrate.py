import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
plt.style.use('seaborn-whitegrid')
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus']=False
xlabel=[0.001,0.005,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05]
g3={
"AUC":[0.5712999658772555, 0.5652270211088225, 0.5599010912102191, 0.5580822822798555, 0.5596934220318946, 0.5606131807729937, 0.5500364013257064, 0.556410076170186, 0.5555110226224468, 0.5597992460560172, 0.5651753661198538],
"Recall":[0.14316205533596837, 0.13090909090909092, 0.12023715415019764, 0.1166798418972332, 0.1198418972332016, 0.12158102766798418, 0.10047430830039525, 0.113201581027668, 0.11138339920948617, 0.12, 0.13067193675889327],
"Precision":[0.4942049062049062, 0.5646394716394716, 0.5608484848484848, 0.4932282162282163, 0.5221733821733822, 0.5767619047619048, 0.506904761904762, 0.5524285714285714, 0.5881904761904762, 0.56008658008658, 0.6236507936507936],
"F1":[0.21458250743904736, 0.2010292229948452, 0.18813255590741687, 0.17806652194623798, 0.18609088433916018, 0.19509000263939086, 0.16493294779212464, 0.18354596899824932, 0.1835086691086691, 0.19037134372191625, 0.2119058187023704]
}
    
g4={
"AUC":[0.5759574755415826, 0.570044090058538, 0.5670350771754127, 0.5783707050341677, 0.5783346877199748, 0.5685776603172362, 0.5813664628117131, 0.5710052284602055, 0.5813680740615184, 0.5731143874701548, 0.5693445670614466],
"Recall":[0.15242165242165245, 0.14056980056980056, 0.1345868945868946, 0.1572079772079772, 0.15715099715099715, 0.13760683760683762, 0.1633048433048433, 0.1425071225071225, 0.16324786324786325, 0.1467806267806268, 0.13931623931623932],
"Precision":[0.5210468975468976, 0.5062112332112332, 0.47599744699744695, 0.5376666666666666, 0.5485526695526696, 0.5180451770451772, 0.5193739593739595, 0.4760940170940171, 0.513902985902986, 0.4800080475080475, 0.43034212845977554],
"F1":[0.23057997429417143, 0.21629176193882077, 0.20666200365440318, 0.23937167430270878, 0.23861755657265885, 0.21273683405591406, 0.24175041313970702, 0.21489443879766462, 0.24251822247393004, 0.21963979500354772, 0.2058223969988676]    
}
g5={
"AUC":[0.5659475563951382, 0.5693274588999004, 0.5857461214846083, 0.5753169284271359, 0.5840551675254828, 0.5670964319937202, 0.5705480783901882, 0.5722864486008936, 0.5787592912336101, 0.581258728741884, 0.5799422872570907],
"Recall":[0.13243697478991595, 0.13936134453781515, 0.1721344537815126, 0.1511596638655462, 0.16867226890756304, 0.13472268907563026, 0.14171428571428574, 0.14514285714285716, 0.15818487394957984, 0.1631596638655462, 0.1604705882352941],
"Precision":[0.4735790875790876, 0.4309618730083127, 0.4927259799024505, 0.5187737458619812, 0.5199159174159174, 0.475478354978355, 0.4471290768055474, 0.4828855588855589, 0.46842312915842327, 0.48814748976513683, 0.4938299478299478],
"F1":[0.20385660668190445, 0.20728868157945524, 0.24990999231979716, 0.2296357277260061, 0.25106420011971486, 0.20724732252461786, 0.2121685027358525, 0.21911555808495797, 0.2333815473121949, 0.23964891614472475, 0.23807267536245416]
}

df3=pd.DataFrame(g3,index=xlabel)
df4=pd.DataFrame(g4,index=xlabel)
df5=pd.DataFrame(g5,index=xlabel)

data=df5
t='S3'

palette = sns.color_palette("colorblind",4)
print(palette)
ax=sns.lineplot(data=data,
            markers=True, dashes=False,
            linewidth=2.5,markersize=13,
            palette=palette
            )
plt.xlabel("GCN学习率",fontsize=15)
plt.ylabel("Score",fontsize=15)
plt.title(t,fontsize='16',fontweight='bold') 
plt.setp(ax.get_legend().get_texts(), fontsize='15') # for legend text
plt.show()

