import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot


df = pd.read_csv("bordocole.csv")

date_time = pd.to_datetime(df.pop('date'), format='%m/%d/%y')

#plot data here
plot_cols = ['formrate']
plot_features = df[plot_cols]
plot_features.index = date_time
_ = plot_features.plot(subplots=True)

plt.show()

X = df[['chngUt-1', 'chngaaa-tr10t-1', 'tobQt-3', 'DfASB1', 'DfASB2', 'regpercapt-1', 'chngHOR' ]]
y = df['formrate']
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model


model = sm.OLS(y, X).fit()
predictions = model.predict(X) # make the predictions by the model

print(model.summary())

write_path = '/Users/joeyedell/Desktop/seminarproject/chngHOR.csv'
with open(write_path, 'w') as f:
    f.write(model.summary().as_csv())



#
#
# print(df.head())
# df.plot()
# pyplot.show()
#
# keys = ['chng real priv inventories', 'PRS30006013', 'deathrate', 'tr10 ', 'DPIC96', 'SPASTT01USQ661N', 'PITGCG01USQ661N', 'PRS84006173', 'civ unemply rate', 'OUTMS', 'population', 'MABMM301USQ657S', 'ULQELP01USQ657S', 'LCULMN01USQ661S', 'CP', 'HOHWMN02USQ065S', 'CSINFT02USQ460S', 'caputz index', 'NFB R output ph', 'RSAHORUSQ156S', 'LRAC25MAUSQ156S', 'real GPDI fixed equip', 'RGDP', 'LFEAAGTTUSQ647S', 'IRSTFR01USQ156N', 'LRIN25MAUSQ156S', 'PRS88003173', 'LREM64TTUSQ156S', 'aaa', 'LES1252881600Q', 'LFWA24TTUSQ647S', 'deliquency rate of comm and ind loans', 'GPDI', 'CAPUTL', 'USAGFCFQDSMEI', 'IMPGSC', 'nonfarm labor share', 'LLBAC', 'A067RP1Q027SBEA', 'births', 'GDPIC', 'IRSTCI01USQ156N', 'baa', 'EXPGSC', 'DPSACBQ158SBOG', 'PRMNTO01USQ657S', 'WASCUR', 'M1V', 'NROU', 'IP index', 'IRLTLT01USQ156N', 'USAPROINDQISMEI', 'DRIWCIL', 'PINCOME']
# vals = [31, 19, 22, 18, 22, 27, 30, 19, 23, 26, 18, 14, 16, 22, 25, 15, 27, 16, 23, 31, 29, 21, 29, 27, 23, 27, 17, 17, 29, 26, 18, 25, 22, 23, 21, 21, 19, 22, 20, 30, 22, 19, 25, 22, 19, 21, 21, 27, 22, 25, 18, 19, 24, 20]
#
# dict = dict.fromkeys(keys,0)
#
# for i in range(len(keys)):
#     dict[keys[i]] = vals[i]
#
# print(dict)
#
# bestkeys = []
# bestvals = []
#
# for i in range(len(keys)):
#     if vals[i] < 30 or keys[i] == 'births':
#         continue
#     else:
#         bestkeys.append(keys[i])
#         bestvals.append(vals[i])
#
# print(bestkeys)
# print(len(keys))
#
# plt.bar(x = range(len(bestkeys)),
#         height=bestvals)
# axis = plt.gca()
# axis.set_xticks(range(len(bestkeys)))
# _ = axis.set_xticklabels(bestkeys, rotation=90)
# plt.show()




