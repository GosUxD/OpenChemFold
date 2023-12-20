import pandas as pd
import matplotlib.pyplot as plt
#read your sub here
df=pd.read_csv("submission_squeezeformer.csv")
#some parameters
font_size=6
id1=269545321
id2=269724007
reshape1=391
reshape2=457
#get predictions
pred_DMS=df[id1:id2+1]['reactivity_DMS_MaP'].to_numpy().reshape(reshape1,reshape2)
pred_2A3=df[id1:id2+1]['reactivity_2A3_MaP'].to_numpy().reshape(reshape1,reshape2)
#plot mutate and map
fig = plt.figure()
plt.subplot(121)
plt.title(f'reactivity_DMS_MaP', fontsize=font_size)
plt.imshow(pred_DMS,vmin=0,vmax=1, cmap='gray_r')
plt.subplot(122)
plt.title(f'reactivity_2A3_MaP', fontsize=font_size)
plt.imshow(pred_2A3,vmin=0,vmax=1, cmap='gray_r')
plt.tight_layout()
plt.savefig(f"plot_squeezeformer.png",dpi=500)
plt.clf()
plt.close()