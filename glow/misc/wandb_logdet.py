import pandas as pd
import wandb
import wandb,os, json
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_theme(style="whitegrid")

run = wandb.init()
file_name ='logdet_batch_0'
run_id = '3ko11be1'
table_name = f'{file_name}.table.json'
artifact = run.use_artifact(f'vincenthu/chaiyujin_number/run-{run_id}-{file_name}:v0', type='run_table')
artifact_dir = artifact.download()
print(artifact_dir)
with open(os.path.join(artifact_dir, table_name)) as f:
    data = json.load(f)
df = pd.DataFrame(data['data'], columns = data['columns'])
print(df.head())
df.drop(columns=['batch_id'])
#print(df.to_latex(index=False))
#construct pandas


#draw pdfs here by filtering...
#https://seaborn.pydata.org/tutorial/distributions.html#kernel-density-estimation

#https://stackoverflow.com/questions/55916061/no-legends-seaborn-lineplot


# importing package
import matplotlib.pyplot as plt

df_copy = df.copy()
y = df_copy[['logdet']].to_numpy().squeeze()
#plt.plot(range(y.shape[0]), y, label = '')


plt.plot(range(y.shape[0]), y)
plt.axvline(x=34, color='r', linestyle='--', label='downsize 32->16')
plt.axvline(x=68, color='r', linestyle='--', label='downsize 16->8')

plt.xlabel("level number")
plt.ylabel("accumulated avg-logdet")
plt.subplots_adjust(bottom=.13, left=.18)

plt.legend()
plt.show()
save_path = os.path.join("/home/thu/lab/stylegan3_main/chaiyujin-glow",f'logdet.pdf')
print(save_path)
plt.savefig(save_path)