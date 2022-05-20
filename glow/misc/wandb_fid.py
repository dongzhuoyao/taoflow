import pandas as pd
import wandb
import wandb,os, json

table_name = 'fid_r2.table.json'
run = wandb.init()
artifact = run.use_artifact('vincenthu/chaiyujin_number/run-kq1ox9ey-fid_r2:v0', type='run_table')
artifact_dir = artifact.download()
print(artifact_dir)
with open(os.path.join(artifact_dir, table_name)) as f:
    data = json.load(f)
df = pd.DataFrame(data['data'], columns = data['columns'])

print(df[['name']])
df = df[['attr_name', 'mask_name', 'level', 'fid', 'kid', 'ics', 'ppl']]
print(df.head())
#print(df.describe())
#print(df.to_latex(index=False))
#construct pandas


# importing package
import matplotlib.pyplot as plt

# create data
x = [16,32,64,96, 100]
target_couple = [('Male', 'hair'),
                             ('Male', 'skin'), ('Black_Hair', 'hair'),
                             ('Wearing_Hat', 'hair'),('Smiling', 'skin'),
                             ('Eyeglasses', 'skin')
                             ]

for _metric in ['fid', 'kid']:
    for attr_name, mask_name in target_couple:
        df_copy = df.copy()
        df_copy = df_copy[df_copy['attr_name'] == attr_name]
        df_copy = df_copy[df_copy['mask_name'] == mask_name]

        y = list(df_copy[_metric])[1:]
        # plot lines
        plt.plot(x, y, label = f'{attr_name}_on_{mask_name}')

    plt.xlabel("level number")
    plt.ylabel(_metric.upper())

    plt.legend()
    plt.show()
    save_path = os.path.join("/home/thu/lab/stylegan3_main/chaiyujin-glow",f'fid_all_{_metric}.pdf')
    print(save_path)
    plt.savefig(save_path)




#draw pdfs here by filtering...
#https://seaborn.pydata.org/tutorial/distributions.html#kernel-density-estimation
