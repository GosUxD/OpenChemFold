import pandas as pd
import re
from sklearn.model_selection import KFold

df = pd.read_csv('datamount/train_data.csv')

reactivity_columns = [col for col in df.columns if re.match(r'reactivity_\d{4}$', col)]
reactivity_error_columns = [col for col in df.columns if re.match(r'reactivity_error_\d{4}$', col)]

df['fold'] = -1

df_2A3 = df.loc[df.experiment_type=='2A3_MaP']
df_DMS = df.loc[df.experiment_type=='DMS_MaP']
df_2A3 = df_2A3.reset_index(drop=True)
df_DMS = df_DMS.reset_index(drop=True)

split = KFold(n_splits=4, random_state=2023, shuffle=True).split(df_DMS)

for fold_idx, (train_idx, valid_idx) in enumerate(split):
        df_2A3.loc[valid_idx,'fold'] = fold_idx
        df_DMS.loc[valid_idx,'fold'] = fold_idx
        print(f'fold{fold_idx}:', 'train', len(train_idx), 'valid', len(valid_idx))
        
df = pd.concat([df_2A3, df_DMS], axis=0)
df['target'] = df[reactivity_columns].values.tolist()
df['target_error'] = df[reactivity_error_columns].values.tolist()

#delete reactivity columns
df.drop(reactivity_columns, axis=1, inplace=True)
df.drop(reactivity_error_columns, axis=1, inplace=True)

df.to_parquet('datamount/train_data.parquet', compression='gzip', index=False)
