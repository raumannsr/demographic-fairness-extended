import pandas as pd
import definitions

class ISICDataset:
    def get_train_and_test_sets(self, filename_training, filename_test, random_seed, training_frac):
        df = pd.read_csv(filename_training)

        query_result = df[(df['sex'] == 'female') & (df['age_approx'] < 60) & (df['target'] == 1)]
        mfa_random_rows = query_result.sample(n=len(query_result), random_state=random_seed)

        query_result = df[(df['sex'] == 'female') & (df['age_approx'] >= 60) & (df['target'] == 1)]
        mfb_random_rows = query_result.sample(n=len(query_result), random_state=random_seed)

        query_result = df[(df['sex'] == 'male') & (df['age_approx'] < 60) & (df['target'] == 1)]
        mma_random_rows = query_result.sample(n=len(query_result), random_state=random_seed)

        query_result = df[(df['sex'] == 'male') & (df['age_approx'] >= 60) & (df['target'] == 1)]
        mmb_random_rows = query_result.sample(n=len(query_result), random_state=random_seed)

        query_result = df[(df['sex'] == 'female') & (df['age_approx'] < 60) & (df['target'] == 0)]
        bfa_random_rows = query_result.sample(n=len(query_result), random_state=random_seed)

        query_result = df[(df['sex'] == 'female') & (df['age_approx'] >= 60) & (df['target'] == 0)]
        bfb_random_rows = query_result.sample(n=len(query_result), random_state=random_seed)

        query_result = df[(df['sex'] == 'male') & (df['age_approx'] < 60) & (df['target'] == 0)]
        bma_random_rows = query_result.sample(n=len(query_result), random_state=random_seed)

        query_result = df[(df['sex'] == 'male') & (df['age_approx'] >= 60) & (df['target'] == 0)]
        bmb_random_rows = query_result.sample(n=len(query_result), random_state=random_seed)

        # Read training data and split in training and validation
        mfa_train_rows = mfa_random_rows.sample(frac=training_frac, random_state=random_seed)
        mfb_train_rows = mfb_random_rows.sample(frac=training_frac, random_state=random_seed)
        mma_train_rows = mma_random_rows.sample(frac=training_frac, random_state=random_seed)
        mmb_train_rows = mmb_random_rows.sample(frac=training_frac, random_state=random_seed)
        bfa_train_rows = bfa_random_rows.sample(frac=training_frac, random_state=random_seed)
        bfb_train_rows = bfb_random_rows.sample(frac=training_frac, random_state=random_seed)
        bma_train_rows = bma_random_rows.sample(frac=training_frac, random_state=random_seed)
        bmb_train_rows = bmb_random_rows.sample(frac=training_frac, random_state=random_seed)

        # Create validation data
        merged_df = pd.merge(mfa_random_rows, mfa_train_rows, how='outer', indicator=True)
        mfa_val_rows = merged_df[merged_df['_merge'] != 'both'].drop('_merge', axis=1)
        merged_df = pd.merge(mfb_random_rows, mfb_train_rows, how='outer', indicator=True)
        mfb_val_rows = merged_df[merged_df['_merge'] != 'both'].drop('_merge', axis=1)
        merged_df = pd.merge(mma_random_rows, mma_train_rows, how='outer', indicator=True)
        mma_val_rows = merged_df[merged_df['_merge'] != 'both'].drop('_merge', axis=1)
        merged_df = pd.merge(mmb_random_rows, mmb_train_rows, how='outer', indicator=True)
        mmb_val_rows = merged_df[merged_df['_merge'] != 'both'].drop('_merge', axis=1)
        merged_df = pd.merge(bfa_random_rows, bfa_train_rows, how='outer', indicator=True)
        bfa_val_rows = merged_df[merged_df['_merge'] != 'both'].drop('_merge', axis=1)
        merged_df = pd.merge(bfb_random_rows, bfb_train_rows, how='outer', indicator=True)
        bfb_val_rows = merged_df[merged_df['_merge'] != 'both'].drop('_merge', axis=1)
        merged_df = pd.merge(bma_random_rows, bma_train_rows, how='outer', indicator=True)
        bma_val_rows = merged_df[merged_df['_merge'] != 'both'].drop('_merge', axis=1)
        merged_df = pd.merge(bmb_random_rows, bmb_train_rows, how='outer', indicator=True)
        bmb_val_rows = merged_df[merged_df['_merge'] != 'both'].drop('_merge', axis=1)

        train_df = pd.concat(
            [mfa_train_rows, mfb_train_rows, mma_train_rows, mmb_train_rows, bfa_train_rows, bfb_train_rows,
             bma_train_rows, bmb_train_rows], axis=0)
        val_df = pd.concat(
            [mfa_val_rows, mfb_val_rows, mma_val_rows, mmb_val_rows, bfa_val_rows, bfb_val_rows, bma_val_rows,
             bmb_val_rows], axis=0)
        test_df = pd.read_csv(filename_test)

        train = definitions.Dataset(train_df['id'], train_df['target'])
        val = definitions.Dataset(val_df['id'], val_df['target'])
        test = definitions.Dataset(test_df['id'], test_df['target'])

        return train, val, test