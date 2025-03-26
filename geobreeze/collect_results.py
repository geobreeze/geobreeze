import pandas as pd
import time
import os
import sys 



def collect_csv(output_dir, return_df='relpath', sleep=0.05):
    """ recursively create all aggregation csv files in nested folder structure """
    
    def add_lvl(df: pd.DataFrame):
        """ for pretty to_string() convert relpath to index with lvl0, lvl1, ... """
        if df.shape[0] == 0:
            return df
        df['lvl'] = df['relpath'].apply(lambda x: x.split('/'))
        max_level = max([len(l) for l in df['lvl']])
        for i in range(max_level):
            df[f'lvl{i}'] = df['lvl'].apply(lambda x: x[i] if len(x) > i else '')
        df.set_index([f'lvl{i}' for i in range(max_level)] + ['metric'], inplace=True)
        df.drop(columns=['lvl'], inplace=True)
        df.sort_index(inplace=True)
        return df
    
    def rec(abspath):
        sub_dirs = [p for p in os.listdir(abspath) if os.path.isdir(os.path.join(abspath,p))]
        sub_results = [rec(os.path.join(abspath,p)) for p in sub_dirs]
        sub_results = [r for r in sub_results if len(r) > 0]

        if len(sub_results) == 0:
            if 'results.csv' in os.listdir(abspath):
                if sleep > 0.0:
                    time.sleep(sleep)
                df = pd.read_csv(os.path.join(abspath, 'results.csv'))
                df['relpath'] = ''
                return (os.path.basename(abspath), df)
            return ()
        
        else:
            dfs = []
            for d, df in sub_results:
                df['relpath'] = df['relpath'].apply(lambda x: os.path.normpath(os.path.join(d,x)))
                dfs.append(df)
            df = pd.concat(dfs).reset_index(drop=True)

            df.to_csv(os.path.join(abspath, 'results.csv'))
            with open(os.path.join(abspath, 'results.txt'),'w+') as f:
                df_print = add_lvl(df.copy()).drop(columns=['relpath'])
                f.write(df_print.to_string())

            return (os.path.basename(abspath), df)
                
    t = rec(output_dir)
    if len(t) == 0:
        df = pd.DataFrame(columns=['metric','value','best_classifier','relpath'])
    else:
        df = t[1]
    
    if return_df == 'lvl':
        return add_lvl(df).drop(columns=['relpath'])
    elif return_df == 'relpath':
        return df
    elif return_df == 'all':
        return add_lvl(df)
    else:
        raise ValueError(f'Unknown return_df {return_df}')
    

if __name__ == '__main__':
    output_dir = sys.argv[1]
    collect_csv(output_dir)