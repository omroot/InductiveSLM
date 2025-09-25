

import pandas as pd 
class DeerToTriplets:
    def __init__(self):
        triplets: list[dict] = None
    def process(self, df: pd.DataFrame):
        column_names = [ 'fact 1.1',
                        'fact 1.2',
                        'fact 2.1',
                        'fact 2.2',
                        'fact 3.1',
                        'fact 3.2']

        triplets = []
        for i in range(df.shape[0]):
            observations = ''
            for name in column_names:
                observations += str(df[name].iloc[i])
                rule_template = df['rule template'].iloc[i]
                rule_type = df['rule type'].iloc[i]
                
            triplets.append({'Training Observations':  observations ,
                'Question':  f'Can you infer a general rule from the observations of type {rule_type} and template {rule_template}?' , 
            'Answer':df['rule'].iloc[i]
            })

        self.triplets = triplets





