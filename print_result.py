import yaml
import csv
# result = {}
import pandas as pd
with open("result.yml", 'r') as stream:
#    if 'name' in stream:
#        del stream['name']
    # print(yaml.load(stream))
    result = yaml.load(stream)
    
########################################
print('-'*20)
print('result: {}'.format(result))
if 'name' in result:
    del result['name']
print(result)

#######################################
# convert to csv file
with open("result.csv", "w") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(result.keys())
    # writer.writecolumn(result.keys())
    writer.writerows(zip(*result.values()))
    # writer.writecolumns(zip(*result.values()))

pd.read_csv('result.csv').T.to_csv('result-Tanspose.csv', header=False)
