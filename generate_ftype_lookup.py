import json

lookup = dict()
for line in open('filenames.txt'):
    key, value = line.strip().split('.')
    lookup[key] = value

with open('ftype_lookup.json', 'w') as fp:
    json.dump(lookup, fp)
