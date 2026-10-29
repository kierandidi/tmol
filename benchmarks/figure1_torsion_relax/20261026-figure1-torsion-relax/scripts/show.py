import json,glob,sys
for f in sorted(glob.glob(sys.argv[1]+'/*.json')):
    if 'running' in f: continue
    r=json.load(open(f))
    print(f.split('/')[-1][15:-5].ljust(42), r['status'], 'batch_s',round(r.get('seconds_per_batch_samples',[0])[0],2), 'per_s',round(r.get('seconds_per_structure_samples',[0])[0],3), 'init',round(r.get('initial_score_mean',0),1),'final',round(r.get('validation_score_mean',0),1), 'mem_GB', round((r.get('peak_memory_bytes') or 0)/1e9,1), r.get('error') or '')
