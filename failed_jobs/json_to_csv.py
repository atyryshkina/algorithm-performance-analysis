import json
import pandas as pd
import ast
import os

def getRidOfCommandLine(job):
#   del job["command_line"]
  del job["dependencies"]
  del job["destination_params"]
  return job

def flattenMetric(job):
  for key in job["metrics"]:
    job[key]=job["metrics"][key]
  del job["metrics"]
  if "uname" in job:
    del job["uname"]
  return job

def flattenDatasets(job):
  if "datasets" in job:
    ds=job["datasets"]
    for dataset in ds:
      job[dataset['param_name']]=dataset['file_size']
      job["%s_filetype"%dataset['param_name']]=dataset['extension']
    del job["datasets"]
  return job

def recursivelyAddDict(job, d, delim):
  
  if type(d) == dict:
    for key in d:
      if type(d[key]) == dict:
        job=recursivelyAddDict(job,d[key],delim+"."+key)
      else:
        job[delim+"."+key]=d[key]
  else:
    pass
 
  return job

def flattenParameters(job):
    for key in job["parameters"]:
#         print(job["parameters"][key])
        try:
            evaled_value=ast.literal_eval(job["parameters"][key])
            if type(evaled_value) == dict:
                job=recursivelyAddDict(job, evaled_value, "parameters.%s"%key)
            else:
                job["parameters."+key]=job["parameters"][key]
        except:
            pass
    del job["parameters"]
    return job


def json_to_csv(filename):
    f=open("%s/%s.json"%(json_folder,filename), "r")

    try:
        jsonj = json.loads(f.read())
    except ValueError as e:
        print("THIS THING (%s) DOESN'T HAVE A json IN IT" % filename)
        return
    
    print("loaded")

    for i in range(len(jsonj)):
        if i%100==0:
            pass
#             print("%d/%d"%(i,len(jsonj)))
        jsonj[i]=getRidOfCommandLine(jsonj[i])
        jsonj[i]=flattenMetric(jsonj[i])
        jsonj[i]=flattenDatasets(jsonj[i])
        jsonj[i]=flattenParameters(jsonj[i])
    print("flattened json")

    df=pd.io.json.json_normalize(jsonj)
    df["create_time"]=pd.to_datetime(df["create_time"],infer_datetime_format=True)
    df["update_time"]=pd.to_datetime(df["update_time"],infer_datetime_format=True)
    df["runtime"]=df["update_time"]-df["create_time"]
    df["runtime"]=[i.total_seconds() for i in df["runtime"]]
    # df=df.set_index("id")
    print("number of jobs: %r" % df.shape[0])
        
    df.to_csv("%s/%s.csv"%(csv_folder,filename), index=False)
    print("saved")

def getfiles(dirpath):
    a = [s for s in os.listdir(dirpath)
         if os.path.isfile(os.path.join(dirpath, s))]
    a.sort(key=lambda s: os.path.getmtime(os.path.join(dirpath, s)), reverse=True)
    return a

# json_folder="dump_errors"
json_folder="walltime_dump"
# csv_folder="csv_dump_errors"
csv_folder="walltime_csv"

finished=getfiles(csv_folder)
filenames=getfiles(json_folder)
# print("length ", len(filenames))
# for file in finished:
#   try:
#     filenames.remove(file[:-4]+".json")
#   except:
#     pass
# print("length ", len(filenames))
try:
  filenames.remove(".DS_Store")
except:
  pass
# filenames.remove(".ipynb_checkpoints")

total_amount=(len(filenames))
for i in range(len(filenames)):
    if i % 100 == 0:
        print("%d/%d"%(i,total_amount))
    print(filenames[i][:-5])
    json_to_csv(filenames[i][:-5])
    print("")
