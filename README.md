# 1. Mindshift Pilferage Three algorithms


```
pip install -r requirements.txt
```

1. If you want to generate Three algorithm files for all termids, put the command this way in your terminal, 

```
python run.py -IA '_your input allmods file path_' -IC '_your input cst file path'  -O '_output folder path_' -t 
```

2. If you have specific termids to run the algorithms for, put the command this way, (put space-separated termids after '-t')

```
python run.py -IA '_your input allmods file path_' -IC '_your input cst file path' -O '_output folder path_' -t 1204000785 1204000506
```

# 2. Mindshift Synthetic-Enriched CST algorithm :

Terminal command instruction:
```
python synthetic_CST.py cst_data_path (space) ignition_data_path (space) hectronics_data_path (space) output_data_path
```
**prerequisites:** 

       *i. You must have both cst and ign data in RDS format, only for required vehicles.*

       *ii. You must have Hectronics Dispensing data in csv format.*
       
       *iii. Input cst must contain mentioned columns : **regNumb**,**termid**,**ts**,**latitude**,**longitude**,**currentFuelVolumeTank1**,**mine**,**class**. Others are optional*

       *iv.  Input ignition must contain mentioned columns : **termid**,**strt**,**stop**,**veh**. Others are optional.*


# 3. Mindshift ID Event algorithm:

Terminal command instruction 
```
python synthetic_CST_ID_event.py enriched_cst_data_path (space) ignition_data_path (space) output_data_path
```
