# Mindshift Pilferage Three algorithms


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

# Mindshift Stationary-Movement Allmods algorithm :

1. First run the below command in terminal to install all dependencies required for main algorithm
```
pip install -r requirements.txt
```

2. Go to *d_config.py* file to set the datapaths according to your system directory. 

3. After changing the *d_config* file, run below command in terminal. 
```
python dist_allmods.py
```

