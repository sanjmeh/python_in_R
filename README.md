# Mindshift

```
pip install -r requirements.txt
```

For Idling Detection
```
python run.py --sites jobner --idling --start 2021-12-1 --end 2021-12-31 --delta_t 10
```

For Drain Detection
```
python run.py --sites dand --drain --start 2021-8-1 --end 2021-8-31
```

For Vehicle Drain Detection
```
python run.py --sites 'EXR-645' 'EXR-466' 'MGR-184(Motor Grader)' 'MGR-185(Motor Grader)' 'EXR-592' 'MGR-21' 'EXR-685' 'EXR-591' 'MGR-97'  --drain-ind --start 2021-8-1 --end 2021-8-31
```