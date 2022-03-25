library(tidyverse)
library(data.table)
library(janitor)

sitedata <- readRDS('~/omni/ritesh/sitedata.RDS')
cr_py_shell_file <- function(strt = '2022-1-1',end = '2022-1-15',output = "~/omni/ritesh/Mindshift-GH/WH-Report/cronrun.sh"){
  s2 <- sitedata$name %>% tolower
  topics <- sitedata$topic
  osite <- sitedata$name
  shrt <- sitedata$short
  1:5 %>% 
    map_chr(
      ~ glue("
  python3 ~/omni/ritesh/Mindshift-GH/WH-Report/run.py -s {s2[.x]} --start {strt} --end {end} --elm_file '/data/backups/{topics[[.x]]}/*.RDS' --fuel_file '/home/sanjay/omni/{osite[.x]}_fdt.RDS.backup' --event_file '/home/sanjay/omni/{osite[.x]}_eventdt.RDS' --output_file '{shrt[.x]}_{strt}_{end}.csv'
  ")
    ) %>% 
    write_lines(file = output)
}