library(dplyr)
library(tidyr)


##### Unpack command-line arguments #####
args <- commandArgs(trailingOnly = TRUE)
maxyear <- as.numeric(args[1])
if (length(maxyear) == 0) {
  stop('Error: arg 1 does not specify a maximum year for newdata')
}


##### 01. Read in data and wrangle ##### 
filename <- list.files(path = '../data', pattern = 'data_.*')

dat <- read.csv(paste0('../data/', filename)) %>% 
  rename_all(~c(
    'rk_ovr','player','team','pos',
    'age','desc_games','starts',
    'pass_comp','pass_att','pass_yds','pass_td','pass_int',
    'rush_att','rush_yds','y_a','rush_td',
    'rec_tgt','rec_rec','rec_yds','y_r','rec_td',
    'fumb_times','fumb_fl',
    'score_tds','score_2pm','score_2pp',
    'fantpt','ppr','dkpt','fdpt','rk_vbd',
    'rk_pos','ovrk',
    'year', 'rk_probowl', 'rk_allpro'
    )
  ) %>% 
  filter(pos != '', !is.na(fantpt)) %>% 
  mutate(
    pos = ifelse(pos == 'FB', 'RB', pos), 
    ppg = (fantpt + (0.5*rec_rec)) / desc_games,
    ) %>% 
  mutate_if(is.numeric, ~replace_na(., 0)) # this is an important line, coding missing vars to 0

# one hot encode positions 
onehot_pos <- data.frame(model.matrix(~ dat$pos + 0)) 
colnames(onehot_pos) <- substr(colnames(onehot_pos), 5, 1000)

# select features we want
df <- dat %>%
  select(-pos) %>%
  cbind(onehot_pos) %>% 
  select(
    player, team, starts_with('pos'), year, ppg, age, starts, 
    matches('^rk_|^y_|^pass_|^rush_|^rec_|^fumb_|^score_')
  ) 

# deal with dups by grabbing most ppg per year 
df <- df %>% 
  group_by(player, year) %>% 
  arrange(desc(ppg)) %>% # grab the most ppg 
  slice(1)  


##### 02. Create newdat and modeldat dataframes #####
# newdat
newdat <- df %>% filter(year == maxyear) 
idx <- which(!colnames(newdat) %in% c('player', 'team', 'year', 'posQB', 'posTE', 'posWR', 'posRB'))
colnames(newdat)[idx] <- paste0('lag_', colnames(newdat)[idx])

# traindat
traindat <- df %>% 
  group_by(player) %>%
  arrange(player, year) %>%
  mutate(lag_ppg = lag(ppg)) %>%
  mutate_at(vars(matches('^rk_|^y_|^pass_|^rush_|^rec_|^fumb_|^score_')), lag) %>%
  filter(!is.na(lag_ppg))
idx <- which(!colnames(traindat) %in% c('player', 'team', 'lag_ppg', 'ppg', 'year', 'posQB', 'posTE', 'posWR', 'posRB'))
colnames(traindat)[idx] <- paste0('lag_', colnames(traindat)[idx])

# check that ppg is only feature not in newdat 
colnames(traindat)[!colnames(traindat) %in% colnames(newdat)]


##### 03. Write out newdat and traindat to data #####
write.csv(newdat, '../data/newdat.csv', row.names = FALSE)
write.csv(traindat, '../data/traindat.csv', row.names = FALSE)
