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

df <- read.csv(paste0('../data/', filename)) %>% 
  rename_all(~c(
    'rk_ovr','player','team','pos',
    'age','desc_games','desc_starts',
    'pass_comp','pass_att','pass_yds','pass_td','pass_int',
    'rush_att','rush_yds','y_a','rush_td',
    'rec_tgt','rec_rec','rec_yds','y_r','rec_td',
    'fumb_times','fumb_fl',
    'score_tds','score_2pm','score_2pp',
    'fantpt','ppr','dkpt','fdpt','vbd',
    'rk_pos','ovrk',
    'year', 'rk_probowl', 'rk_allpro'
    )
  ) %>% 
  filter(pos != '', !is.na(fantpt)) %>% 
  mutate(
    pos = ifelse(pos == 'FB', 'RB', pos), 
    ppg = (fantpt + (0.5*rec_rec)) / desc_games,
    ) %>% 
  mutate_if(is.numeric, ~replace_na(., 999)) # this is an important line, coding missing vars to 999

# one hot encode positions 
onehot_pos <- data.frame(model.matrix(~df$pos + 0))
colnames(onehot_pos) <- substr(colnames(onehot_pos), 4, 1000)

# select features we want
df <- df %>% 
  select(-pos) %>%
  cbind(onehot_pos) %>% 
  select(
    player, team, ppg, year, age, 
    matches('^pos|^rk_|^pass_|^rush_|^rec_|^fumb_|^score_')
  )


##### 02. Create newdat and modeldat dataframes #####
newdat <- df %>% 
  mutate(lag_ppg = ppg) %>% 
  filter(year == maxyear) %>% 
  select(-year)

traindat <- df %>% 
  group_by(player) %>%
  arrange(player, year) %>%
  mutate(lag_ppg = lag(ppg)) %>%
  mutate_at(vars(matches('^pos|^rk_|^pass_|^rush_|^rec_|^fumb_|^score_')), lag) %>%
  filter(!is.na(lag_ppg)) %>%
  select(-year)


##### 03. Write out newdat and traindat to data #####
write.csv(newdat, '../data/newdat.csv', row.names = FALSE)
write.csv(traindat, '../data/traindat.csv', row.names = FALSE)
