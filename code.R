library(baseballr)
library(dplyr)
library(tidyr)
#import data
vid_df <-
  read.csv('pitch_desc_samp.csv', fileEncoding = 'UTF-8')[, c(1:2, 4:5, 7, 9:14, 16)]
#add names
names(vid_df) <-
  c(
    'pitch_type',
    'release_speed',
    'pitcher_name',
    'player_name',
    'release_spin_rate',
    'zone',
    'game_date',
    'count',
    'inning',
    'outcome',
    'des',
    'url'
  )
#create columns to join on in baseball savant table
vid_df <-
  tibble(
    vid_df %>% separate(count, c('balls', 'strikes'), sep = '-') %>% separate(inning, c('inning_topbot', 'inning'), sep =
                                                                                ' ') %>% mutate(type = ifelse(outcome == 'ball', 'B', 'S')) %>% select(-outcome)
  )
#adjust data types to join with baseball savant table
vid_df$game_date <- as.Date(vid_df$game_date)
vid_df[, c('balls', 'strikes', 'inning')] <-
  lapply(vid_df[, c('balls', 'strikes', 'inning')], as.numeric)
#note pitcher names don't have accents
vid_df$pitcher_name <-
  gsub(
    iconv(vid_df$pitcher_name, to = "ASCII//TRANSLIT"),
    pattern = '\\s',
    replacement = ''
  )
#add the name of the video
vid_df$vid_name <-
  gsub(
    file.info(paste0('vids/', list.files('vids'))) %>% arrange(mtime) %>% rownames(),
    pattern = '.*/',
    replacement = ''
  )


#scrape more pitch information for all pitches in MLB up to the point when scraped
#done in a daily fashion so the scrape function call doesn't crash
daily <- data.frame()
range <-
  seq(as.Date("2021/04/01"), as.Date("2021/5/07"), by = "day")
for (i in c(1:length(range))) {
  daily <-
    rbind(daily,
          scrape_statcast_savant(start_date = range[i], end_date = range[i]))
}

#this will search for names of the pitchers which are not currently included in the dataset
chadwick <- get_chadwick_lu()
#there are acronym names with spaces in one source but not the lookup so the names have eliminate all whitespace
chadwick$pitcher_look <-
  paste0(
    gsub(
      chadwick$name_last,
      pattern = '\\s',
      replacement = ''
    ),
    ',',
    gsub(
      chadwick$name_first,
      pattern = '\\s',
      replacement = ''
    )
  )
daily1 <-
  daily %>% left_join(chadwick %>% select(key_mlbam, pitcher_look),
                      by = c('pitcher' = 'key_mlbam'))

#note there are some names which are not the same in the lookup as in mlbam here are some that I found
joiner <-
  data.frame(
    pitcher_look = c(
      'Boyd,Matt',
      'Underwood,Duane',
      'McCullers,Lance',
      'King,Mike',
      'Honeywell,Brent',
      'Lakins,Travis',
      'Pérez,Cionel',
      'López,Yoan'
    ),
    pitcher_mlbam = c(
      'Boyd,Matthew',
      'UnderwoodJr.,Duane',
      'McCullersJr.,Lance',
      'King,Michael',
      'HoneywellJr,Brent',
      'LakinsSr.,Travis',
      'Perez,Cionel',
      'Lopez,Yoan'
    )
  )
daily2 <-
  daily1 %>% left_join(joiner) %>% mutate(pitcher_name = tolower(ifelse(
    is.na(pitcher_mlbam), pitcher_look, pitcher_mlbam
  ))) %>%
  select(-c(pitcher_mlbam, pitcher_look))

#finally join the dataset to get more pitch info
vid_df_join <-
  vid_df %>% mutate(pitcher_name = tolower(pitcher_name)) %>%
  left_join(
    daily2 %>% select(
      pitch_type,
      game_date,
      game_pk,
      release_speed,
      release_pos_x,
      release_pos_z,
      player_name,
      pitcher_name,
      batter,
      pitcher,
      des,
      zone,
      game_type,
      plate_x,
      plate_z,
      pfx_x,
      pfx_z,
      stand,
      p_throws,
      balls,
      strikes,
      type,
      on_3b,
      on_2b,
      on_1b,
      outs_when_up,
      inning,
      inning_topbot,
      vx0,
      vy0,
      vz0,
      ax,
      ay,
      az,
      sz_top,
      sz_bot,
      effective_speed,
      release_spin_rate,
      release_extension,
      release_pos_y,
      spin_axis,
      at_bat_number,
      pitch_number,
      bat_score,
      fld_score,
      fielder_2_1,
      fielder_3,
      fielder_4,
      fielder_5,
      fielder_6,
      fielder_7,
      fielder_8,
      fielder_9,
      home_team,
      away_team,
      game_year
    )
  )
#note the names will be different for different samples of pitches, so this checks to see if any pitchers didn't get matched
vid_df_join %>% filter(is.na(release_pos_x)) %>% View()

#delete the error videos
if (length(which(
  is.na(vid_df_join$release_pos_x) |
  grepl(vid_df_join$vid_name, pattern = 'err-\\d')
)) > 0) {
  fin_vid_df_join <-
    vid_df_join[-which(
      is.na(vid_df_join$release_pos_x) |
        grepl(vid_df_join$vid_name, pattern = 'err-\\d')
    ), ]
  
  lapply(vid_df_join[which(
    is.na(vid_df_join$release_pos_x) |
      grepl(vid_df_join$vid_name, pattern = 'err-\\d')
  ), 'vid_name'], function(x)
    file.remove(paste0('vids/', x)))
  
}

#write to file
write.csv(fin_vid_df_join, 'full_pitch_info.csv', row.names = F)

##################
library(ggplot2)
fin_vid_df_join<-read.csv('full_pitch_info.csv')
#+- .7 gives me approximately 95% strikes when top and bottom conditions are met
fin_vid_df_join %>% filter(plate_z >= sz_bot &
                             plate_z <= sz_top &
                             plate_x >= -.7 &
                             plate_x <= .7) %>% group_by(type) %>% summarise(n = n()) %>% mutate(perc =
                                                                                                   n / sum(n))
ggplot(fin_vid_df_join, aes(plate_x, plate_z, col = type)) + geom_point() +
  geom_rect(mapping = aes(
    xmax = .7,
    xmin = -.7,
    ymin = mean(sz_bot),
    ymax = mean(sz_top)),
    alpha = 0,
    color = "black"
  )

model <- glm(
  type_bin ~ plate_x + I(plate_x^2) + plate_z + I(plate_z^2) + sz_top + sz_bot,
  fin_vid_df_join %>% mutate(type_bin = ifelse(type=='S',1,0)),
  family = 'binomial'
)

summary(model)

vid_df_pred <-
  fin_vid_df_join %>% mutate(type_bin = ifelse(type == 'S', 1, 0)) %>%
  cbind(data.frame(pred = predict(model, fin_vid_df_join, type = 'response')))

k <- vid_df_pred %>% mutate(
  err = type_bin - pred,
  stl = ifelse(pred < .5 &
                 type == 'S', 1, 0),
  lss = ifelse(pred > .5 &
                 type == 'B', 1, 0)
) %>% group_by(fielder_2_1) %>% summarise(
  n = n(),
  err = mean(err),
  steals = sum(stl),
  losses = sum(lss)
)
