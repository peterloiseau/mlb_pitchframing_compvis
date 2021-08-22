library(dplyr)
library(tidyr)
library(tidyverse)
library(purrr)
#0 closed_mitt
#1 helmet
#2 mitt
#3 other
track <- read.csv('track_data.csv')
names(track) <- c('prob', 'class','frames', 'fps', 'vid', 'frame', 'x', 'y', 'w', 'h')
pitch_info <- read.csv('full_pitch_info.csv') %>% mutate(vid = row_number()-1)
track_pitch <- track %>% left_join(pitch_info, by='vid')
#Note that the home broadcast all have different angle, so the coordinates on the screen are scaled
track_pitch_scale <-
  track_pitch %>% group_by(home_team, class) %>% mutate(
    x = scale(x)[, 1],
    y = scale(y)[, 1],
    w = scale(w)[, 1],
    h = scale(h)[, 1]
  ) %>% ungroup() %>% select_at(1:10)

pitch_info_nest <- pitch_info  %>% nest(-vid)
track_vid <-
  track_pitch_scale %>% group_by(vid) %>% 
  filter(0%in%class) %>% mutate(time = frame / fps, len = frames/fps) %>% distinct() %>% nest(-c(vid,fps,frames,len))

track_vid_info <- track_vid %>% left_join(pitch_info_nest, by = 'vid')

#try to find the most likely point where the mitt is closed
closed_mitt<-function(data,len){  
  if(nrow(data%>%filter(class==0))>2){  
    loessMod <- loess(prob ~ time, data=data%>%filter(class==0), span=.8)
    smoothed <- data.frame(prob=predict(loessMod)) %>% cbind.data.frame(data.frame(time=(data%>%filter(class==0))$time))
    t<-smoothed[diff(diff(smoothed$prob)>=0)<0,]
    if(nrow(t)==0){
      times<-smoothed[which.max(smoothed$prob),'time']
    }else if(nrow(t)>1){
      times<-t[which.min(abs(t$time-mean(rbeta(10000,7.5,10))*len)),'time']
    }else{
      times<-t[,'time'] 
    }
  }else if(nrow(data%>%filter(class==0))%in%c(1,2)){
    times<-as.data.frame(data%>%filter(class==0))[which.max((data%>%filter(class==0))%>%pull(prob)),'time']
  }else{
    times<-NA
  }
  times
}

track_vid_times <-
  track_vid_info %>% ungroup() %>% mutate(closed_time = map2_dbl(data.x, len,  ~
                                                                   closed_mitt(.x, .y)))
hist(track_vid_times%>%mutate(closed_perc=closed_time/len)%>%pull(closed_perc))
hist(rbeta(10000,7.5,10))

#find the end frame with the fps time end time and find beginning frame by using release speed
#converting to feet per second and finding how long it takes to go 60.5 feet 
track_vid_bounds <-
  track_vid_times %>% mutate(
    release_speed = map_dbl(data.y,  ~ .x$release_speed),
    end_frame = ceiling(closed_time * fps),
    begin_frame = floor(fps * (closed_time - (60.5 / (
      release_speed * 5280 / 3600
    ))))
  )

#get only the relevant frames during the pitch. Which gives me almost 3600 vids.
track_vid_filt <- 
  track_vid_bounds %>% mutate(rel_frames = pmap(list(x = data.x, y = begin_frame, z = end_frame), 
                                                function(x,y,z) x %>% filter(frame >= y & frame <= z))) %>%
  select(-data.x) %>% filter(map_lgl(rel_frames, ~nrow(.x)>1))

track_motion<-function(frames, item){
  if(nrow(frames%>%filter(class==item))>1){
    frames<-frames%>%filter(class==item)%>%mutate(x_lag=lag(x,order_by=frame), y_lag=lag(y,order_by=frame))%>%na.omit()
    frames_diff<-frames%>%mutate(dist_l = pmax(x_lag-x,0), dist_r = pmax(x-x_lag,0), dist_d = pmax(y_lag-y,0), dist_u = pmax(y-y_lag,0), dist_x=abs(x-x_lag), dist_y=abs(y-y_lag), dist_t=sqrt(dist_x^2+dist_y^2))
    dist<-as.data.frame(t(frames_diff%>%select(starts_with('dist_'))%>%colSums()))
  }else{
    dist<-data.frame(dist_l=NA,dist_r=NA,dist_d=NA,dist_u=NA,dist_x=NA,dist_y=NA,dist_t=NA)
  }
  dist
}

track_vid_motion <- track_vid_filt %>% mutate(glove_motion=map(rel_frames,~track_motion(.x,2)),helmet_motion=map(rel_frames,~track_motion(.x,1)))

track_vid_unnest <- track_vid_motion%>%select(-c(rel_frames,release_speed))%>%unnest(data.y)%>%unnest(c(glove_motion,helmet_motion),names_repair = tidyr_legacy)
#write.csv(track_vid_unnest,'motion_track_pitch_info.csv',row.names = F)

#calculate in or out of the zone
t1<-c()
t2<-c()
for(i in seq(.05,1.6,.05)){
  track_vid_fin <- track_vid_unnest %>% mutate(in_zone = ifelse(plate_z >= sz_bot &
                                                 plate_z <= sz_top &
                                                 plate_x >= -i &
                                                 plate_x <= i,
                                                 1,0))
  t1<-c(t1,(track_vid_fin%>%group_by(type)%>%summarise(in_zone=mean(in_zone))%>%pull(in_zone))[1])
  t2<-c(t2,(track_vid_fin%>%group_by(type)%>%summarise(in_zone=mean(in_zone))%>%pull(in_zone))[2])
}

plot(t1,t2)
seq(.05,1.6,.05)[which.max(t2-t1)]

track_vid_fin <- track_vid_unnest %>% mutate(in_zone = ifelse(plate_z >= sz_bot &
                                                                plate_z <= sz_top &
                                                                plate_x >= -.95 &
                                                                plate_x <= .95,
                                                              1,0),
                                             count = paste0(balls,'-',strikes),
                                             same_hand = ifelse(p_throws==stand,1,0),
                                             hand_combos = paste0(p_throws, stand),
                                             strike = ifelse(type == 'S', 1, 0))
track_vid_fin[, c('in_zone', 'count', 'same_hand', 'hand_combos','strike')] <-
  lapply(track_vid_fin[, c('in_zone', 'count', 'same_hand', 'hand_combos', 'strike')], factor)

#mitt tracks in over 3200 while helmet is only in 970, overlap is 900
track_vid_m <- track_vid_fin %>% filter(!is.na(dist_l))%>%mutate_if(is.numeric, scale)
track_vid_h <- track_vid_fin %>% filter(!is.na(dist_l1))
track_vid_hm <- track_vid_fin %>% filter(!is.na(dist_l1)&!is.na(dist_l))%>%mutate_if(is.numeric, scale)
#really nothing doing on the mitt motion
summary(glm(strike ~ in_zone + dist_t, track_vid_m, family = 'binomial'))
#however we see some potential effect of the helmet movement albeit in a smaller sample 
summary(glm(strike ~ in_zone + dist_d1, track_vid_h, family = 'binomial'))

track_vid_h%>%group_by(type, in_zone)%>%summarise(n = n(), dist=mean(dist_d1))

ggplot(track_vid_h, aes(dist_d1, as.numeric(strike)-1)) + geom_jitter(height = 0.05) +
  geom_smooth(method = "glm", method.args = list(family = "binomial"))
