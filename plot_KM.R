library("survminer")
library('readr')
require("survival")

args = commandArgs(trailingOnly=TRUE)
data_path=args[1]#'/Users/manage/Desktop/final_oneFlorida/cluster_stats/MCI2AD_time_cluster_age_sex_gender_df.csv'
save_path=args[2]#/Users/manage/Desktop/final_oneFlorida/KM/

# load AD subphenotype data

ADData<-read.csv(paste0(data_path,'MCI2AD_time_cluster_df.csv'),header = T)

# AD 
fit<- survfit(Surv(MCI2AD_time, AD) ~ Cluster, data = ADData) # MCI2AD_time, AD, Cluster are columns in input data
# add method to grid.draw
grid.draw.ggsurvplot <- function(x){
  survminer:::print.ggsurvplot(x, newpage = FALSE)
}
xmax=max(ADData$MCI2AD_time)
# Drawing survival curves
surp<-ggsurvplot(fit,
                 data = ADData,
                 legend.title = "Subphenotypes",
                 legend.labs = c("Subphenotype 1","Subphenotype 2","Subphenotype 3","Subphenotype 4","Subphenotype 5"),
                 
                 palette = c("#E7B800","#2E9FDF","#E64B35FF","#3C5488FF","#00A087FF"),
                 censor = FALSE, # TRUE, FALSE
                 
                 conf.int = TRUE, # Add confidence interval
                 conf.int.alpha = 0.1,
                 pval = TRUE, # Add p-value
                 surv.plot.height = 0.2,
                 
                 ## change font
                 font.title    = c(38), #30 for oneFlorida,  38 for INSIGHT,
                 font.x        = c(38), #30 for oneFlorida,  38 for INSIGHT
                 font.y        = c(38), #30 for oneFlorida,  38 for INSIGHT
                 font.tickslab = 16, # 16
                 font.legend = list(size = 20),
                 
                 ##
                 risk.table = TRUE,        # Add risk table
                 risk.table.col = "strata",# Risk table color by groups
                 risk.table.height = 0.25, # Useful to change when you have multiple groups
                 fontsize = 7,  # 8.5
                 #risk.table.fontsize = 38,
                 
                 xlab = 'Time (Day)',
                 #xscale = 30,
                 #xscale = "d_m",
                 xlim = c(0, xmax), #3200 for oneFlorida #7100 for INSIGHT #6005 for UCSF
                 ylim = c(0, 1),
                 break.time.by = 90,
                 ggtheme = theme_bw(),
                 
                 
)

#change KM curve font
surp$plot = surp$plot + theme(axis.text.x = element_text(angle=60,vjust = 0.5))
surp

# save the final figure
# set figure name
final_figure_save_name = paste0(save_path,"survplot.png")
ggsave(filename = final_figure_save_name, surp,  width = 40, height = 15, dpi = 200, limitsize = FALSE) # width = 60, height = 30,dpi = 300,




