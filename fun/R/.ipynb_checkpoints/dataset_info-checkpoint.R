dataset_info <-function(GSE_ID){
    Def<-list()
    if ('GSE19439' %in% GSE_ID){
      Def$stage<-'illness:ch1' #Define which column in pheno matrix indicating disease status
      Def$labels <- list('HC'=c('Control (BCG+)','Control (BCG-)'),'LTBI'='Latent','ATB'='PTB') 
    }else if ('GSE19442' %in% GSE_ID){
      Def$stage<-'illness:ch1'
      Def$labels<-list('LTBI'='LATENT TB','ATB'='PTB') 
    } else if ('GSE19444' %in% GSE_ID){
      Def$stage<-'illness:ch1'
      Def$labels<-list('HC'='Control (BCG+)','LTBI'='Latent','ATB'='PTB') 
    } else if ('GSE22098' %in% GSE_ID){  
      Def$stage<-'illness:ch1'
      Def$labels<-list('HC'='NA','OD'=c('Still','ASLE','PSLE','Staph','Strep')) 
    } else if ('GSE28623' %in% GSE_ID){
      Def$stage<-'source_name_ch1'
      Def$labels<-list('HC'='Peripheral blood NID','LTBI'='Peripheral blood LTBI','ATB'='Peripheral blood TB') 
    } else if ('GSE29536' %in% GSE_ID){  
      Def$stage<-'disease_status:ch1'
      Def$labels<-list('HC'=c('Healthy','Control BCG+','Control BCG-'),
                       'OD'=c('SOJIA','AHI Enrollment','SLE','other infection','Other infection','melioidosis'),'ATB'='PTB') 
    } else if ('GSE34608' %in% GSE_ID){ 
      Def$stage<-'disease state:ch1'
      Def$labels<-list('HC'='control','OD'='sarcoidosis','ATB'='tuberculosis') 
    } else if ('GSE37250' %in% GSE_ID){  
      Def$stage<-'disease state:ch1'
      Def$labels<-list('OD'='other disease','LTBI'='latent TB infection','ATB'='active tuberculosis') 
    } else if ('GSE39939' %in% GSE_ID){  
      Def$stage<-'illness:ch1'
      Def$labels<-list('OD'=c('other disease','other disease (IGRA +)'),'LTBI'='latent TB infection',
                       'ATB'=c('active tuberculosis (culture confirmed)','active tuberculosis (culture negative)')) 
    } else if ('GSE39940' %in% GSE_ID){  
      Def$stage<-'disease status:ch1'
      Def$labels<-list('OD'='other disease','LTBI'='latent TB infection','ATB'='active tuberculosis') 
    } else if ('GSE41055' %in% GSE_ID){  
      Def$stage<-'source_name_ch1'
      Def$labels<-list('HC'='whole blood, healthy control','LTBI'='whole blood, latent TB infection','ATB'='whole blood, active TB infection') 
    } else if ('GSE42825' %in% GSE_ID){  
      Def$stage<-'disease state:ch1'
      Def$labels<-list('HC'='Control','OD'=c('Active sarcoidosis','Non-active sarcoidosis'),'ATB'='TB') 
    } else if ('GSE42826' %in% GSE_ID){  
      Def$stage<-'disease state:ch1'
      Def$labels<-list('HC'='Control','OD'=c('Non-active sarcoidosis','Active Sarcoid','lung cancer','Pneumonia'),'ATB'='TB') 
    } else if ('GSE42830' %in% GSE_ID){  
      Def$stage<-'disease state:ch1'
      Def$labels<-list('HC'='Control','OD'=c('lung cancer','Active Sarcoid','Non-active sarcoidosis','Baseline'),'ATB'='TB') 
    } else if ('GSE50834' %in% GSE_ID){  
      Def$stage<-'disease state:ch1'
      Def$labels<-list('HC'='HIV only','ATB'='HIV/TB') 
    } else if ('GSE56153' %in% GSE_ID){  
      Def$stage<-'condition:ch1'
      Def$labels<-list('HC'='Control','ATB'='Active','Tret'=c('Treatment','Recover')) 
    } else if ('GSE54992' %in% GSE_ID){  
      Def$stage<-'disease state:ch1'
      Def$labels<-list('HC'='healthy donor',
                       'LTBI'='latent tuberculosis infection','ATB'='tuberculosis',
                       'Tret'=c('TB patient after anti-tuberculosis treatment for 3 months','TB patient after anti-tuberculosis treatment for 6 months')) 
    } else if ('GSE62147' %in% GSE_ID){  
      Def$stage<-'source_name_ch1'
      Def$labels<-list('Tret'='Peripheral blood Mtb_post','ATB'='Peripheral blood Mtb_recruit') 
    } else if ('GSE62525' %in% GSE_ID){
      Def$stage<-'disease state:ch1'
      Def$labels<-list('HC'='healthy control','LTBI'='latent TB infection','ATB'='active TB') 
    } else if ('GSE69581' %in% GSE_ID){  
      Def$stage<-'tb status:ch1'
      Def$labels<-list('LTBI'='Latent','ATB'='Active') 
    } else if ('GSE73408' %in% GSE_ID){  
      Def$stage<-'clinical group:ch1'
      Def$labels<-list('OD'='PNA','LTBI'='LTBI','ATB'='TB') 
    } else if ('GSE83456' %in% GSE_ID){  
      Def$stage<-'disease state:ch1'
      Def$labels<-list('HC'='HC','OD'='Sarcoid','ATB'=c('PTB','EPTB')) 
    } else if ('GSE40553' %in% GSE_ID){  
      Def$stage<-'status:ch1'
      Def$labels<-list('Tret'=c('active TB 2 weeks post treatment','active TB 2 months post treatment',
                               'active TB 6 months post treatment','active TB 12 months post treatment'),
                               'ATB'='active TB pre-treatment','LTBI'='untreated latent TB') 
    } else if ('GSE31348_GSE36238' %in% GSE_ID){  
      Def$stage<-'time point:ch1'
      Def$labels<-list('Tret'=c('Week 1','Week 2','Week 4','Week 26'),'ATB'='Week 0') 
    } else if ('GSE84076' %in% GSE_ID){  
      Def$stage<-'clinical information:ch1'
      Def$labels<- list('HC'=c('Control - BCG - Unvaccinated','Control - BCG - vaccinated'),
                        'LTBI'=c('Latent Tuberculosis - BCG - Unvaccinated',
                                 'Latent Tuberculosis - BCG -vaccinated'),
                        'ATB'='Active Tuberculosis') 
    } else if ('GSE101705' %in% GSE_ID){  
      Def$stage<-'condition:ch1'
      Def$labels<-list('LTBI'='latent TB infection','ATB'='TB') 
    } else if ('GSE107993' %in% GSE_ID){  
      Def$stage<-'group:ch1'
      Def$labels<-list('HC'='Control','LTBI'='LTBI') 
    } else if ('GSE107994' %in% GSE_ID){  
      Def$stage<-'group:ch1'
      Def$labels<-list('HC'='Control','LTBI'='LTBI','ATB'='Active_TB') 
    }
    return(Def)
}