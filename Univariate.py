import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
class Preprocess():
    def QuanQual(df):
        quan=[]
        qual=[]
        for i in df:
            if df[i].dtype=="O":
                qual.append(i)
            else:
                quan.append(i)
        return quan,qual
    def Univariate_Table(quan,df,describe):
        
        Quan_df=pd.DataFrame(columns=quan,index=["Mean","Median","Mode",
                                                 "Q1-25%","Q2-50%","Q3-75%","99%","Q4-100%",
                                                 "IQR","1.5*IQR","Lower_Fence",
                                                 "Higher_Fence","minimum","maximum",
                                                 "Skew","Kurtosis","Var","Std"])
        for i in quan:
            Quan_df[i]["Mean"]=df[i].mean()
            Quan_df[i]["Median"]=df[i].median()
            Quan_df[i]["Mode"]=df[i].mode()[0]
            Quan_df[i]["Q1-25%"]=df.describe()[i]["25%"]
            Quan_df[i]["Q2-50%"]=df.describe()[i]["50%"]
            Quan_df[i]["Q3-75%"]=df.describe()[i]["75%"]
            Quan_df[i]["99%"]=np.percentile(df[i],99)
            Quan_df[i]["Q4-100%"]=df.describe()[i]["max"]
            Quan_df[i]["IQR"]=Quan_df[i]["Q3-75%"]-Quan_df[i]["Q1-25%"]
            Quan_df[i]["1.5*IQR"]=1.5*Quan_df[i]["IQR"]
            Quan_df[i]["Lower_Fence"]=Quan_df[i]["Q1-25%"]-Quan_df[i]["1.5*IQR"]
            Quan_df[i]["Higher_Fence"]=Quan_df[i]["Q3-75%"]+Quan_df[i]["1.5*IQR"]
            Quan_df[i]["minimum"]=df[i].min()
            Quan_df[i]["maximum"]=df[i].max()
            Quan_df[i]["Skew"]=df[i].skew()
            Quan_df[i]["Kurtosis"]=df[i].kurtosis()
            Quan_df[i]["Var"]=df[i].var()
            Quan_df[i]["Std"]=df[i].std()
            
        return Quan_df
    def Finding_Outlier(Quan_df):
        lower=[]
        higher=[]
        for i in Quan_df:
            if Quan_df[i]["Lower_Fence"]>Quan_df[i]["minimum"]:
                lower.append(i)
            if Quan_df[i]["Higher_Fence"]<Quan_df[i]["maximum"]:
                higher.append(i)
        return lower,higher
    def Replace_Outlier(lower,df,Quan_df,higher):
        for i in lower:
            df[i][df[i]<Quan_df[i]["Lower_Fence"]]=Quan_df[i]["Lower_Fence"]
        for i in higher:
            df[i][df[i]>Quan_df[i]["Higher_Fence"]]=Quan_df[i]["Higher_Fence"]
        return df
    def Frequancy_Table(df,column_name):
        Frequancy_Table=pd.DataFrame(columns=["Unique_Value","Frequancy","Relative_Frequancy",
                                              "Cumilative_Frequancy"])
        Frequancy_Table["Unique_Value"]=df[column_name].value_counts().index
        Frequancy_Table["Frequancy"]=df[column_name].value_counts().values
        Frequancy_Table["Relative_Frequancy"]=(Frequancy_Table["Frequancy"])/len(Frequancy_Table["Frequancy"])
        Frequancy_Table["Cumilative_Frequancy"]=Frequancy_Table["Relative_Frequancy"].cumsum()

        return Frequancy_Table
    def PDF_Visual(df,column_name,startrange,endrange):
        sns.distplot(df[column_name],kde_kws={"color":"blue"},color="green")
        plt.axvline(startrange,color="red")
        plt.axvline(endrange,color="red")
        mean=df[column_name].mean()
        std=df[column_name].std()
        print(f"Mean_Value : {mean},Std_Value : {std}")
        dist=norm(mean,std)
        values=[i for i in range(startrange,endrange)]
        probability=[dist.pdf(i) for i in values]
        sum_prob=sum(probability)
        print(f"Percentage of Probability : {sum_prob}")
        return sum_prob
    def Standard_Normal_Distribution(df,column_name):
        mean=df[column_name].mean()
        std=df[column_name].std()
        z_score=[(x-mean)/std for x in df[column_name]]
        sns.distplot(z_score,color="red",kde_kws={"color":"blue"})