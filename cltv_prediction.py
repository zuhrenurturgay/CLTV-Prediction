
############ BGNBD & GG ile CLTV Tahmini ve Sonuçların Uzak Sunucuya Gönderilmesi ############


import datetime as dt
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 300)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

###############################################################
# 1. İş Problemi (Business Problem)
###############################################################

# Bir e-ticaret sitesi müşteri aksiyonları için
# müşterilerinin CLTV değerlerine göre ileriye
# dönük bir projeksiyon yapılmasını istemektedir.
# Elinizdeki veriseti ile 1 aylık yada 6 aylık zaman
# periyotları içerisinde en çok gelir getirebilecek
# müşterileri tespit etmek mümkün müdür?

## VERİ SETİ HİKAYESİ ##

# Online Retail II isimli veri seti İngiltere merkezli online bir satış
# mağazasının 01/12/2009 - 09/12/2011 tarihleri arasındaki satışlarını içeriyor.
# Bu şirketin ürün kataloğunda hediyelik eşyalar yer alıyor. Promosyon ürünleri olarak da düşünülebilir.
# Çoğu müşterisinin toptancı olduğu bilgisi de mevcut.

## DEĞİŞKENLER ##

# InvoiceNo: Fatura numarası. Her işleme yani faturaya ait eşsiz numara. C ile başlıyorsa iptal edilen işlem.
# StockCode: Ürün kodu. Her bir ürün için eşsiz numara.
# Description: Ürün ismi
# Quantity: Ürün adedi. Faturalardaki ürünlerden kaçar tane satıldığını ifade etmektedir.
# InvoiceDate: Fatura tarihi ve zamanı.
# UnitPrice: Ürün fiyatı (Sterlin cinsinden)
# CustomerID: Eşsiz müşteri numarası
# Country: Ülke ismi. Müşterinin yaşadığı ülke.

df_ = pd.read_excel("datasets/online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")
df = df_.copy()

df.shape

#########################
# Verinin Veri Tabanından Okunması
#########################

creds = {
          'user': 'group_4',
         'passwd': 'miuul',
         'host': '34.79.73.237',
         'port': 3306,
         'db': 'group_4'
}

connstr = 'mysql+mysqlconnector://{user}:{passwd}@{host}:{port}/{db}'
conn = sqlalchemy.create_engine(connstr.format(**creds))

pd.read_sql_query("show databases;", conn)
pd.read_sql_query("show tables", conn)
pd.read_sql_query("select * from online_retail_2010_2011 limit 10", conn)
retail_mysql_df=pd.read_sql_query("select * from online_retail_2010_2011 limit 10", conn)

retail_mysql_df.shape
retail_mysql_df.head()

########
#GÖREV-1: 6 aylık CLTV Prediction
########

# Veri Ön İşleme

df.describe().T
df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

df["TotalPrice"] = df["Quantity"] * df["Price"]

today_date = dt.datetime(2011, 12, 11)

#########################
# Lifetime Veri Yapısının Hazırlanması
#########################

cltv_df = df.groupby("Customer ID").agg({"InvoiceDate": [lambda date: (date.max() - date.min()).days,
                                                         lambda date: (today_date - date.min()).days],
                                         "Invoice": lambda num: num.nunique(),
                                         "TotalPrice":lambda TotalPrice: TotalPrice.sum()})

cltv_df.columns=cltv_df.columns.droplevel(0)
cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']
cltv_df.head(10)

cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]

cltv_df = cltv_df[(cltv_df["frequency"]>1)]

cltv_df["recency"] = cltv_df["recency"] / 7

cltv_df["T"] = cltv_df["T"] / 7

##############################################################
# 2. BG-NBD Modelinin Kurulması
##############################################################

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df["frequency"],
        cltv_df["recency"],
        cltv_df["T"])

##############################################################
# 3. GAMMA-GAMMA Modelinin Kurulması
##############################################################

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df["frequency"], cltv_df["monetary"])


ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                        cltv_df["monetary"]).head(10)

ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                        cltv_df["monetary"]).sort_values(ascending=False).head(10)

cltv_df["expected_average_profit"]= ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                                                            cltv_df["monetary"])

##############################################################
# 4. BG-NBD ve GG modeli ile CLTV'nin hesaplanması.
### 6 AYLIK CLTV PREDICTION
##############################################################

cltv= ggf.customer_lifetime_value(bgf,
                                  cltv_df["frequency"],
                                  cltv_df["recency"],
                                  cltv_df["T"],
                                  cltv_df["monetary"],
                                  time=6,
                                  freq="W",
                                  discount_rate=0.01)

cltv.head()

cltv=cltv.reset_index()

cltv.sort_values(by="clv", ascending=False).head(25)

cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")

cltv_final.sort_values(by="clv", ascending=False).head(10)


scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(cltv_final[["clv"]])
cltv_final["scaled_clv"] = scaler.transform(cltv_final[["clv"]])

########
#GÖREV-2:Farklı zaman periyotlarından oluşan CLTV analizi
########

cltv_1_month= ggf.customer_lifetime_value(bgf,
                                  cltv_df["frequency"],
                                  cltv_df["recency"],
                                  cltv_df["T"],
                                  cltv_df["monetary"],
                                  time=1,
                                  freq="W",
                                  discount_rate=0.01)

cltv_1_month.sort_values(ascending=False).head(10)

cltv_12_months= ggf.customer_lifetime_value(bgf,
                                  cltv_df["frequency"],
                                  cltv_df["recency"],
                                  cltv_df["T"],
                                  cltv_df["monetary"],
                                  time=12,
                                  freq="W",
                                  discount_rate=0.01)

cltv_12_months.sort_values(ascending=False).head(10)

## 12 aylık cltvde gözle görülür bir şekilde artış yaşandığı görülmektedir. 1 aylık ile 12 aylık cltv arasındaki bu
#farkın nedeni 12 ay gibi uzun bir sürede şirketin yapmış olduğu kampanyalarla müşteriyi kendine bağlaması olduğunu düşünüyorum.


########
#GÖREV-3:Segmentasyon ve Aksiyon Önerileri
########

cltv_final["segment"]=pd.qcut(cltv_final["scaled_clv"], 4,labels=["D","C","B","A"])

cltv_final.head()
cltv_final.sort_values(by="scaled_clv", ascending=False).head(25)

cltv_final.groupby("segment").agg({"mean","sum","count"})

## D Segmenti
# Bu segmentteki müşteriler şirket için kaybedilmek üzere olan müşterilerdir. Bu müşterilerin frequency ortalaması
# ne kadar yükselme eğiliminde olsa da clv si düşüktür, müşterileri alışverişe yönlendirmek için kampanyalar düzenlenebilir
# -kargo bedava,indirim kuponu vs- bu kampanyalarla müşterilerin monetary_value değerini arttırılması hedeflenmelidir.

## A Segmenti
# Bu segmentteki müşteriler şirketin en kıymetli müşterilerdir. Fakat A segmentinin gittikçe frequencysi düşmektedir.
# Satın alma sayılarını arttırmak için bu segmentteki müşterilere ayrıcalıklar tanınmalıdır. Belli monetarye ulaşan müşterileri
# içine alan bir topluluk oluşturup bunlara özel indirimler yapılmasını uygun buluyorum.


########
#GÖREV-4:Veri tabanına kayıt gönderme
########

cltv_final.head()

cltv_final["Customer ID"] = cltv_final["Customer ID"].astype(int)

cltv_final.to_sql(name='zuhrenur_turgay', con=conn, if_exists='replace', index=False)

pd.read_sql_query("select * from zuhrenur_turgay limit 10", conn)