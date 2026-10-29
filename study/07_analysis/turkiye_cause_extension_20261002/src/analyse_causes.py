"""Cause-resolved customer burden, using the internally consistent 2025 export."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
d=pd.read_parquet(ROOT/'data/sedas_2025_numeric.parquet')
groups={
 'Wind / lightning':['Hava Muhalefeti (Rüzgar/Yıldırım)','Rüzgar'],
 'Ice loading':['Hava Muhalefeti (Buz Yükü)','Buz Yükü'],
 'Vegetation contact':['Ağaç Değmesi'],
 'Animals':['Kuş Çarpılması','Kuş Harici Hayvanların Etkisi','Canlı Çarpılması'],
 'Overload':['Aşırı Yüklenme-Şalter, Sigorta','Aşırı Yüklenme','Trafo Aşırı Yüklenme'],
 'Third-party / customer damage':['3. Şahısların Verdiği Arıza/Zarar','3. Şahısların Verdiği Zarar','Kullanıcıların Verdiği Arıza/Zarar','Kullanıcıların Verdiği Zarar','Hırsızlık Sonucu Arıza/Zarar','Hırsızlık Sonucu Zarar'],
 'Operations / safety':['Bakım Onarım Çalışması','Tesis Çalışması','Emniyet Amaçlı Kesinti','Manevra','Abone Bağlantısı için Kesinti','Fiziki İrtibat','Bakım','Tüzel Kişi Talebi ve Acil Durum','Manevra-Arızai Durum','Manevra-Planlı Çalışma Manevra','Enerjili Şebekeye Yakın Çalışma','Borçtan Açma - Kesme','Yatırım','Aydınlatma Çalışması','Yeni Bağlantı','Şebekeden Afiş vb Sökülmesi'],
 'Temporary / unspecified fault':['Geçici Arıza','Önlenemeyecek Arızalar'],
 'Fire, unspecified origin':['Yangın Sonucu Zarar'],
 'Equipment / network fault':['Abone Kablo Arızası','Malzemelerin Aşınması (Korozyon)','İletken Kopması','Jumper/Sıkıbağ/Klemens Arızası','OG Kablo Arızası','AG Kablo Arızası','OG Kablo Başlığı Arızası','Kablo Arızası','İzolatör Arızası','Direk Kırılması/Yıkılması','AG Pano Arızası (Pano Arızaları/Yangın)','Havai Hat Arızaları','Abone Bağlantı Hattı Arızası (Branşman)','Kesici Arızası','Trafo Arızası','Alpek (AER) Arızası','Ayırıcı Arızası (Seksiyoner)','Kablo Başlığı','Ayırıcı Arızası','AG Pano Şalter Arızası (TMŞ Arızası)','Parafudr Arızası','Anahtarlama Düzeni Arızası (Sigorta vb.)','Hatta Yabancı Cisim Teması','Sehim Bozukluğu','AG Dağıtım Panosu İç Arızası','Hücre Arızası','Sigortalı Yük Ayırıcı Arızası (DSYA,SYA)','Röle Arızası','Kaide Arızası/Box Tahribatı','Dağıtım-Güç Trafosu','OG Ayırıcı Arızası','Gerilim / Ölçü Hücresi / Devresi Arızası','Geçit İzolatörü Arızası','Trafo Buşing Arızası','Koruma Sistemi (Röle/Kumanda Panosu)','Konsol/Travers Kırılması/Arızası','Koruma Rölesi Arızası (Bucholz/Termik)','Trafo Yağ Kaçağı','AG Kablo Başlığı Arızası','Topraklama','Modüler Hücre Anahtar/Kontrol Düzenleri','Binanın Fiziki Durumundaki Eksiklikler','Ölçü Trafosu (Akım Gerilim)','Parafudr','DC Sistem']}
mapping={native:group for group,natives in groups.items() for native in natives}
assert d.KesintiNedeniAciklama4.notna().all()
unmapped=set(d.KesintiNedeniAciklama4)-set(mapping)
assert not unmapped,unmapped
cross=pd.DataFrame([dict(cause_native=c,cause_group=mapping[c]) for c in sorted(d.KesintiNedeniAciklama4.unique())])
cross['note']='Reported cause or failure mode; equipment and operational categories are not all anthropogenic triggers.'
cross.loc[cross.cause_group.eq('Wind / lightning'),'note']='Source combines wind and lightning; preserve this joint category.'
cross.loc[cross.cause_group.eq('Fire, unspecified origin'),'note']='Source does not distinguish wildfire from other fires.'
cross.to_csv(ROOT/'data/cause_crosswalk.csv',index=False)
d['cause_group']=d.KesintiNedeniAciklama4.map(mapping)
u=d[(d.KesintiniSinifi5BildirimeGore=='Bildirimsiz')&(d.KesintiniSinifi5SureyeGore5B=='Uzun')].copy()
published=pd.read_csv(ROOT/'data/published_annual_settlement_metrics.csv').query("year==2025 and notification=='Unannounced'")
long=[]
for group,tr in [('Urban','Kentsel'),('Suburban','Kentalti'),('Rural','Kirsal')]:
    for v,english in [('AG','LV'),('OG','MV')]:
        z=u[['Id','KesintininKodu1','Kademe2','Il3A','Ilce3B','KesintininBaslamaBitisTarihi','month','cause_group','KesintiNedeniAciklama4','KesintiSuresiSaat']].copy()
        z['settlement']=group;z['voltage']=english
        z['customer_interruptions']=u['EtkilenenKullaniciSayisi'+tr+v]
        z['reported_customer_hours']=u['ToplamEtkilenmeSuresi'+tr+v]
        z['product_customer_hours']=z.customer_interruptions*u.KesintiSuresiSaat
        z['formula_discrepancy']=(z.reported_customer_hours-z.product_customer_hours).abs()>(.011+.001*z.customer_interruptions)
        long.append(z)
l=pd.concat(long,ignore_index=True)
l.to_parquet(ROOT/'data/unannounced_customer_stages_2025.parquet',index=False)
agg=l.groupby(['settlement','voltage','cause_group'],as_index=False).agg(customer_interruptions=('customer_interruptions','sum'),customer_hours=('reported_customer_hours','sum'),product_customer_hours=('product_customer_hours','sum'),positive_customer_rows=('customer_interruptions',lambda s:int((s>0).sum())))
allv=agg.groupby(['settlement','cause_group'],as_index=False).sum(numeric_only=True).assign(voltage='All')
agg=pd.concat([agg,allv],ignore_index=True).merge(published[['settlement','voltage','users']],on=['settlement','voltage'],validate='many_to_one')
agg['frequency_contribution']=agg.customer_interruptions/agg.users
agg['annual_hours_contribution']=agg.customer_hours/agg.users
agg['reported_conditional_duration_h']=agg.customer_hours/agg.customer_interruptions
agg.to_csv(ROOT/'data/cause_settlement_metrics_2025.csv',index=False)
native=agg.groupby(['settlement','voltage'],as_index=False).agg(frequency_contribution=('frequency_contribution','sum'),annual_hours_contribution=('annual_hours_contribution','sum'))
rec=native.merge(published,on=['settlement','voltage'],validate='one_to_one')
rec['frequency_ratio_to_published']=rec.frequency_contribution/rec.frequency
rec['hours_ratio_to_published']=rec.annual_hours_contribution/rec.hours
rec.to_csv(ROOT/'data/annual_reconciliation_2025.csv',index=False)
part=[]
for voltage in ['All','LV','MV']:
    q=agg[agg.voltage.eq(voltage)].pivot(index='cause_group',columns='settlement',values='annual_hours_contribution')
    for c,r in q.iterrows():part.append(dict(voltage=voltage,cause_group=c,rural_urban_hour_gap=r.Rural-r.Urban,urban_hours=r.Urban,rural_hours=r.Rural))
    p=published[published.voltage.eq(voltage)].set_index('settlement')
    p_gap=p.loc['Rural','hours']-p.loc['Urban','hours']
    part.append(dict(voltage=voltage,cause_group='Reconciliation residual',rural_urban_hour_gap=p_gap-(q.Rural-q.Urban).sum(),urban_hours=p.loc['Urban','hours']-q.Urban.sum(),rural_hours=p.loc['Rural','hours']-q.Rural.sum()))
part=pd.DataFrame(part);part.to_csv(ROOT/'data/cause_contributions_to_rural_urban_gap_2025.csv',index=False)
reg=l.groupby(['Il3A','cause_group'],as_index=False).agg(customer_hours=('reported_customer_hours','sum'),customer_interruptions=('customer_interruptions','sum'))
reg['customer_hour_share']=reg.customer_hours/reg.groupby('Il3A').customer_hours.transform('sum')
reg.to_csv(ROOT/'data/province_cause_burdens_2025.csv',index=False)
counts=dict(raw_rows_2025=len(d),unannounced_sustained_rows=len(u),unannounced_sustained_incident_ids=u.KesintininKodu1.nunique(),
            customer_interruptions=float(l.customer_interruptions.sum()),reported_customer_hours=float(l.reported_customer_hours.sum()),
            formula_discrepancy_cells=int(l.formula_discrepancy.sum()),
            formula_discrepancy_abs_h=float((l.loc[l.formula_discrepancy,'reported_customer_hours']-l.loc[l.formula_discrepancy,'product_customer_hours']).abs().sum()))
(ROOT/'metadata/cause_analysis_summary.json').write_text(json.dumps(counts,indent=2))
print(json.dumps(counts,indent=2))
print(rec.query("voltage=='All'")[['settlement','frequency_ratio_to_published','hours_ratio_to_published']].to_string(index=False))
print(part.query("voltage=='All'").sort_values('rural_urban_hour_gap',ascending=False).to_string(index=False))
