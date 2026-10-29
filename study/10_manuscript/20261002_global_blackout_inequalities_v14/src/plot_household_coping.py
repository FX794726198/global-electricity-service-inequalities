"""Supplementary resource profiles; existing Python publication-figure workflow."""
from figure_style import *


def main():
    d=pd.read_csv(P/'source_data/household_coping_quintiles.csv')
    f=page(138)
    colours={'appliance_no_backup':BLUE,'lighting_no_backup':GOLD,'no_repair_channel':'#79688D'}
    legend(f,[Line2D([],[],color=BLUE,marker='o',ms=4,lw=1.3),Line2D([],[],color=GOLD,marker='s',ms=4,lw=1.3)],
           ['Appliance electricity','Lighting'],110,7,2,fontsize=10)
    for j,(country,label) in enumerate([('cambodia','Cambodia'),('ethiopia','Ethiopia'),('kenya','Kenya')]):
        x=20+62*j
        top=ax(f,x,22,56,43);bot=ax(f,x,80,56,43)
        title(f,top,chr(97+j),label)
        title(f,bot,chr(100+j),'')
        for a,metrics in [(top,['appliance_no_backup','lighting_no_backup']),(bot,['no_repair_channel'])]:
            for m in metrics:
                z=d[(d.country==country)&(d.metric==m)].sort_values('quintile')
                a.fill_between(z.quintile,100*z.lo,100*z.hi,color=colours[m],alpha=.14,lw=0,zorder=1)
                a.plot(z.quintile,100*z.estimate,color=colours[m],marker='s' if m=='lighting_no_backup' else 'o',
                       markersize=4,lw=1.5,zorder=3)
            a.set(xlim=(.8,5.2),ylim=(-3,105),xticks=range(1,6),yticks=[0,25,50,75,100])
            a.set_xticklabels(['Q1','Q2','Q3','Q4','Q5'])
            clean(a,grid=True)
            if j:a.set_yticklabels([])
        if j==0:
            top.set_ylabel('No backup source (%)')
            bot.set_ylabel('No repair channel (%)')
    text(f,110,135,'Within-country living-standard quintile',ha='center',fontsize=10.5)
    save(f,'SuppFig10_household_coping_and_response')


if __name__=='__main__':main()
