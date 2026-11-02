"""SAYL operations dashboard · Phase 2 / Person 5.

Start from the repository root: python -m streamlit run app.py
The dashboard reuses the original team's SaylModel and ACL bus; no fake agent data.
"""
from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import streamlit as st

from sayl.model import SaylModel, SimParams
from sayl.scenarios import SCENARIOS
from sayl.acl import Performative

st.set_page_config(page_title='SAYL | Flood Response Command', page_icon='🌊', layout='wide', initial_sidebar_state='expanded')
NAVY='#101A37'; PURPLE='#4F3BB4'; CYAN='#26B8D6'; GOLD='#F4B95C'; BG='#F5F7FC'
FREE='Free play · Custom incident'

st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;500;600;700&family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"],.stApp{font-family:Inter,'IBM Plex Sans Arabic',sans-serif}
.stApp{background:#F5F7FC;color:#101A37}
[data-testid="stSidebar"]{background:#fff;border-right:1px solid #e7eaf4}
[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{color:#101A37}
.block-container{padding-top:1.5rem;padding-bottom:2rem;max-width:1550px}
h1,h2,h3{letter-spacing:-.03em}
div[data-testid="stMetric"]{background:white;border:1px solid #e4e9f3;box-shadow:0 4px 18px #15254808;padding:15px 18px;border-radius:15px}
[data-testid="stMetricLabel"]{font-size:12px;color:#69758b}
[data-testid="stMetricValue"]{font-size:25px;font-weight:800;color:#172147}
.stButton button[kind="primary"]{background:#4F3BB4;border:0;color:white}
.stButton button{border-radius:11px;font-weight:600}
[data-testid="stDataFrame"]{border:1px solid #e6eafa;border-radius:10px}
.sayl-hero{background:linear-gradient(105deg,#17234c,#4b38a2 68%,#7253cc);padding:25px 28px;border-radius:21px;color:white;box-shadow:0 14px 32px #312f7c25;margin-bottom:20px}
.sayl-kicker{font-size:11px;letter-spacing:2px;color:#b5edf9;font-weight:800}
.sayl-title{font-size:31px;font-weight:800;margin:5px 0 8px;color:white;letter-spacing:-1px}
.sayl-sub{color:#dde4fb;font-size:13px;margin:0}
.section-title{font-size:17px;font-weight:800;color:#172147;margin:12px 0 9px}
.info-chip{display:inline-block;border:1px solid #b9c4ec80;background:#ffffff1a;border-radius:999px;color:#f0f4ff;padding:6px 12px;font-size:12px;font-weight:600;margin-top:9px}
.small-muted{color:#7b87a0;font-size:12px}
.safe-note{padding:9px 12px;background:#ecf9f7;color:#11615d;border-radius:10px;font-size:12px;border:1px solid #cdeee7}
footer{visibility:hidden}
</style>''', unsafe_allow_html=True)


def build_model(selection: str, custom: SimParams, use_defaults: bool) -> SaylModel:
    if selection == FREE:
        return SaylModel(custom)
    settings, events = SCENARIOS[selection]()
    return SaylModel(settings if use_defaults else custom, events)


def ensure_state():
    if 'selected_scenario' not in st.session_state:
        selection = next(iter(SCENARIOS))
        st.session_state.selected_scenario = selection
        st.session_state.model = build_model(selection, SimParams(), True)
        st.session_state.running = False
        st.session_state.ui_speed = .7

ensure_state()
model: SaylModel = st.session_state.model

with st.sidebar:
    st.markdown('## 🌊 SAYL | سَيْل')
    st.caption('Emergency coordination simulator · Phase 2')
    st.divider()
    st.markdown('### Mission control')
    options = list(SCENARIOS) + [FREE]
    current_index = options.index(st.session_state.selected_scenario) if st.session_state.selected_scenario in options else 0
    selection = st.selectbox('Training scenario', options, index=current_index,
                            help='Choose a scenario, then press Apply / Reset to load it.')
    defaults = st.checkbox('Use scenario preset settings', value=True,
                           help='Uncheck to apply your resource sliders on reset.')
    st.markdown('### Resource configuration')
    drones = st.slider('Drones',1,8,3)
    boats = st.slider('Rescue boats',1,6,2)
    shelter = st.slider('Shelter capacity per site',5,100,20,5)
    rain = st.slider('Rain intensity',0.0,1.0,float(model.env.rain_intensity),.05)
    failure = st.slider('Equipment failure chance / tick',0.0,.20,float(model.params.failure_prob),.01)
    rate = st.slider('Random incidents / tick chance',0.0,.5,float(model.params.distress_rate),.05)
    speed = st.slider('Simulation refresh (seconds)',.3,2.0,float(st.session_state.ui_speed),.1)
    st.session_state.ui_speed=speed
    st.caption('Rain, failure and random incident settings update the active simulation. Resource counts apply after Reset.')
    model.env.rain_intensity=rain
    model.params.distress_rate=rate
    model.params.failure_prob=failure
    for agent in model.drones+model.boats:
        agent.failure_prob=failure
    st.divider()
    scripted=st.session_state.selected_scenario != FREE
    complete=scripted and model.scenario_complete()
    c1,c2=st.columns(2)
    if c1.button('⏸ Pause' if st.session_state.running else '▶ Run',type='primary',use_container_width=True,disabled=complete):
        st.session_state.running=not st.session_state.running
        st.rerun()
    if c2.button('⏭ Step',use_container_width=True,disabled=complete or st.session_state.running):
        model.step()
        st.rerun()
    if st.button('↻ Apply scenario / Reset',use_container_width=True):
        params=SimParams(n_drones=drones,n_boats=boats,shelter_capacity=shelter,
                         rain_intensity=rain,failure_prob=failure,distress_rate=rate)
        st.session_state.model=build_model(selection,params,defaults)
        st.session_state.selected_scenario=selection
        st.session_state.running=False
        st.rerun()
    if selection != st.session_state.selected_scenario:
        st.info('Press Apply scenario / Reset to load your new selection.')
    if complete:
        st.success('Scenario completed. Use Reset to replay.')
    st.divider()
    st.caption('Educational simulation only · Not a live public emergency service')

st.markdown('''<div class="sayl-hero"><div class="sayl-kicker">SAYL / FLOOD OPERATIONS COMMAND</div>
<div class="sayl-title">Command center <span style="font-weight:400;color:#b9e6ff">| مركز القيادة</span></div>
<p class="sayl-sub">Real-time agent coordination, flood progression and emergency dispatch in one operational view.</p>
<span class="info-chip">● SIMULATION MODE</span> &nbsp;<span class="info-chip">Jeddah neighborhood · Synthetic map</span></div>''',unsafe_allow_html=True)

with st.expander('🚨 Submit an Arabic distress report',expanded=False):
    with st.form('report',clear_on_submit=True):
        a,b,c=st.columns([6,1,1])
        report=a.text_input('Arabic distress message',placeholder='الحقونا، الموية دخلت البيت وفيه أطفال وكبار سن')
        x=b.number_input('Grid X',min_value=0,max_value=model.env.width-1,value=model.env.width//2)
        y=c.number_input('Grid Y',min_value=0,max_value=model.env.height-1,value=model.env.height//2)
        submitted=st.form_submit_button('Send report to Distress Agent',type='primary')
        if submitted:
            if report.strip():
                model.submit_message(report.strip(),(int(x),int(y)))
                st.success('Report queued for agent evaluation. Advance a simulation tick to process it.')
            else:
                st.warning('Enter a message before sending.')

# map palette and reusable view functions
TERRAIN=LinearSegmentedColormap.from_list('sayl_terrain',['#e6edf0','#c8dbd7','#c3cfb5'])
WATER=LinearSegmentedColormap.from_list('sayl_water',['#b9e6f0','#4cadd8','#24639d'])
STATUS_COLOR={'NEW':'#E84B67','VERIFYING':'#EF9E45','VERIFIED':'#DC7259','RESCUING':'#6854CB'}


def render_map(m:SaylModel,roads_visible=True,labels_visible=True):
    env=m.env
    fig,ax=plt.subplots(figsize=(10.5,6.3),dpi=125)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#e8f0ef')
    ax.imshow(env.elevation,cmap=TERRAIN,origin='upper',interpolation='nearest')
    depths=env.depth_map()
    ax.imshow(depths,cmap=WATER,vmin=0,vmax=2,alpha=(depths>.02).astype(float)*.88,origin='upper')
    if roads_visible:
        road_y,road_x=np.nonzero(env.road_map())
        ax.scatter(road_x,road_y,c='#f6fbfa',marker='s',s=4,alpha=.72,zorder=2,linewidths=0)
        closed_y,closed_x=np.nonzero(env.closed_road_map())
        ax.scatter(closed_x,closed_y,c='#ef5664',marker='s',s=17,alpha=.96,zorder=3,linewidths=0)
    cx,cy=env.command_center_pos
    ax.scatter([cx],[cy],marker='*',s=330,c='#273267',edgecolors='white',linewidths=1.5,zorder=9)
    for place in m.shelters:
        color={'OPEN':'#31a88d','FULL':'#e3a347','CLOSED':'#777c91'}.get(place.state,'#7d8292')
        ax.scatter(place.pos[0],place.pos[1],marker='s',s=220,c=color,edgecolors='white',linewidths=1.5,zorder=8)
        if labels_visible:
            ax.annotate(f'{place.name[-1]}  {place.occupied}/{place.capacity}',place.pos,xytext=(3,-19),textcoords='offset points',fontsize=8,fontweight='bold',color=NAVY,zorder=10,bbox=dict(boxstyle='round,pad=.2',facecolor='white',alpha=.82,edgecolor='none'))
    for incident in env.open_incidents():
        ax.scatter(*incident.pos,s=85+incident.severity*28,c=STATUS_COLOR.get(incident.status,'#E84B67'),edgecolors='white',linewidths=1.5,zorder=11)
        ax.text(*incident.pos,str(incident.severity),ha='center',va='center',color='white',fontsize=8,fontweight='bold',zorder=12)
    for agent in m.drones:
        failed=agent.state=='FAILED'
        ax.scatter(*agent.pos,marker='X' if failed else '^',s=140,c='#E84B67' if failed else '#F4B95C',edgecolors=NAVY,zorder=13,linewidths=1)
    for agent in m.boats:
        ax.scatter(*agent.pos,marker='D',s=105,c='#5943ae',edgecolors='#e84b67' if agent.state in ('STUCK','FAILED') else 'white',linewidths=1.5,zorder=13)
    ax.set_xlim(-.7,env.width-.3);ax.set_ylim(env.height-.3,-.7)
    ax.set_xticks([]);ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)
    fig.tight_layout(pad=.5)
    return fig


def agent_rows(m:SaylModel):
    result=[]
    for a in m.agents:
        detail=a.status()
        extra=', '.join(f'{k}: {v}' for k,v in detail.items() if k not in ('agent','kind','type','state'))
        result.append({'Agent':str(detail.get('agent',a.name)),'Type':str(detail.get('type','')),
                       'State':str(detail.get('state','')),'Details':extra})
    return result


def message_rows(m:SaylModel):
    return [{'Tick':msg.tick,'Performative':msg.performative.value,'Sender':msg.sender,
             'Receiver':msg.receiver,'Ontology':msg.ontology.value,
             'Content':json.dumps(msg.content,ensure_ascii=False,default=str)} for msg in reversed(m.bus.log)]


@st.fragment(run_every=speed if st.session_state.running else None)
def dashboard():
    m:SaylModel=st.session_state.model
    if st.session_state.running:
        m.step()
        if st.session_state.selected_scenario!=FREE and m.scenario_complete():
            st.session_state.running=False
            st.rerun()
    stats=m.stats()
    if st.session_state.selected_scenario!=FREE and m.scenario_complete():
        st.success('✅ Scenario complete — all tracked incidents and rescue handoffs are finished.')
    cols=st.columns(6)
    metrics=[('SIM TICK',stats['tick']),('WATER DEPTH',f"{stats['water_level_m']:.2f} m"),
             ('OPEN CASES',stats['open']),('RESCUED PEOPLE',stats['people_rescued']),
             ('CLOSED ROADS',f"{m.env.road_closure_rate()*100:.0f}%"),('ACL MESSAGES',stats['messages'])]
    for col,(name,value) in zip(cols,metrics): col.metric(name,value)
    left,right=st.columns([1.42,1],gap='large')
    with left:
        st.markdown('<div class="section-title">🗺️ Live flood situation map</div>',unsafe_allow_html=True)
        r1,r2=st.columns(2)
        roads=r1.toggle('Show roads & closures',value=True,key='roads_toggle')
        labels=r2.toggle('Show shelter labels',value=True,key='labels_toggle')
        fig=render_map(m,roads,labels)
        st.pyplot(fig,use_container_width=True)
        plt.close(fig)
        st.caption('★ Command · ▲ Drone · ◆ Boat · ■ Shelter · ● Distress case / severity · Red cells = flooded road closures')
        with st.expander('Legend & flood behavior'):
            st.markdown('**Water:** deeper blue represents deeper flood water. **Roads:** pale marks are roads; red marks are closed by flooding (> 0.6 m). **Cases:** number indicates severity (1–5). Flood map is a simulated grid, not a real Jeddah map.')
    with right:
        st.markdown('<div class="section-title">📡 Live agent status</div>',unsafe_allow_html=True)
        df=pd.DataFrame(agent_rows(m))
        st.dataframe(df,hide_index=True,use_container_width=True,height=395)
        st.markdown('<div class="section-title">📊 Mission progress</div>',unsafe_allow_html=True)
        total=stats['incidents']
        resolved=max(0,total-stats['open'])
        st.progress(resolved/total if total else 0,text=f'Resolved cases: {resolved}/{total}')
        st.caption(f"Avg response: {stats['avg_response_ticks'] if stats['avg_response_ticks'] is not None else '—'} simulated minutes · Sheltered cases: {stats['sheltered_incidents']}")
    st.markdown('<div class="section-title">📨 Agent communication · ACL message log</div>',unsafe_allow_html=True)
    logs=message_rows(m)
    c1,c2,c3=st.columns([2,2,1])
    kinds=sorted({item['Performative'] for item in logs})
    selected=c1.multiselect('Performative filter',kinds,default=kinds,key='acl_types')
    keyword=c2.text_input('Search sender / receiver / content',placeholder='e.g. CommandCenter or Boat-1',key='acl_search')
    limit=c3.selectbox('Rows',[25,50,100,300],index=1)
    filtered=[r for r in logs if r['Performative'] in selected and keyword.casefold() in ' '.join(str(v) for v in r.values()).casefold()][:limit]
    if filtered: st.dataframe(pd.DataFrame(filtered),hide_index=True,use_container_width=True,height=285)
    else: st.info('No ACL messages match the current filters. Run or step the scenario to generate events.')
    st.download_button('⬇ Export all ACL messages (CSV)',data=pd.DataFrame(logs,columns=['Tick','Performative','Sender','Receiver','Ontology','Content']).to_csv(index=False).encode('utf-8-sig'),file_name=f'sayl_acl_tick_{m.tick}.csv',mime='text/csv',key='csv_download')
    with st.expander('Incident details and rescue assignments'):
        incidents=[]
        for inc in m.env.incidents.values():
            incidents.append({'ID':inc.incident_id,'Severity':inc.severity,'People':inc.people,'Status':inc.status,
                              'Location':str(inc.pos),'Drone':inc.drone or '—','Boat':inc.boat or '—',
                              'Shelter':inc.shelter or '—','Report':inc.text})
        if incidents: st.dataframe(pd.DataFrame(incidents),hide_index=True,use_container_width=True)
        else: st.caption('No incidents reported yet.')

dashboard()
st.markdown('<p class="small-muted" style="text-align:center;padding-top:26px">SAYL · Intelligent multi-agent flash-flood rescue simulator · Princess Nourah bint Abdulrahman University · Educational use only</p>',unsafe_allow_html=True)
