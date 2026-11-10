"""SAYL operations dashboard with live simulation data and ACL monitoring.

Start from the repository root with ``python -m streamlit run app.py``.
"""
from __future__ import annotations

import json

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
from sayl.ui_theme import NAVY, TEAL, BLUE, AMBER

st.set_page_config(page_title='SAYL | Flood Response Command', page_icon='🌊',
                   layout='wide', initial_sidebar_state='collapsed')
FREE = 'Free play · Custom incident'


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

# Streamlit-native right-hand control panel: no custom HTML/CSS or JavaScript.
# The large content area stays on the left while configuration starts on the right.
main_column, controls_column = st.columns([3.05, 1.15], gap='large')

with controls_column:
    with st.container(border=True):
        st.markdown('#### 🎛️ Mission control | التحكم')
        st.caption('Choose a scenario, configure it, then apply your settings.')
        options = list(SCENARIOS) + [FREE]
        current_index = options.index(st.session_state.selected_scenario) if st.session_state.selected_scenario in options else 0
        selection = st.selectbox('Training scenario', options, index=current_index,
                                 help='Select a scenario and click Apply / Reset to load it.')
        defaults = st.checkbox('Use scenario preset settings', value=True,
                               help='Uncheck to use your resource settings when resetting a scenario.')
        if selection != st.session_state.selected_scenario:
            st.info('Apply / Reset to load your selected scenario.')

    with st.container(border=True):
        st.markdown('#### 🚤 Rescue resources | الموارد')
        st.caption('These settings take effect when you select Apply / Reset.')
        drones = st.slider('Drones', 1, 8, 3)
        boats = st.slider('Rescue boats', 1, 6, 2)
        shelter = st.slider('Shelter capacity / site', 5, 100, 20, 5)

    with st.expander('🌧️ Weather & advanced controls', expanded=False):
        rain = st.slider('Rain intensity', 0.0, 1.0, float(model.env.rain_intensity), .05)
        failure = st.slider('Equipment failure chance / tick', 0.0, .20, float(model.params.failure_prob), .01)
        rate = st.slider('Random incidents / tick', 0.0, .5, float(model.params.distress_rate), .05)
        speed = st.slider('Refresh (seconds)', .3, 2.0, float(st.session_state.ui_speed), .1)
        st.session_state.ui_speed = speed
        st.caption('Rainfall, equipment failure and incident rates update the current simulation immediately.')

    model.env.rain_intensity = rain
    model.params.distress_rate = rate
    model.params.failure_prob = failure
    for agent in model.drones + model.boats:
        agent.failure_prob = failure

    with st.container(border=True):
        st.markdown('#### ▶️ Simulation controls')
        scripted = st.session_state.selected_scenario != FREE
        complete = scripted and model.scenario_complete()
        run_col, step_col = st.columns(2, gap='small')
        if run_col.button('⏸ Pause' if st.session_state.running else '▶ Run', type='primary',
                          width='stretch', disabled=complete):
            st.session_state.running = not st.session_state.running
            st.rerun()
        if step_col.button('⏭ Step', width='stretch', disabled=complete or st.session_state.running):
            model.step()
            st.rerun()
        if st.button('↻ Apply scenario / Reset', width='stretch'):
            params = SimParams(n_drones=drones, n_boats=boats, shelter_capacity=shelter,
                               rain_intensity=rain, failure_prob=failure, distress_rate=rate)
            st.session_state.model = build_model(selection, params, defaults)
            st.session_state.selected_scenario = selection
            st.session_state.running = False
            st.rerun()
        if complete:
            st.success('Scenario complete. Reset to replay.')
        st.caption('Training simulation only — not a real emergency dispatch service.')

with main_column:
    with st.container(border=True):
        title_col, info_col = st.columns([3.3, 1], vertical_alignment='center', gap='small')
        with title_col:
            st.caption('CAI461  /  MULTI-AGENT FLOOD RESCUE')
            st.markdown('## 🌊 SAYL | سَيْل')
            st.markdown('**Flood Rescue Command Center  •  مركز قيادة الإنقاذ**')
            st.caption('Flood monitoring · coordinated rescue · real-time agent communication')
        with info_col:
            st.badge('SIMULATION', color='blue')
            st.badge('PYTHON / STREAMLIT', color='green')
    with st.expander('🚨 New Arabic distress report | بلاغ استغاثة', expanded=False):
        with st.form('report', clear_on_submit=True):
            st.caption('Send a new Arabic emergency message to the Distress Agent.')
            report = st.text_area('Arabic distress message', height=85,
                                  placeholder='الحقونا، الموية دخلت البيت وفيه أطفال وكبار سن')
            x_col, y_col, send_col = st.columns([1, 1, 1.5], vertical_alignment='bottom')
            x = x_col.number_input('Grid X', min_value=0, max_value=model.env.width-1, value=model.env.width//2)
            y = y_col.number_input('Grid Y', min_value=0, max_value=model.env.height-1, value=model.env.height//2)
            with send_col:
                submitted = st.form_submit_button('📨 Send report', type='primary', width='stretch')
            if submitted:
                if report.strip():
                    model.submit_message(report.strip(), (int(x), int(y)))
                    st.success('Report queued. Run or step the simulation to process it.')
                else:
                    st.warning('Please enter a distress message first.')

# map palette and reusable view functions
TERRAIN=LinearSegmentedColormap.from_list('sayl_terrain',['#e8f0ed','#cbdcdb','#b6d0c3'])
WATER=LinearSegmentedColormap.from_list('sayl_water',['#B5E3E8','#4EA4C4','#1D6A9A'])
STATUS_COLOR={'NEW':'#C85161','VERIFYING':'#D18C35','VERIFIED':'#D16E43','RESCUING':'#2277A5'}


def render_map(m:SaylModel,roads_visible=True,labels_visible=True):
    env=m.env
    fig,ax=plt.subplots(figsize=(10.5,6.3),dpi=125)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#eaf2f1')
    ax.imshow(env.elevation,cmap=TERRAIN,origin='upper',interpolation='nearest')
    depths=env.depth_map()
    ax.imshow(depths,cmap=WATER,vmin=0,vmax=2,alpha=(depths>.02).astype(float)*.88,origin='upper')
    if roads_visible:
        road_y,road_x=np.nonzero(env.road_map())
        ax.scatter(road_x,road_y,c='#f6fbfa',marker='s',s=4,alpha=.72,zorder=2,linewidths=0)
        closed_y,closed_x=np.nonzero(env.closed_road_map())
        ax.scatter(closed_x,closed_y,c='#C85161',marker='s',s=17,alpha=.96,zorder=3,linewidths=0)
    cx,cy=env.command_center_pos
    ax.scatter([cx],[cy],marker='*',s=330,c=NAVY,edgecolors='white',linewidths=1.5,zorder=9)
    for place in m.shelters:
        color={'OPEN':TEAL,'FULL':AMBER,'CLOSED':'#667C86'}.get(place.state,'#7d8292')
        ax.scatter(place.pos[0],place.pos[1],marker='s',s=220,c=color,edgecolors='white',linewidths=1.5,zorder=8)
        if labels_visible:
            ax.annotate(f'{place.name[-1]}  {place.occupied}/{place.capacity}',place.pos,xytext=(3,-19),textcoords='offset points',fontsize=8,fontweight='bold',color=NAVY,zorder=10,bbox=dict(boxstyle='round,pad=.2',facecolor='white',alpha=.82,edgecolor='none'))
    for incident in env.open_incidents():
        ax.scatter(*incident.pos,s=85+incident.severity*28,c=STATUS_COLOR.get(incident.status,'#E84B67'),edgecolors='white',linewidths=1.5,zorder=11)
        ax.text(*incident.pos,str(incident.severity),ha='center',va='center',color='white',fontsize=8,fontweight='bold',zorder=12)
    for agent in m.drones:
        failed=agent.state=='FAILED'
        ax.scatter(*agent.pos,marker='X' if failed else '^',s=140,c='#C85161' if failed else AMBER,edgecolors=NAVY,zorder=13,linewidths=1)
    for agent in m.boats:
        ax.scatter(*agent.pos,marker='D',s=105,c=BLUE,edgecolors='#C85161' if agent.state in ('STUCK','FAILED') else 'white',linewidths=1.5,zorder=13)
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
    metrics=[('SIM TICK',stats['tick']),('WATER DEPTH',f"{stats['water_level_m']:.2f} m"),
             ('OPEN CASES',stats['open']),('ACL MESSAGES',stats['messages']),
             ('PEOPLE RESCUED',stats['people_rescued']),('SHELTERED',stats['people_sheltered']),
             ('AT STAGING',stats['people_staged']),('CLOSED ROADS',f"{m.env.road_closure_rate()*100:.0f}%")]
    for start in (0, 4):
        cols=st.columns(4)
        for col,(name,value) in zip(cols,metrics[start:start+4]):
            col.metric(name,value,border=True)
    left,right=st.columns([1.42,1],gap='large')
    with left:
        st.markdown('#### 🗺️ Live flood situation map')
        r1,r2=st.columns(2)
        roads=r1.toggle('Show roads & closures',value=True,key='roads_toggle')
        labels=r2.toggle('Show shelter labels',value=True,key='labels_toggle')
        fig=render_map(m,roads,labels)
        st.pyplot(fig,width='stretch')
        plt.close(fig)
        st.caption('★ Command  ·  ▲ Drone  ·  ◆ Boat  ·  ■ Shelter  ·  ● Distress case  ·  Red cells: closed roads')
        with st.expander('Legend & flood behavior'):
            st.markdown('**Water:** deeper blue represents deeper flood water. **Roads:** pale marks are roads; red marks are closed by flooding (> 0.6 m). **Cases:** number indicates severity (1–5). Flood map is a simulated grid, not a real Jeddah map.')
    with right:
        st.markdown('#### 📡 Live agent status')
        df=pd.DataFrame(agent_rows(m))
        st.dataframe(df,hide_index=True,width='stretch',height=395)
        st.markdown('#### 📊 Mission progress')
        total=stats['incidents']
        resolved=max(0,total-stats['open'])
        st.progress(resolved/total if total else 0,text=f'Resolved cases: {resolved}/{total}')
        st.caption(f"Avg response: {stats['avg_response_ticks'] if stats['avg_response_ticks'] is not None else '—'} simulated minutes · Fully sheltered cases: {stats['sheltered_incidents']}")
        st.caption('Rescued = picked up by a boat; sheltered = admission confirmed by a shelter; staging = temporarily at command centre.')
    st.markdown('#### 📨 Agent communication · ACL log')
    logs=message_rows(m)
    c1,c2,c3=st.columns([2,2,1])
    kinds=sorted({item['Performative'] for item in logs})
    selected=c1.multiselect('Performative filter',kinds,default=kinds,key='acl_types')
    keyword=c2.text_input('Search sender / receiver / content',placeholder='e.g. CommandCenter or Boat-1',key='acl_search')
    limit=c3.selectbox('Rows',[25,50,100,300],index=1)
    filtered=[r for r in logs if r['Performative'] in selected and keyword.casefold() in ' '.join(str(v) for v in r.values()).casefold()][:limit]
    if filtered:
        st.dataframe(
            pd.DataFrame(filtered),
            hide_index=True,
            width='stretch',
            height=300,
            column_config={
                'Content': st.column_config.TextColumn('Content', width='large'),
                'Sender': st.column_config.TextColumn('Sender', width='medium'),
                'Receiver': st.column_config.TextColumn('Receiver', width='medium'),
            },
        )
        with st.expander('🔎 View full ACL message'):
            message_index = st.selectbox(
                'Select a message',
                range(len(filtered)),
                format_func=lambda i: (
                    f"Tick {filtered[i]['Tick']} | {filtered[i]['Sender']} → "
                    f"{filtered[i]['Receiver']} | {filtered[i]['Performative']}"
                ),
                key='acl_message_details',
            )
            chosen = filtered[message_index]
            st.write(f"**Ontology:** {chosen['Ontology']}")
            st.json(json.loads(chosen['Content']))
    else:
        st.info('No ACL messages match the current filters. Run or step the scenario to generate events.')
    st.download_button('⬇ Export all ACL messages (CSV)',data=pd.DataFrame(logs,columns=['Tick','Performative','Sender','Receiver','Ontology','Content']).to_csv(index=False).encode('utf-8-sig'),file_name=f'sayl_acl_tick_{m.tick}.csv',mime='text/csv',key='csv_download')
    with st.expander('Incident details and rescue assignments'):
        incidents=[]
        for inc in m.env.incidents.values():
            incidents.append({'ID':inc.incident_id,'Severity':inc.severity,'Awaiting pickup':inc.people,'Initial group':inc.total_people,'Rescued':inc.rescued_people,
                              'Sheltered':inc.sheltered_people,'At staging':inc.staged_people,'Status':inc.status,
                              'Location':str(inc.pos),'Drone':inc.drone or '—','Boat':inc.boat or '—',
                              'Shelter':inc.shelter or '—','Report':inc.text})
        if incidents: st.dataframe(pd.DataFrame(incidents),hide_index=True,width='stretch',column_config={'Report': st.column_config.TextColumn('Report',width='large')})
        else: st.caption('No incidents reported yet.')

with main_column:
    dashboard()
    st.divider()
    st.caption('SAYL · Multi-Agent Flood Rescue · CAI461 Agent-Based Systems · Educational simulation only')
