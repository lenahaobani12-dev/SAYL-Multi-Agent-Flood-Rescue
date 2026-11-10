"""Static safety regression check for the Streamlit dashboard source."""
from pathlib import Path

def test_dashboard_uses_real_model_and_required_panels():
    src=(Path(__file__).resolve().parents[1]/'app.py').read_text(encoding='utf-8')
    for token in ('SaylModel','render_map','closed_road_map','agent_rows','message_rows','st.download_button','st.slider','st.fragment','submit_message','scenario_complete'):
        assert token in src
