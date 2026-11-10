"""Checks for the Python-only Streamlit presentation layer."""

from pathlib import Path
import ast
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_has_no_custom_html_css_or_js():
    source = (ROOT / 'app.py').read_text(encoding='utf-8')
    ast.parse(source)
    assert 'unsafe_allow_html' not in source
    for snippet in ('<style', '</style>', '<div', '</div>', '<script', '</script>', '<span', '</span>'):
        assert snippet not in source
    theme_source = (ROOT / 'sayl' / 'ui_theme.py').read_text(encoding='utf-8')
    assert 'STYLES =' not in theme_source
    assert '<style' not in theme_source
    assert not any(
        p.suffix in {'.html', '.css', '.js', '.jsx', '.ts', '.tsx'}
        for p in ROOT.rglob('*') if p.is_file()
    )


def test_native_dashboard_and_acl_detail_panel_remain_available():
    source = (ROOT / 'app.py').read_text(encoding='utf-8')
    for token in (
        'st.container(border=True)',
        'col.metric(name,value,border=True)',
        "with st.expander('🔎 View full ACL message')",
        "st.json(json.loads(chosen['Content']))",
        'st.download_button',
        'st.fragment',
        'st.slider',
        'st.form_submit_button',
    ):
        assert token in source


def test_native_streamlit_color_settings_are_valid():
    config = tomllib.loads((ROOT / '.streamlit' / 'config.toml').read_text(encoding='utf-8'))
    theme = config['theme']
    assert theme['base'] == 'light'
    assert theme['primaryColor'] == '#087F8C'
    assert theme['textColor'] == '#17354A'
    assert theme['dataframeHeaderTextColor'] == '#17354A'


def test_dashboard_controls_are_on_right_not_in_sidebar():
    """Keep the requested compact right-hand native Streamlit layout."""
    source = (ROOT / 'app.py').read_text(encoding='utf-8')
    assert 'main_column, controls_column = st.columns(' in source
    assert 'with controls_column:' in source
    assert 'with main_column:' in source
    assert "initial_sidebar_state='collapsed'" in source
    assert 'with st.sidebar:' not in source
    for token in ('Apply scenario / Reset', 'Mission control', 'Rescue resources',
                  'Weather & advanced controls', 'Simulation controls'):
        assert token in source
