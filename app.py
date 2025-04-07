# -*- coding: UTF-8 -*-
"""
@filename: app.py
@author: JunLeon
@time: 2025-03-11
"""

import streamlit as st
import numpy as np
import json
import pickle
import os
import base64
from sklearn.preprocessing import StandardScaler


# ============================================
# 辅助函数
# ============================================

def img_to_base64(image_path):
    """将图片转换为Base64编码"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


def set_background_image(image_url, opacity):
    """设置背景图片"""
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            opacity: {opacity};
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# ============================================
# 数据加载函数 (使用缓存提高性能)
# ============================================

@st.cache_resource
def load_model_and_scaler():
    """加载模型和特征缩放器"""
    try:
        if not os.path.exists('model.pkl') or not os.path.exists('scaler.pkl'):
            raise FileNotFoundError("模型或scaler文件未找到")

        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)

        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        return model, scaler
    except Exception as e:
        st.error(f"加载模型失败: {str(e)}")
        st.stop()


@st.cache_resource
def load_teams_data():
    """加载队伍数据"""
    try:
        with open('merged_teams_data.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"加载队伍数据失败: {str(e)}")
        st.stop()


@st.cache_resource
def load_team_mapping():
    """加载队伍名称映射"""
    try:
        with open('teams.json', 'r') as f:
            teams_data = json.load(f)

        team_name_to_abbr = {}
        for abbr, team_info in teams_data.items():
            for team_name in team_info:
                team_name_to_abbr[team_name] = abbr
        return team_name_to_abbr
    except Exception as e:
        st.error(f"加载队伍映射失败: {str(e)}")
        st.stop()


# ============================================
# 预测函数
# ============================================

def predict_match(team1_abbr, team2_abbr, map_name, model, scaler, teams_data):
    """预测比赛结果"""
    try:
        team1 = teams_data[team1_abbr]
        team2 = teams_data[team2_abbr]

        # 计算全局平均评分
        all_ratings = [t.get("avg_rating", 0) for t in teams_data.values()]
        global_avg_rating = np.mean(all_ratings) if all_ratings else 0

        # 特征工程
        DEFAULT_WIN_RATE = 0.5

        # 基础特征
        map_diff = (
                team1["map_win_rate"].get(map_name, DEFAULT_WIN_RATE) -
                team2["map_win_rate"].get(map_name, DEFAULT_WIN_RATE)
        )

        # 队伍评分特征
        t1_rating = team1.get("avg_rating", global_avg_rating)
        t2_rating = team2.get("avg_rating", global_avg_rating)
        rating_diff = t1_rating - t2_rating

        # 全局评分等级特征
        t1_rating_level = t1_rating - global_avg_rating
        t2_rating_level = t2_rating - global_avg_rating
        rating_level_diff = t1_rating_level - t2_rating_level

        # 地图稳定性特征
        t1_rates = list(team1["map_win_rate"].values()) or [DEFAULT_WIN_RATE]
        t2_rates = list(team2["map_win_rate"].values()) or [DEFAULT_WIN_RATE]
        map_stability_diff = np.std(t1_rates) - np.std(t2_rates)

        features = [rating_diff, map_diff, rating_level_diff, map_stability_diff]

        # 特征标准化
        features_scaled = scaler.transform([features])

        # 预测
        prob = model.predict_proba(features_scaled)[0][1]

        return {
            "team1": team1_abbr,
            "team2": team2_abbr,
            "map": map_name,
            "win_prob": round(prob, 3),
            "confidence": "高可信度" if abs(prob - 0.5) > 0.3 else "需谨慎参考"
        }
    except Exception as e:
        return {"error": f"预测失败: {str(e)}"}


# ============================================
# 应用配置
# ============================================

# 设置背景
set_background_image(
    "https://wallpaperm.cmcm.com/85fff36cb2c02b45a8d26f4369dd36f5.jpg",
    0.85
)

# 定义赛区和队伍
regions = {
    "VCT CN": ["AG", "BLG", "EDG", "FPX", "JDG", "NOVA", "TE", "TEC", "TYL", "WOL", "DRG", "XLG"],
    "VCT EMEA": ["BBL", "FNC", "FUT", "GX", "KC", "KOI", "NAVI", "TH", "TL", "VIT", "M8", "APK"],
    "VCT PACIFIC": ["DFM", "DRX", "GEN", "GE", "PRX", "RRQ", "T1", "TLN", "TS", "ZETA", "NS", "BME"],
    "VCT AMERICA": ["100T", "C9", "EG", "FUR", "KRU", "LEV", "LOUD", "MIBR", "NRG", "SEN", "G2", "2G"],
}

# 定义地图
maps = ["Ascent", "Bind", "Breeze", "Fracture", "Haven", "Icebox", "Lotus", "Pearl", "Split", "Sunset"]

# 加载队伍图标 (示例，请替换为实际路径)
team_images = {
    team: f"data:image/png;base64,{img_to_base64(f'static/{team}.png')}"
    for team in sum(regions.values(), [])
}

# 自定义CSS
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        height: 60px;
        font-size: 20px;
        background-color: #FF0000;
        color: white;
        border: none;
        border-radius: 10px;
    }
    .stButton>button:hover {
        background-color: #CC0000;
    }
    .prediction-card {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .team-name {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .win-prob {
        font-size: 28px;
        font-weight: bold;
        color: #FF0000;
    }
    .confidence {
        font-size: 16px;
        color: #555555;
        margin-top: 10px;
    }
    .vs-text {
        font-size: 32px;
        font-weight: bold;
        text-align: center;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================
# 主应用逻辑
# ============================================

def main():
    """主应用函数"""

    # 初始化session state
    if "page" not in st.session_state:
        st.session_state.page = "home"

    # 加载数据
    model, scaler = load_model_and_scaler()
    teams_data = load_teams_data()
    team_mapping = load_team_mapping()

    # 主页
    if st.session_state.page == "home":
        show_home_page(model, scaler, teams_data)

    # 预测结果页
    elif st.session_state.page == "prediction":
        show_prediction_page(model, scaler, teams_data)


def show_home_page(model, scaler, teams_data):
    """显示主页"""
    st.title("无畏契约(VCT)比赛预测系统")

    # 初始化session_state中的选择
    if 'a_region' not in st.session_state:
        st.session_state.a_region = list(regions.keys())[0]
    if 'a_team' not in st.session_state:
        st.session_state.a_team = regions[st.session_state.a_region][0]
    if 'b_region' not in st.session_state:
        st.session_state.b_region = list(regions.keys())[0]
    if 'b_team' not in st.session_state:
        st.session_state.b_team = regions[st.session_state.b_region][1]
    if 'selected_map' not in st.session_state:
        st.session_state.selected_map = maps[0]

    # 队伍选择
    col1, col2, col3 = st.columns([4, 1, 4])

    # A队选择
    with col1:
        st.header("A队选择")
        st.session_state.a_region = st.selectbox(
            "选择A队的赛区",
            list(regions.keys()),
            index=list(regions.keys()).index(st.session_state.a_region),
            key="a_region_select"
        )

        # 更新A队选择
        available_a_teams = regions[st.session_state.a_region]
        if st.session_state.a_team not in available_a_teams:
            st.session_state.a_team = available_a_teams[0]

        st.session_state.a_team = st.selectbox(
            "选择A队的队伍",
            available_a_teams,
            index=available_a_teams.index(st.session_state.a_team),
            key="a_team_select"
        )

        if st.session_state.a_team in team_images:
            st.write(f"你选择的A队是: {st.session_state.a_team}")
            st.image(team_images[st.session_state.a_team], width=200)

    # VS标志
    with col2:
        st.header("")
        st.markdown("<h1 style='text-align: center;'>VS</h1>", unsafe_allow_html=True)

    # B队选择
    with col3:
        st.header("B队选择")
        st.session_state.b_region = st.selectbox(
            "选择B队的赛区",
            list(regions.keys()),
            index=list(regions.keys()).index(st.session_state.b_region),
            key="b_region_select"
        )

        # 过滤掉A队已选的队伍
        available_b_teams = [
            team for team in regions[st.session_state.b_region]
            if st.session_state.b_region != st.session_state.a_region or team != st.session_state.a_team
        ]

        # 更新B队选择
        if st.session_state.b_team not in available_b_teams:
            st.session_state.b_team = available_b_teams[0] if available_b_teams else None

        if available_b_teams:
            st.session_state.b_team = st.selectbox(
                "选择B队的队伍",
                available_b_teams,
                index=available_b_teams.index(
                    st.session_state.b_team) if st.session_state.b_team in available_b_teams else 0,
                key="b_team_select"
            )

            if st.session_state.b_team in team_images:
                st.write(f"你选择的B队是: {st.session_state.b_team}")
                st.image(team_images[st.session_state.b_team], width=200)
        else:
            st.warning("没有可选的队伍，请更改A队或赛区选择")

    # 地图选择
    st.header("比赛地图选择")
    st.session_state.selected_map = st.selectbox(
        "选择比赛地图",
        maps,
        index=maps.index(st.session_state.selected_map),
        key="map_select"
    )

    # 预测按钮
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("开始预测", key="predict_btn"):
            if st.session_state.a_team and st.session_state.b_team and st.session_state.selected_map:
                st.session_state.page = "prediction"
                st.rerun()
            else:
                st.warning("请确保已选择两支不同的队伍和一张地图")


def show_prediction_page(model, scaler, teams_data):
    """显示预测结果页"""
    st.title("预测结果")

    # 获取选择的队伍和地图
    a_team = st.session_state.get("a_team")
    b_team = st.session_state.get("b_team")
    selected_map = st.session_state.get("selected_map")

    if not all([a_team, b_team, selected_map]):
        st.warning("缺少必要信息，请返回重新选择")
        if st.button("返回"):
            st.session_state.page = "home"
            st.rerun()
        return

    # 执行预测
    with st.spinner("正在计算预测结果..."):
        result = predict_match(a_team, b_team, selected_map, model, scaler, teams_data)

    if "error" in result:
        st.error(result["error"])
        if st.button("返回"):
            st.session_state.page = "home"
            st.rerun()
        return

    # 显示预测结果
    st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)

    # 队伍对比
    col1, col2, col3 = st.columns([4, 2, 4])

    with col1:
        st.markdown(f"<div class='team-name'>{a_team}</div>", unsafe_allow_html=True)
        st.image(team_images[a_team], width=150)
        st.markdown(f"<div class='win-prob'>{result['win_prob'] * 100:.1f}%</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='vs-text'>VS</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align: center;'>地图: {selected_map}</div>", unsafe_allow_html=True)

    with col3:
        st.markdown(f"<div class='team-name'>{b_team}</div>", unsafe_allow_html=True)
        st.image(team_images[b_team], width=150)
        st.markdown(f"<div class='win-prob'>{(1 - result['win_prob']) * 100:.1f}%</div>", unsafe_allow_html=True)

    # 可信度
    st.markdown(f"<div class='confidence'>预测可信度: {result['confidence']}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # 返回按钮
    if st.button("返回继续预测"):
        st.session_state.page = "home"
        st.rerun()


if __name__ == "__main__":
    main()