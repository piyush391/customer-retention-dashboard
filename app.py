# =========================================================
# IMPORTS
# =========================================================
import streamlit as st
import pandas as pd
import networkx as nx

import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components

from preprocess import preprocess
from networkx.algorithms.community import greedy_modularity_communities
from data import load_processed
from model import load_model, predict
from metrics import compute_rsi


# =========================================================
# MAIN
# =========================================================

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Customer Retention Intelligence",
    page_icon="💡",
    layout="wide"
)

# =========================================================
# GLOBAL CSS
# =========================================================
st.markdown("""
<style>

.stApp {
    background:
        linear-gradient(
            135deg,
            #050816,
            #0b1120,
            #111827,
            #0f172a
        );
    color: white;
}

.main,
.block-container,
[data-testid="stAppViewContainer"] {
    background: transparent !important;
}

[data-testid="metric-container"] {

    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);

    border-radius: 18px;
    padding: 16px;

    border: 1px solid rgba(255,255,255,0.08);

    box-shadow:
        0 0 20px rgba(0,195,255,0.08);
}

section[data-testid="stSidebar"] {
    background: rgba(5,10,20,0.92);
}

.stTabs [data-baseweb="tab"] {

    background: rgba(255,255,255,0.04);

    border-radius: 12px;

    padding: 10px 18px;

    margin-right: 6px;
}

.stTabs [aria-selected="true"] {

    background: rgba(0,195,255,0.12);

    border: 1px solid rgba(0,195,255,0.25);
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# VANTA HEADER
# =========================================================
vanta_html = """
<!DOCTYPE html>
<html>

<head>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r121/three.min.js"></script>

<script src="https://cdn.jsdelivr.net/npm/vanta@latest/dist/vanta.net.min.js"></script>

<style>

html, body {

    margin: 0;
    padding: 0;
    overflow: hidden;

    font-family: Arial, sans-serif;
}

#vanta-bg {

    width: 100vw;
    height: 350px;

    position: relative;

    border-radius: 20px;

    overflow: hidden;
}

.overlay {

    position: absolute;

    top: 50%;
    left: 50%;

    transform: translate(-50%, -50%);

    text-align: center;

    color: white;

    z-index: 10;

    width: 100%;
}

.overlay h1 {

    font-size: 52px;

    margin-bottom: 14px;

    font-weight: 800;

    letter-spacing: 1px;
}

.overlay p {

    font-size: 22px;

    opacity: 0.9;
}

</style>

</head>

<body>

<div id="vanta-bg">

    <div class="overlay">

        <h1>
            Customer Retention Intelligence
        </h1>

        <p>
            AI-Powered Banking Engagement & Product Utilization Analytics
        </p>

    </div>

</div>

<script>

VANTA.NET({

    el: "#vanta-bg",

    mouseControls: true,
    touchControls: true,
    gyroControls: false,

    color: 0x00c3ff,
    backgroundColor: 0x050816,

    points: 12,
    maxDistance: 22,
    spacing: 18
})

</script>

</body>
</html>
"""

components.html(
    vanta_html,
    height=350,
    scrolling=False
)

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_data():

    raw_df = pd.read_csv("European_Bank.csv")

    processed_df = preprocess(raw_df)

    return raw_df, processed_df


# =========================================================
# PREDICTIONS
# =========================================================
@st.cache_data
def generate_predictions(df_processed):

    model = load_model()

    X = df_processed.drop("Exited", axis=1)

    df_processed["ChurnProb"] = model.predict_proba(X)[:, 1]

    return df_processed

# =========================================================
# LOAD EVERYTHING
# =========================================================
df, df_processed = load_data()

df_processed = generate_predictions(df_processed)

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("Dashboard Controls")

balance_threshold = st.sidebar.slider(
    "Balance Threshold",
    0,
    int(df["Balance"].max()),
    100000
)

product_filter = st.sidebar.slider(
    "Minimum Products",
    1,
    4,
    1
)

geo_filter = st.sidebar.multiselect(
    "Geography",
    options=df["Geography"].unique(),
    default=df["Geography"].unique()
)

# =========================================================
# FILTER DATA
# =========================================================
filtered_df = df_processed[
    (df_processed["Balance"] >= balance_threshold) &
    (df_processed["NumOfProducts"] >= product_filter) &
    (df["Geography"].isin(geo_filter))
].copy()

# =========================================================
# EMPTY FILTER CHECK
# =========================================================
if filtered_df.empty:

    st.error("""
    No customers available under current filter selection.

    Reduce:
    - balance threshold
    - product threshold
    - or expand geography selection
    """)

    st.stop()

# =========================================================
# KPI SECTION (WITH INDIVIDUAL DROPDOWNS)
# =========================================================

k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.metric("Customers", len(filtered_df))
    with st.expander("What is this?"):
        st.markdown("""
        Total number of customers in the current filter scope.  
        This defines the dataset size used for all analytics and segmentation.
        """)

with k2:
    churn_rate = filtered_df["Exited"].mean() * 100
    st.metric("Churn Rate", f"{churn_rate:.2f}%")
    with st.expander("What is this?"):
        st.markdown("""
        Percentage of customers who have exited the bank.  
        A key retention health indicator — higher values signal weak engagement or satisfaction issues.
        """)

with k3:
    st.metric("Average RSI", f"{filtered_df['RSI'].mean():.2f}")
    with st.expander("What is this?"):
        st.markdown("""
        Relationship Strength Index (RSI) measures customer engagement depth.  
        Higher RSI = stronger loyalty, more product usage, and lower churn risk.
        """)

with k4:
    st.metric("Avg Churn Risk", f"{filtered_df['ChurnProb'].mean():.2f}")
    with st.expander("What is this?"):
        st.markdown("""
        Machine learning predicted probability of churn.  
        This is a forward-looking risk score used for proactive retention targeting.
        """)

with k5:
    premium_count = int(filtered_df["PremiumCustomer"].sum())
    st.metric("Premium Customers", premium_count)
    with st.expander("What is this?"):
        st.markdown("""
        High-value customers based on balance and financial behavior.  
        Losing these customers has a disproportionately high revenue impact.
        """)

# =========================================================
# CHART FUNCTION
# =========================================================
def show_chart(fig, key_name):

    fig.update_layout(
        height=550,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white")
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key=key_name
    )

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Engagement Analytics",
    "Product Utilization",
    "Premium Risk",
    "Retention Engine",
    "Network Intelligence",
    "Executive Insights"
])

# =========================================================
# TAB 1
# =========================================================
with tab1:

    st.subheader("Activity Status vs Churn Risk")

    st.info("""
    PURPOSE

    Measures effect of customer activity
    on churn probability.

    BUSINESS USE

    Helps identify disengaged customers
    requiring intervention.
    """)

    engagement_data = filtered_df.groupby(
        "IsActiveMember"
    )["ChurnProb"].mean().reset_index()

    engagement_data["Status"] = engagement_data[
        "IsActiveMember"
    ].map({
        0: "Inactive",
        1: "Active"
    })

    fig_engagement = px.bar(
        engagement_data,
        x="Status",
        y="ChurnProb",
        color="Status",
        text_auto=".2f",
        title="Average Churn Risk by Activity"
    )

    show_chart(
        fig_engagement,
        "engagement_chart"
    )

    active_data = engagement_data.loc[
    engagement_data["Status"] == "Active",
    "ChurnProb"
    ]

    inactive_data = engagement_data.loc[
        engagement_data["Status"] == "Inactive",
        "ChurnProb"
    ]

    if len(active_data) > 0 and len(inactive_data) > 0:

        active_risk = active_data.values[0]
        inactive_risk = inactive_data.values[0]

        gap = inactive_risk - active_risk

        st.success(f"""
        LIVE INSIGHT

        Inactive customers currently show
        higher churn probability than active customers.

        Current Risk Gap:
        {gap:.2f}
        """)

    else:

        st.warning("""
        Not enough active/inactive customer data
        under current filters.
        """)
        # =====================================================
        # RSI DISTRIBUTION
        # =====================================================

    st.subheader(
            "Relationship Strength Distribution"
        )

    st.info("""
        PURPOSE

        Visualizes customer relationship strength
        across the selected population.

        BUSINESS USE

        Higher RSI indicates:
        - stronger engagement
        - deeper product adoption
        - greater loyalty

        STRATEGIC VALUE

        Detects weak relationship segments
        before customer exit occurs.
        """)

    fig_rsi_distribution = px.histogram(
            filtered_df,
            x="RSI",
            nbins=40,
            title="Relationship Strength Distribution",
            color_discrete_sequence=["#00c3ff"]
        )

    fig_rsi_distribution.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            height=550
        )

    show_chart(
            fig_rsi_distribution,
            "rsi_distribution_unique"
        )
    
# =====================================================
# LIVE INSIGHT — RSI DISTRIBUTION
# =====================================================

    if len(filtered_df) > 0:

        low_rsi = filtered_df[filtered_df["RSI"] < 0.30]
        mid_rsi = filtered_df[(filtered_df["RSI"] >= 0.30) & (filtered_df["RSI"] < 0.70)]
        high_rsi = filtered_df[filtered_df["RSI"] >= 0.70]

        low_risk_rate = low_rsi["Exited"].mean() if len(low_rsi) > 0 else 0
        high_risk_rate = high_rsi["Exited"].mean() if len(high_rsi) > 0 else 0

        st.success(f"""
        LIVE INSIGHT

        RSI SEGMENTATION SIGNAL

        Low RSI Customers (<0.30):
        {len(low_rsi)} customers | Churn Rate: {low_risk_rate:.2%}

        High RSI Customers (>0.70):
        {len(high_rsi)} customers | Churn Rate: {high_risk_rate:.2%}

        STRATEGIC INTERPRETATION

        Customers with low RSI show significantly
        higher churn vulnerability.

        ACTION:

        Prioritize:
        - onboarding reinforcement
        - engagement campaigns
        - product cross-sell for low RSI segment
        """)
        # =====================================================
        # RSI RISK ANALYSIS
        # =====================================================

    st.subheader(
            "RSI Risk Threshold Analysis"
        )

    st.info("""
        PURPOSE

        Identifies RSI levels where churn risk rises sharply.

        BUSINESS USE

        Helps detect:
        - silent churn zones
        - disengagement patterns
        - weak customer relationships

        STRATEGIC VALUE

        Enables proactive intervention before churn escalation.
        """)

    rsi_risk = filtered_df.copy()

    if len(rsi_risk) > 0:

            rsi_risk["RSI_Band"] = pd.cut(
                rsi_risk["RSI"],
                bins=10
            )

            rsi_group = rsi_risk.groupby(
                "RSI_Band"
            )["Exited"].mean().reset_index()

            rsi_group["RSI_Label"] = rsi_group[
                "RSI_Band"
            ].astype(str)

            fig_rsi_risk = px.line(
                rsi_group,
                x="RSI_Label",
                y="Exited",
                markers=True,
                title="Churn Rate Across RSI Levels"
            )

            fig_rsi_risk.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                height=550
            )

            show_chart(
                fig_rsi_risk,
                "rsi_risk_unique"
            )

            high_risk = rsi_group[
                rsi_group["Exited"] > 0.40
            ]

            if not high_risk.empty:

                st.error(f"""
                LIVE INSIGHT

                High churn-risk RSI ranges detected:

                {', '.join(high_risk['RSI_Label'].astype(str).tolist())}

                BUSINESS INTERPRETATION

                Weak relationship strength
                strongly correlates with customer exits.
                """)

    else:

        st.warning("""
            RSI analysis unavailable
            under current filter conditions.
            """)

# =========================================================
# TAB 2 — PRODUCT UTILIZATION
# =========================================================
with tab2:

    st.subheader(
        "Product Utilization Intelligence"
    )

    st.info("""
    PURPOSE

    Evaluates how banking product adoption
    influences churn behavior.

    BUSINESS USE

    Helps identify:
    - weak product engagement
    - cross-sell opportunities
    - retention-strengthening segments

    STRATEGIC VALUE

    Multi-product customers typically
    demonstrate stronger loyalty.
    """)

    # =====================================================
    # PRODUCT COUNT ANALYSIS
    # =====================================================

    product_data = filtered_df.groupby(
        "NumOfProducts"
    )["Exited"].mean().reset_index()

    if len(product_data) > 0:

        fig_product = px.line(
            product_data,
            x="NumOfProducts",
            y="Exited",
            markers=True,
            title="Churn Rate by Product Count"
        )

        fig_product.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            height=550
        )

        show_chart(
            fig_product,
            "product_chart_unique"
        )

        # =================================================
        # LIVE INSIGHT
        # =================================================

        if len(product_data) > 1:

            lowest_churn = product_data.loc[
                product_data["Exited"].idxmin()
            ]

            st.success(f"""
            LIVE INSIGHT

            Customers with
            {int(lowest_churn['NumOfProducts'])}
            products currently show
            the lowest churn rate.

            STRATEGIC INTERPRETATION

            Product depth appears strongly linked
            to customer retention stability.
            """)

    else:

        st.warning("""
        Product utilization analysis unavailable
        under current filter conditions.
        """)

    # =====================================================
    # CREDIT CARD ANALYSIS
    # =====================================================

    st.subheader(
        "Credit Card Stickiness Analysis"
    )

    st.info("""
    PURPOSE

    Measures retention influence
    of credit card ownership.

    BUSINESS USE

    Credit card customers often show:
    - stronger daily engagement
    - deeper banking dependency
    - higher switching resistance

    STRATEGIC VALUE

    Helps evaluate loyalty impact
    of card-based products.
    """)

    card_data = filtered_df.groupby(
        "HasCrCard"
    )["Exited"].mean().reset_index()

    card_data["CardStatus"] = card_data[
        "HasCrCard"
    ].map({
        0: "No Card",
        1: "Has Card"
    })

    if len(card_data) > 0:

        fig_card = px.bar(
            card_data,
            x="CardStatus",
            y="Exited",
            color="CardStatus",
            text_auto=".2f",
            title="Retention Impact of Credit Card Ownership"
        )

        fig_card.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            height=550
        )

        show_chart(
            fig_card,
            "card_chart_unique"
        )

        # ================================================
        # LIVE INSIGHT
        # ================================================

        card_risk = card_data.loc[
            card_data["CardStatus"] == "Has Card",
            "Exited"
        ]

        no_card_risk = card_data.loc[
            card_data["CardStatus"] == "No Card",
            "Exited"
        ]

        if (
            len(card_risk) > 0 and
            len(no_card_risk) > 0
        ):

            difference = (
                no_card_risk.values[0] -
                card_risk.values[0]
            )

            st.success(f"""
            LIVE INSIGHT

            Customers without credit cards
            currently show higher churn behavior.

            Current Churn Difference:
            {difference:.2f}

            STRATEGIC INTERPRETATION

            Card ownership may strengthen
            long-term banking engagement.
            """)

    else:

        st.warning("""
        Credit card analysis unavailable
        under current filters.
        """)
# =========================================================
# TAB 4 — PREMIUM RISK
# =========================================================
with tab3:

    st.subheader(
        "High-Value Disengaged Customer Detection"
    )

    st.info("""
    PURPOSE

    Detects financially valuable customers
    with weak engagement behavior.

    BUSINESS USE

    Premium inactive customers represent
    potential silent churn risk.

    STRATEGIC VALUE

    Enables targeted intervention
    for high-value banking clients.
    """)

    premium = filtered_df[
        (filtered_df["PremiumCustomer"] == 1) &
        (filtered_df["IsActiveMember"] == 0)
    ]

    st.metric(
        "High-Risk Premium Customers",
        len(premium)
    )

    # =====================================================
    # SAFE EMPTY CHECK
    # =====================================================

    if len(premium) > 0:

        fig_premium = px.scatter(
            premium,
            x="Balance",
            y="ChurnProb",
            size="RSI",
            color="Exited",
            hover_data=[
                "NumOfProducts",
                "EstimatedSalary"
            ],
            title="Premium Customer Risk Map"
        )

        show_chart(
            fig_premium,
            "premium_chart_unique"
        )

        top_risk = premium["ChurnProb"].max()

        avg_risk = premium["ChurnProb"].mean()

        st.error(f"""
        LIVE INSIGHT

        Premium customer churn risk
        currently reaches as high as:

        {top_risk:.2f}

        Average Premium Risk:
        {avg_risk:.2f}

        INTERPRETATION

        Some high-value customers
        show severe disengagement signals.
        """)

    else:

        st.warning("""
        No premium inactive customers
        available under current filters.

        Try:
        - lowering balance threshold
        - expanding geography selection
        - reducing product filter
        """)
# =========================================================
# TAB 4
# =========================================================
with tab4:

    st.subheader(
        "Retention Recommendation Engine"
    )

    def recommend(row):

        if row["RSI"] < 0.30:
            return "Immediate Retention"

        elif row["NumOfProducts"] == 1:
            return "Cross-Sell"

        elif row["IsActiveMember"] == 0:
            return "Reactivation"

        else:
            return "Loyalty Program"

    filtered_df["Recommendation"] = filtered_df.apply(
        recommend,
        axis=1
    )

    fig_retention = px.scatter(
        filtered_df,
        x="RSI",
        y="ChurnProb",
        color="Recommendation",
        title="Retention Segmentation"
    )

    show_chart(
        fig_retention,
        "retention_chart"
    )

    st.success("""
    LIVE INSIGHT

    Customers with weak relationship strength
    are concentrated in high churn zones.

    STRATEGIC USE

    Enables proactive churn prevention
    before customer exit.
    """)

# =========================================================
# NETWORK GRAPH FUNCTION 
# =========================================================
    @st.cache_data(show_spinner=False)
    def build_network_graph(df, sample_size):

        if len(df) < 30:
            return None, None

        sample_n = min(sample_size, len(df))

        if sample_n <= 1:
            return None, None

        df_sample = df.sample(n=sample_n, random_state=42)

        # =====================================================
        # GRAPH INIT
        # =====================================================
        G = nx.Graph()

        for idx, row in df_sample.iterrows():
            G.add_node(
                idx,
                RSI=row["RSI"],
                Churn=row["ChurnProb"],
                Products=row["NumOfProducts"],
                Balance=row["Balance"]
            )

        nodes = list(G.nodes())

        # =====================================================
        # SIMILARITY FUNCTION
        # =====================================================
        def similarity(n1, n2):

            rsi_d = abs(n1["RSI"] - n2["RSI"])
            churn_d = abs(n1["Churn"] - n2["Churn"])
            prod_d = abs(n1["Products"] - n2["Products"])

            return (
                rsi_d * 0.6 +
                churn_d * 0.3 +
                (prod_d / 3) * 0.1
            )

        # =====================================================
        # EDGE CREATION 
        # =====================================================
        max_edges_per_node = 3
        edge_count = {n: 0 for n in nodes}

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):

                n1 = G.nodes[nodes[i]]
                n2 = G.nodes[nodes[j]]

                sim = similarity(n1, n2)

                if sim < 0.10:

                    if edge_count[nodes[i]] < max_edges_per_node and edge_count[nodes[j]] < max_edges_per_node:

                        G.add_edge(nodes[i], nodes[j])
                        edge_count[nodes[i]] += 1
                        edge_count[nodes[j]] += 1

        # fallback connectivity
        if len(G.edges()) == 0:
            for i in range(len(nodes) - 1):
                G.add_edge(nodes[i], nodes[i + 1])

        # =====================================================
        # COMMUNITY DETECTION
        # =====================================================
        communities = list(greedy_modularity_communities(G))

        community_map = {}
        for i, community in enumerate(communities):
            for node in community:
                community_map[node] = i

        # =====================================================
        # LAYOUT
        # =====================================================
        pos = nx.spring_layout(G, dim=3, seed=42)

        # =====================================================
        # EDGE TRACE
        # =====================================================
        edge_x, edge_y, edge_z = [], [], []

        for edge in G.edges():

            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]

            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
            edge_z += [z0, z1, None]

        edge_trace = go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode="lines",
            line=dict(width=1, color="rgba(255,255,255,0.06)"),
            hoverinfo="none"
        )

       
# =====================================================
# NODE TRACE
# =====================================================
        node_x, node_y, node_z = [], [], []
        node_size = []
        node_color = []
        hover_text = []

        for node in G.nodes():

            x, y, z = pos[node]
            data = G.nodes[node]

            node_x.append(x)
            node_y.append(y)
            node_z.append(z)

            rsi = data["RSI"]

            # size scaling
            node_size.append((rsi * 18) + 5)

            # color = RSI (normalized later by plotly)
            node_color.append(rsi)

            hover_text.append(
                f"""
                Customer ID: {node}
                <br>RSI: {rsi:.2f}
                <br>Churn Risk: {data['Churn']:.2f}
                <br>Products: {data['Products']}
                <br>Balance: {data['Balance']:.0f}
                """
            )

        node_trace = go.Scatter3d(
            x=node_x,
            y=node_y,
            z=node_z,
            mode="markers",
            hoverinfo="text",
            hovertext=hover_text,
            marker=dict(
                size=node_size,

                # 🔥 THIS IS THE KEY FIX
                color=node_color,
                colorscale="Viridis",
                cmin=min(node_color),
                cmax=max(node_color),

                opacity=0.9,
                line=dict(width=0.4, color="rgba(255,255,255,0.6)")
            )
        )

        # =====================================================
        # FIGURE
        # =====================================================
        fig = go.Figure(data=[edge_trace, node_trace])

        fig.update_layout(
            title="Behavioral Customer Network Intelligence",
            height=750,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", size=12),
            showlegend=False,
            margin=dict(l=0, r=0, t=60, b=0),
            scene=dict(
                bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="Behavioral Similarity", color="white"),
                yaxis=dict(title="Customer Engagement", color="white"),
                zaxis=dict(title="Retention Risk", color="white"),
                camera=dict(eye=dict(x=1.4, y=1.4, z=1.1))
            )
        )

        return fig, G
    

# =========================================================
# TAB 5 — NETWORK INTELLIGENCE
# ========================================================= 

with tab5:
    
    st.subheader(
        "Behavioral Network Intelligence"
    )

    st.info("""
    PURPOSE

    Visualizes behavioral similarity clusters
    across customers using relationship strength,
    churn probability, and product utilization.

    BUSINESS USE

    Helps identify:
    - isolated customers
    - hidden behavioral segments
    - vulnerable relationship clusters

    STRATEGIC VALUE

    Supports advanced retention intelligence
    and customer segmentation analysis.
    """)

    st.subheader("Network Intelligence Guide")

    view = st.selectbox(
        "Select explanation layer",
        [
            "What am I seeing?",
            "How nodes are connected",
            "What edges mean",
            "How risk is calculated",
            "Business interpretation"
        ]
    )    
    if view == "What am I seeing?":

        st.info("""
        This is a behavioral customer network.

        Each node = one customer
        Connections = similarity in behavior (RSI, churn risk, product usage)

        The goal is to detect hidden customer clusters.
        """)

    elif view == "How nodes are connected":

        st.info("""
        Nodes are connected using a similarity score:

        similarity =
            RSI difference (60%)
            + Churn probability difference (30%)
            + Product usage difference (10%)

        Only customers with high similarity are linked.
        """)

    elif view == "What edges mean":

        st.info("""
        Each line represents behavioral closeness.

        Thicker clustering = similar customer behavior.

        No line = weak or no behavioral similarity.
        """)

    elif view == "How risk is calculated":

        st.info("""
        Risk is derived from:
        - RSI (Relationship Strength Index)
        - ML churn probability
        - Product engagement level

        Lower RSI + higher churn probability = higher risk zone.
        """)

    elif view == "Business interpretation":

        st.success("""
        KEY INSIGHT

        Customers form behavioral clusters.

        Isolated or weakly connected nodes = high churn vulnerability.

        ACTION:
        - Target isolated customers
        - Strengthen weak clusters
        - Improve product penetration
        """)
    # =====================================================
    # DATA AVAILABILITY CHECK
    # =====================================================

    if len(filtered_df) < 15:

        st.warning(f"""
        Network Intelligence Unavailable

        Current filtered dataset contains only
        {len(filtered_df)} customers.

        The behavioral network model requires
        a larger customer population to generate
        meaningful relationship clusters.

        Try:
        - lowering balance threshold
        - reducing product filter
        - expanding geography selection
        """)

    else:

        sample_size = st.slider(
            "Network Sample Size",
            30,
            min(120, len(filtered_df)),
            min(60, len(filtered_df))
        )

        fig_network, G = build_network_graph(
            filtered_df,
            sample_size
        )

        # =================================================
        # SAFE GRAPH CHECK
        # =================================================

        if fig_network is not None and G is not None:

            show_chart(
                fig_network,
                "network_chart"
            )

            isolated = list(nx.isolates(G))

            st.metric(
                "Behaviorally Isolated Customers",
                len(isolated)
            )

            st.success(f"""
            LIVE INSIGHT

            Network analysis detected
            {len(isolated)} isolated customers.

            INTERPRETATION

            Weakly connected behavioral clusters
            may indicate elevated churn vulnerability
            and low engagement stability.
            """)

        else:

            st.warning("""
            Network graph could not be generated
            under current filtering conditions.

            This usually occurs when:
            - customer similarity is too low
            - dataset becomes highly fragmented
            - filter constraints are too strict

            Try relaxing dashboard filters.
            """)

        
# =========================================================
# TAB 6
# =========================================================
with tab6:

    st.subheader(
        "Executive Retention Insights"
    )

    active_churn = filtered_df[
        filtered_df["IsActiveMember"] == 1
    ]["Exited"].mean()

    inactive_churn = filtered_df[
        filtered_df["IsActiveMember"] == 0
    ]["Exited"].mean()

    multi_product = filtered_df[
        filtered_df["NumOfProducts"] > 1
    ]["Exited"].mean()

    single_product = filtered_df[
        filtered_df["NumOfProducts"] == 1
    ]["Exited"].mean()

    st.success(f"""
    EXECUTIVE INSIGHT

    Active Customer Churn:
    {active_churn*100:.2f}%

    Inactive Customer Churn:
    {inactive_churn*100:.2f}%

    STRATEGIC INTERPRETATION

    Engagement remains a primary
    retention driver.
    """)

    st.info(f"""
    PRODUCT DEPTH INSIGHT

    Single Product Churn:
    {single_product*100:.2f}%

    Multi Product Churn:
    {multi_product*100:.2f}%

    STRATEGIC INTERPRETATION

    Product depth strongly improves
    customer retention stability.
    """)